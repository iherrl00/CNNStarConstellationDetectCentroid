import os
import sys
import time
import json
import math
import webbrowser
import torch
import cv2
import numpy as np
import requests
from pathlib import Path
from astropy.io import fits
from torchvision.transforms import Normalize

# Configuración de rutas dinámicas para asegurar portabilidad.
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
TRAINING_DIR = SRC_DIR / "training"

sys.path.insert(0, str(TRAINING_DIR))
# Estadísticas de normalización validadas para el dataset V2
MEAN_V2 = [5.9355394]
STD_V2 = [12.10830327]


def resolve_path(path_value):
    """Resuelve rutas relativas primero desde este módulo y luego desde el cwd."""
    path = Path(path_value)
    if path.is_absolute():
        return path

    candidate = CURRENT_DIR / path
    if candidate.exists():
        return candidate

    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate

    return path


class StarTrackerPipelineV2:
    """
    Pipeline de detección, envío a Astrometry y reporte final.
    """

    def __init__(self, model_path, api_key="wnmzxqdgrsmercxg"):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.api_key = api_key
        self.api_url = "http://nova.astrometry.net/api"
        self.session = None
        self.last_pixscale_arcsec = None

        print(f"[Modelo] Cargando en {self.device}...")
        self.model = torch.load(
            model_path, map_location=self.device, weights_only=False)
        self.model.eval()
        self.norm = Normalize(MEAN_V2, STD_V2)

    def process(self, image_path):
        """
        Ejecuta el flujo completo: IA -> Top 40 -> FITS -> Astrometry
        """
        start_time = time.time()
        image_path = resolve_path(image_path)
        img_orig = cv2.imread(str(image_path))
        if img_orig is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")

        img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
        h, w = img_gray.shape

        print(f"[Proceso] Imagen detectada: {w}x{h}")
        centroids = self._tiling_inference(img_gray)
        print(f"[Proceso] Estrellas detectadas: {len(centroids)}")

        # Selección de estrellas para Astrometry
        if len(centroids) < 200:
            stars_for_astrometry = self._photometric_filter(
                centroids, img_gray, top_k=None)
            print(f"[Filtro] Menos de 200 estrellas; se envían todas.")
        else:
            stars_for_astrometry = self._photometric_filter(
                centroids, img_gray, top_k=40)
            print(f"[Filtro] Se enviarán las 40 más brillantes.")

        fits_path = image_path.with_name(image_path.stem + "_clean_sky.fits")
        self._generate_fits_image(stars_for_astrometry, h, w, fits_path)

        results = self._solve_plate(fits_path, w, h)

        self._print_report(len(centroids), results, time.time() - start_time)
        if results and results.get("status") == "success":
            self._open_astrometry_web_view(results)
        return results

    def _tiling_inference(self, img_gray):
        """Divide la imagen en parches de 480x480 con solape."""
        h, w = img_gray.shape
        tile_size = 480
        overlap = 64
        stride = tile_size - overlap

        all_centroids = []

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)

                patch = img_gray[y:y_end, x:x_end]
                ph, pw = patch.shape

                # Relleno para mantener entrada constante de 480x480
                input_patch = np.pad(
                    patch, ((0, tile_size - ph), (0, tile_size - pw)), mode='constant')

                patch_tensor = torch.from_numpy(input_patch).float(
                ).unsqueeze(0).unsqueeze(0).to(self.device)
                patch_norm = self.norm(patch_tensor)

                with torch.no_grad():
                    pred = self.model(patch_norm)
                    prob_map = torch.sigmoid(pred[0, 0]).cpu().numpy()

                seg_map = (prob_map > 0.5).astype(np.uint8)[:ph, :pw]

                # Extracción por componentes conectadas
                contours, _ = cv2.findContours(
                    seg_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                margin = overlap // 2
                for cnt in contours:
                    m = cv2.moments(cnt)
                    if m["m00"] > 1e-8:
                        cx, cy = m["m10"]/m["m00"], m["m01"]/m["m00"]

                        # Zona de responsabilidad del parche
                        is_valid = True
                        if x > 0 and cx < margin:
                            is_valid = False
                        if y > 0 and cy < margin:
                            is_valid = False
                        if x_end < w and cx >= (tile_size - margin):
                            is_valid = False
                        if y_end < h and cy >= (tile_size - margin):
                            is_valid = False

                        if is_valid:
                            all_centroids.append((cx + x, cy + y))

        # Supresión por proximidad
        final_list = []
        for c in all_centroids:
            if all(math.hypot(c[0]-f[0], c[1]-f[1]) > 8 for f in final_list):
                final_list.append(c)
        return final_list

    def _photometric_filter(self, centroids, img_gray, top_k=40):
        """Ordena detecciones por intensidad local."""
        stars = []
        h, w = img_gray.shape
        for cx, cy in centroids:
            ix, iy = int(round(cx)), int(round(cy))
            window = img_gray[max(0, iy-2):min(h, iy+3),
                              max(0, ix-2):min(w, ix+3)]
            brightness = np.sum(window)
            stars.append({'x': cx, 'y': cy, 'flux': brightness})

        stars.sort(key=lambda s: s['flux'], reverse=True)
        if top_k is None:
            return stars
        return stars[:top_k]

    def _generate_fits_image(self, stars, h, w, output_path):
        """Genera el FITS de entrada para Astrometry."""
        canvas = np.zeros((h, w), dtype=np.float32)
        for s in stars:
            ix, iy = int(round(s['x'])), int(round(s['y']))
            if 0 <= iy < h and 0 <= ix < w:
                canvas[iy, ix] = s['flux']

        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

        hdu = fits.PrimaryHDU(canvas)
        if output_path.exists():
            os.remove(output_path)
        hdu.writeto(str(output_path))
        print(f"[Salida] FITS generado: {output_path.name}")

    def _open_astrometry_web_view(self, results):
        """Abre la visualización anotada pública de Astrometry en el navegador."""
        job_id = results.get("job_id")
        if not job_id:
            return

        url = f"http://nova.astrometry.net/annotated_display/{job_id}"
        try:
            webbrowser.open(url, new=2)
            print(f"[Visual] Vista web: {url}")
        except Exception as e:
            print(f"[Visual] No se pudo abrir la vista web: {e}")

    def _build_upload_request_json(self, use_scale_hint):
        """Construye request-json para upload con hint opcional de escala."""
        request_json = {
            "session": self.session,
            "crpix_center": True,
            "publicly_visible": "n"
        }

        if use_scale_hint and self.last_pixscale_arcsec is not None:
            scale_lower = max(0.1, self.last_pixscale_arcsec * 0.6)
            scale_upper = min(3600.0, self.last_pixscale_arcsec * 1.6)
            request_json.update({
                "scale_units": "arcsecperpix",
                "scale_lower": scale_lower,
                "scale_upper": scale_upper,
            })
            mode_desc = f"escala estimada ({scale_lower:.2f}-{scale_upper:.2f} arcsec/px)"
        else:
            mode_desc = "sin escala predefinida"

        return request_json, mode_desc

    def _solve_plate(self, fits_path, w, h):
        """Envía la imagen a Astrometry y espera el resultado."""
        print("[Astrometry] Conectando...")
        try:
            # Login
            r = requests.post(
                f"{self.api_url}/login", data={"request-json": json.dumps({"apikey": self.api_key})})
            self.session = r.json().get("session")

            attempts = [False]
            if self.last_pixscale_arcsec is not None:
                attempts = [True, False]

            last_result = {
                "status": "timeout",
                "subid": None,
                "job_id": None,
                "cal": {},
                "objs": []
            }

            for idx, use_scale_hint in enumerate(attempts, start=1):
                upload_request, mode_desc = self._build_upload_request_json(
                    use_scale_hint)
                print(f"[Astrometry] Intento {idx}/{len(attempts)}: {mode_desc}")

                with open(fits_path, 'rb') as f:
                    r = requests.post(
                        f"{self.api_url}/upload",
                        files={'file': f},
                        data={"request-json": json.dumps(upload_request)}
                    )

                subid = r.json().get("subid")
                if not subid:
                    print("[Astrometry] Error: upload sin subid.")
                    last_result = {
                        "status": "upload-error",
                        "subid": None,
                        "job_id": None,
                        "cal": {},
                        "objs": []
                    }
                    continue

                print(f"[Astrometry] Subida exitosa (ID: {subid}). Resolviendo", end="", flush=True)

                # 1) Esperar job_id
                job_id = None
                for _ in range(30):  # Timeout ~150s para asignación de job
                    time.sleep(5)
                    print(".", end="", flush=True)
                    r = requests.get(f"{self.api_url}/submissions/{subid}")
                    jobs = [job for job in r.json().get("jobs", []) if job]
                    if jobs:
                        job_id = jobs[0]
                        break

                if not job_id:
                    print("\n[Astrometry] Timeout: no se obtuvo job_id.")
                    last_result = {
                        "status": "timeout",
                        "subid": subid,
                        "job_id": None,
                        "cal": {},
                        "objs": []
                    }
                    continue

                # 2) Esperar status final del job
                job_status = None
                for _ in range(60):  # Timeout ~300s para solución final
                    time.sleep(5)
                    print(".", end="", flush=True)
                    r_job = requests.get(f"{self.api_url}/jobs/{job_id}")
                    job_status = r_job.json().get("status")
                    if job_status in {"success", "failure"}:
                        break

                if job_status != "success":
                    print(f"\n[Astrometry] Estado final del job {job_id}: {job_status or 'timeout'}")
                    last_result = {
                        "status": job_status or "timeout",
                        "subid": subid,
                        "job_id": job_id,
                        "cal": {},
                        "objs": []
                    }
                    continue

                # 3) Recuperar calibración y objetos
                cal = {}
                objs = []
                for _ in range(12):  # Hasta ~24s adicionales
                    cal = requests.get(
                        f"{self.api_url}/jobs/{job_id}/calibration").json()
                    objs = requests.get(
                        f"{self.api_url}/jobs/{job_id}/objects_in_field").json().get("objects_in_field", [])
                    if self._extract_calibration_value(cal, "ra") is not None:
                        break
                    time.sleep(2)

                pixscale = self._extract_calibration_value(cal, "pixscale")
                if pixscale is not None:
                    self.last_pixscale_arcsec = float(pixscale)

                return {
                    "status": "success",
                    "subid": subid,
                    "job_id": job_id,
                    "cal": cal,
                    "objs": objs
                }

            return last_result

        except Exception as e:
            print(f"\n[Astrometry] Error de conexión: {e}")
            return None

    def _extract_calibration_value(self, cal_data, key):
        """Obtiene un valor de calibración desde respuestas planas o anidadas."""
        if not isinstance(cal_data, dict):
            return None

        if key in cal_data:
            return cal_data.get(key)

        nested = cal_data.get("calibration")
        if isinstance(nested, dict):
            return nested.get(key)

        return None

    def _print_report(self, n_total, results, duration):
        print(" FASE 2: PROCESAMIENTO COMPLETADO")
        print(f"Estrellas detectadas: {n_total}")
        print(f"Selección:          <200 => todas | >=200 => 40 más brillantes")
        print(f"Tiempo total:       {duration:.1f} s")

        if results:
            status = results.get("status", "unknown")
            if status == "success":
                print(f"Estado Astrometry:  resuelto")
            else:
                print(f"Estado Astrometry:  {status}")
            ra = self._extract_calibration_value(results["cal"], "ra")
            dec = self._extract_calibration_value(results["cal"], "dec")
            pixscale = self._extract_calibration_value(
                results["cal"], "pixscale")

            if ra is not None:
                print(f"RA centro:         {ra:.4f}°")
            else:
                print("RA centro:         no disponible")

            if dec is not None:
                print(f"Dec centro:        {dec:.4f}°")
            else:
                print("Dec centro:        no disponible")

            if pixscale is not None:
                print(f"Escala:            {pixscale:.2f} arcsec/px")
            else:
                print("Escala:            no disponible")

            if results.get("objs"):
                print(f"Objetos:           {', '.join(results['objs'][:5])}")
            else:
                print("Objetos:           no disponibles")
        else:
            print(f"Estado Astrometry:  error o timeout")

if __name__ == "__main__":
    # Parametrización
    MODELO_FINAL = TRAINING_DIR / "saved_models" / "MobileUNet_2_40.pt"
    # Imagen de prueba 
    TEST_IMAGE = CURRENT_DIR / "VideosConstelaciones" / "thumbnails" / "OsaMayor1.png"
   

    pipeline = StarTrackerPipelineV2(model_path=str(MODELO_FINAL))
    pipeline.process(TEST_IMAGE)
