import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from astropy.io import fits

from .constants import DEFAULT_MEAN, DEFAULT_STD, TRAINING_DIR
from .io_utils import ensure_bgr_from_gray

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))


class ModelInferenceEngine:
    def __init__(self, mean: float = DEFAULT_MEAN, std: float = DEFAULT_STD) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None
        self.mean = float(mean)
        self.std = float(std) if float(std) > 1e-8 else 1.0

    def update_normalization(self, mean: float, std: float) -> None:
        self.mean = float(mean)
        self.std = float(std) if float(std) > 1e-8 else 1.0

    def load_model(self, model_path: Path) -> None:
        model = torch.load(str(model_path), map_location=self.device, weights_only=False)
        model.eval()
        self.model = model

    def infer_prob_map(self, gray_image: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("No hay modelo cargado")
        h, w = gray_image.shape
        pad_h = (32 - (h % 32)) % 32
        pad_w = (32 - (w % 32)) % 32
        img_padded = np.pad(gray_image, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
        img_tensor = torch.from_numpy(img_padded).float().unsqueeze(0).unsqueeze(0).to(self.device)
        img_norm = (img_tensor - self.mean) / self.std
        with torch.no_grad():
            output = self.model(img_norm)
            logits = self._extract_logits(output)
            prob_map = torch.sigmoid(logits).cpu().numpy()
        return np.asarray(prob_map[:h, :w], dtype=np.float32)

    def _extract_logits(self, output: Any) -> torch.Tensor:
        if isinstance(output, dict):
            for key in ("logits", "out", "pred"):
                if key in output:
                    output = output[key]
                    break
        if isinstance(output, (list, tuple)):
            output = output[0]
        if not torch.is_tensor(output):
            raise TypeError(f"Salida no soportada: {type(output)}")
        if output.ndim == 4:
            return output[0, 0]
        if output.ndim == 3:
            return output[0]
        if output.ndim == 2:
            return output
        raise ValueError(f"Dimensión de salida no soportada: {output.ndim}")

    def extract_stars(self, gray_image: np.ndarray, prob_map: np.ndarray, threshold: float) -> List[Dict[str, float]]:
        seg_map = (prob_map > threshold).astype(np.uint8)
        contours, _ = cv2.findContours(seg_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stars: List[Dict[str, float]] = []
        h, w = gray_image.shape
        for contour in contours:
            moments = cv2.moments(contour)
            if moments["m00"] <= 1e-8:
                continue
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            ix, iy = int(round(cx)), int(round(cy))
            patch = gray_image[max(0, iy - 2):min(h, iy + 3), max(0, ix - 2):min(w, ix + 3)]
            flux = float(np.sum(patch)) if patch.size else 0.0
            stars.append({"x": float(cx), "y": float(cy), "flux": flux})
        stars.sort(key=lambda item: item["flux"], reverse=True)
        return stars

    def render_clean_sky(self, stars: List[Dict[str, float]], shape: Tuple[int, int], top_k_if_dense: int) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        selected = stars
        if len(stars) >= 200 and top_k_if_dense > 0:
            selected = stars[:top_k_if_dense]
        h, w = shape
        canvas = np.zeros((h, w), dtype=np.float32)
        for star in selected:
            ix, iy = int(round(star["x"])), int(round(star["y"]))
            if 0 <= iy < h and 0 <= ix < w:
                canvas[iy, ix] = max(canvas[iy, ix], float(star["flux"]))
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
        return canvas, selected

    def to_fits_bytes(self, image_2d: np.ndarray) -> bytes:
        hdu = fits.PrimaryHDU(np.asarray(image_2d, dtype=np.float32))
        buffer = io.BytesIO()
        hdu.writeto(buffer, overwrite=True)
        return buffer.getvalue()

    def build_overlay(self, gray_image: np.ndarray, prob_map: np.ndarray, threshold: float) -> Tuple[np.ndarray, int, float]:
        base = ensure_bgr_from_gray(gray_image)
        seg_map = (prob_map > threshold).astype(np.uint8)
        contours, _ = cv2.findContours(seg_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            cv2.circle(base, (int(x), int(y)), int(radius) + 2, (0, 255, 0), 1)
        max_prob = float(np.max(prob_map)) if prob_map.size else 0.0
        label = f"Detecciones: {len(contours)}  |  Max prob: {max_prob:.4f}"
        cv2.putText(base, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 230, 255), 2, cv2.LINE_AA)
        return base, len(contours), max_prob
