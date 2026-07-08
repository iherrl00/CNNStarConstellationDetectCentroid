import torch
import numpy as np
import cv2
from pathlib import Path
import tqdm
import math


def eval_v2_metrics(model_path, data_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cargando modelo para evaluación formal: {model_path} en {device}")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    mean = [5.9355394]
    std = [12.10830327]

    test_raw_dir = Path(data_dir) / "test_raw"
    test_centroid_dir = Path(data_dir) / "test_centroid"

    test_files = list(test_raw_dir.glob("raw_image_*.npy"))
    print(f"Evaluando {len(test_files)} imágenes de prueba...")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_squared_error = 0.0
    total_matched = 0

    match_radius = 5.0  # pixels

    for fpath in tqdm(test_files, desc="Calculando métricas"):
        stem = fpath.stem.replace("raw_image_", "")

        # Cargar imagen cruda
        img_raw = np.load(fpath)

        # Cargar centroides reales
        gt_centroid_path = test_centroid_dir / f"centroid_{stem}.npy"
        gt_centroids_data = np.load(gt_centroid_path)
        gt_centroids = []
        if gt_centroids_data.ndim == 2 and gt_centroids_data.shape[1] >= 2:
            # Volver de micras a pixeles
            pixel_size = 6/1000.0
            for i in range(gt_centroids_data.shape[0]):
                cx = (gt_centroids_data[i, 0] / pixel_size) - 0.5
                cy = (gt_centroids_data[i, 1] / pixel_size) - 0.5
                gt_centroids.append((cx, cy))

        # Inferencia
        h, w = img_raw.shape
        pad_h = (32 - (h % 32)) % 32
        pad_w = (32 - (w % 32)) % 32
        img_padded = np.pad(
            img_raw,
            ((0, pad_h), (0, pad_w)),
            mode='constant',
            constant_values=0,
        )

        img_tensor = torch.from_numpy(
            img_padded).float().unsqueeze(0).unsqueeze(0).to(device)
        img_norm = (img_tensor - mean[0]) / std[0]

        with torch.no_grad():
            prediction = model(img_norm)
            probs = torch.sigmoid(prediction[0, 0])

        pred_seg = (probs > 0.5).cpu().numpy().astype(np.uint8)
        pred_seg = pred_seg[:h, :w]

        # Sacar centroides predichos
        contours, _ = cv2.findContours(
            pred_seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pred_centroids = []
        for c in contours:
            moments = cv2.moments(c)
            if moments["m00"] > 1e-8:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]
                pred_centroids.append((cx, cy))
            else:
                (cx, cy), _ = cv2.minEnclosingCircle(c)
                pred_centroids.append((cx, cy))

            # Emparejar
        gt_matched = set()
        pred_matched = set()

        for p_idx, (px, py) in enumerate(pred_centroids):
            best_dist = float('inf')
            best_gt_idx = -1
            for g_idx, (gx, gy) in enumerate(gt_centroids):
                if g_idx in gt_matched:
                    continue
                dist = math.hypot(px - gx, py - gy)
                if dist < best_dist and dist <= match_radius:
                    best_dist = dist
                    best_gt_idx = g_idx

            if best_gt_idx != -1:
                gt_matched.add(best_gt_idx)
                pred_matched.add(p_idx)
                total_squared_error += best_dist ** 2
                total_matched += 1

        tp = len(gt_matched)
        fp = len(pred_centroids) - len(pred_matched)
        fn = len(gt_centroids) - len(gt_matched)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / \
        (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / \
        (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision +
                                     recall) if (precision + recall) > 0 else 0
    rmse = math.sqrt(total_squared_error /
                     total_matched) if total_matched > 0 else 0

    print("Resultados Fase 1 (V2)")
    print(f"Total Imágenes Test: {len(test_files)}")
    print(f"Total Estrellas Reales (GT): {total_tp + total_fn}")
    print(f"Estrellas Detectadas (IA): {total_tp + total_fp}")
    print(f"  - Verdaderos Positivos (TP): {total_tp}")
    print(f"  - Falsos Positivos (FP): {total_fp}")
    print(f"  - Falsos Negativos (FN): {total_fn}")
    print(
        f"Precision: {precision*100:.2f}% (De lo que predice, qué tanto es real)")
    print(
        f"Recall:    {recall*100:.2f}% (De lo real, qué tanto es capaz de predecir)")
    print(f"F1-Score:  {f1*100:.2f}%")
    print(f"RMSE:      {rmse:.4f} píxeles (Precisión sub-píxel posicional)")

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    m_path = SCRIPT_DIR / "saved_models" / "MobileUNet_2_40.pt"
    d_dir = SCRIPT_DIR.parent / "data_generation" / "training_data"
    eval_v2_metrics(m_path, d_dir)
