import argparse
from pathlib import Path
import random

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def discover_images(input_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return [p for p in input_dir.rglob("*") if p.suffix.lower() in exts]


def preprocess_and_detect(gray_f32: np.ndarray, pixel_area_min: int = 3, kernel_size: int = 11):
    img_u8 = cv2.normalize(gray_f32, None, 0, 255,
                           cv2.NORM_MINMAX).astype(np.uint8)

    # Aísla picos de luz pequeños.
    kernel_tophat = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    top_hat = cv2.morphologyEx(img_u8, cv2.MORPH_TOPHAT, kernel_tophat)

    # Suaviza un poco para unir la estrella.
    blur = cv2.GaussianBlur(top_hat, (3, 3), 0)

    # Umbral simple.
    _, binary = cv2.threshold(blur, 8, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray_f32.shape
    seg_map = np.zeros((h, w), dtype=np.float32)
    centers_px = []

    for cnt in contours:
        # Ignora ruido muy pequeño.
        area = cv2.contourArea(cnt)
        if area < float(pixel_area_min):
            continue

        moments = cv2.moments(cnt)
        if moments["m00"] <= 1e-8:
            continue

        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        centers_px.append((cx, cy))

        # Marca la zona detectada.
        cv2.drawContours(seg_map, [cnt], contourIdx=-
                         1, color=1.0, thickness=-1)

    return seg_map, centers_px


def build_dist_map(shape, centers_px, pixel_size):
    h, w = shape
    if len(centers_px) == 0:
        return np.zeros((h, w), dtype=np.float32)

    seed = np.ones((h, w), dtype=np.uint8)
    for cx, cy in centers_px:
        x = int(np.clip(np.round(cx), 0, w - 1))
        y = int(np.clip(np.round(cy), 0, h - 1))
        seed[y, x] = 0

    dist_map = distance_transform_edt(seed).astype(np.float32)
    dist_map = dist_map / pixel_size
    return dist_map


def build_centroid_tensor(centers_px, pixel_size):
    if len(centers_px) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    centroid = np.zeros((len(centers_px), 4), dtype=np.float32)
    for i, (cx, cy) in enumerate(centers_px):
        u = (cx + 0.5) * pixel_size
        v = (cy + 0.5) * pixel_size
        centroid[i] = np.array([u, v, 0.0, 0.0], dtype=np.float32)
    return centroid


def make_split_indices(total, train_ratio, val_ratio, seed):
    indices = list(range(total))
    random.Random(seed).shuffle(indices)

    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_train = min(n_train, total)
    n_val = min(n_val, total - n_train)

    train_idx = set(indices[:n_train])
    val_idx = set(indices[n_train:n_train + n_val])
    test_idx = set(indices[n_train + n_val:])
    return train_idx, val_idx, test_idx


def ensure_output_tree(output_dir: Path):
    for split in ["train", "val", "test"]:
        (output_dir / f"{split}_raw").mkdir(parents=True, exist_ok=True)
        (output_dir / f"{split}_seg_map").mkdir(parents=True, exist_ok=True)
        (output_dir / f"{split}_dist_map").mkdir(parents=True, exist_ok=True)
        (output_dir / f"{split}_centroid").mkdir(parents=True, exist_ok=True)


def resolve_split(i, train_idx, val_idx):
    if i in train_idx:
        return "train"
    if i in val_idx:
        return "val"
    return "test"


def main():
    default_input = Path(__file__).resolve().parents[2] / "dataset6k"
    default_output = Path(__file__).resolve().parent / "training_data"

    parser = argparse.ArgumentParser(
        description="Genera pseudo ground truth desde imágenes Stellarium."
    )
    parser.add_argument("--input_dir", type=str, default=str(default_input))
    parser.add_argument("--output_dir", type=str, default=str(default_output))
    parser.add_argument("--pixel_size", type=float, default=6 / 1000.0)
    parser.add_argument("--pixel_area_min", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=15)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"No existe input_dir: {input_dir}")

    images = discover_images(input_dir)
    if len(images) == 0:
        raise RuntimeError(f"No se encontraron imágenes en: {input_dir}")

    ensure_output_tree(output_dir)

    train_idx, val_idx, test_idx = make_split_indices(
        total=len(images),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    split_counters = {"train": 0, "val": 0, "test": 0}

    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] No se pudo leer: {img_path}")
            continue

        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        gray_f32 = gray.astype(np.float32)

        seg_map, centers_px = preprocess_and_detect(
            gray_f32,
            pixel_area_min=args.pixel_area_min,
            kernel_size=args.kernel_size,
        )
        dist_map = build_dist_map(gray_f32.shape, centers_px, args.pixel_size)
        centroid = build_centroid_tensor(centers_px, args.pixel_size)

        split = resolve_split(i, train_idx, val_idx)
        idx_split = split_counters[split]
        split_counters[split] += 1

        stem = f"{idx_split:06d}"
        np.save(output_dir / f"{split}_raw" /
                f"raw_image_{stem}.npy", gray_f32)
        np.save(output_dir / f"{split}_seg_map" /
                f"seg_map_{stem}.npy", seg_map)
        np.save(output_dir / f"{split}_dist_map" /
                f"dist_map_{stem}.npy", dist_map)
        np.save(output_dir / f"{split}_centroid" /
                f"centroid_{stem}.npy", centroid)

        if (i + 1) % 200 == 0:
            print(f"Procesadas {i + 1}/{len(images)} imágenes")

    print("Generación completada.")
    print(
        f"Train: {split_counters['train']} | Val: {split_counters['val']} | Test: {split_counters['test']}")
    print(f"Salida: {output_dir}")


if __name__ == "__main__":
    main()
