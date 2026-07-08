from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from astropy.io import fits
from PyQt6.QtGui import QImage, QPixmap


def load_env_file(env_path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not env_path.exists():
        return env
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            env[key] = value
    return env


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    if img.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def read_image_to_gray(image_path: Path) -> np.ndarray:
    suffix = image_path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(str(image_path))
        if arr.ndim == 3:
            arr = arr[..., 0]
        return np.asarray(arr, dtype=np.float32)
    if suffix in {".fits", ".fit", ".fts"}:
        data = fits.getdata(str(image_path))
        if data is None:
            raise ValueError(f"FITS sin datos: {image_path}")
        if data.ndim > 2:
            data = data[0]
        return np.asarray(data, dtype=np.float32)
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return np.asarray(gray, dtype=np.float32)


def ensure_bgr_from_gray(gray_image: np.ndarray) -> np.ndarray:
    gray_u8 = normalize_to_uint8(gray_image)
    return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)


def ndarray_to_pixmap(bgr_image: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(q_img.copy())
