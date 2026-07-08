import argparse
import os
import subprocess
import sys
from pathlib import Path


def extract_with_opencv(video_path: Path, output_path: Path) -> bool:
    try:
        import cv2
    except ImportError:
        return False

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"ERROR: No se puede abrir el video: {video_path}")
        return False

    # Intentar obtener un cuadro en el segundo 1 si es posible
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    target_frame = int(fps * 1)
    capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    ret, frame = capture.read()
    capture.release()
    if not ret or frame is None:
        print(f"ERROR: No se pudo leer un cuadro de {video_path}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), frame)
    if not success:
        print(f"ERROR: No se pudo guardar la imagen: {output_path}")
        return False

    return True


def extract_with_ffmpeg(video_path: Path, output_path: Path) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ss",
        "00:00:01",
        "-vframes",
        "1",
        str(output_path),
    ]
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if completed.returncode != 0:
            print(f"ERROR: ffmpeg falló para {video_path}\n{completed.stderr}")
            return False
        return True
    except FileNotFoundError:
        return False


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def find_mp4_files(source_dir: Path):
    return sorted(source_dir.glob("*.mp4"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extrae una imagen PNG de alta resolución de cada video MP4 en una carpeta de salida."
    )
    parser.add_argument(
        "source_dir",
        nargs="?",
        default=".",
        help="Carpeta donde están los archivos MP4 (por defecto: carpeta actual)",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="thumbnails",
        help="Carpeta de salida donde se guardan los PNG (por defecto: thumbnails)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not source_dir.exists() or not source_dir.is_dir():
        print(f"ERROR: La carpeta de origen no existe: {source_dir}")
        sys.exit(1)

    mp4_files = find_mp4_files(source_dir)
    if not mp4_files:
        print(f"No se encontraron archivos MP4 en {source_dir}")
        sys.exit(0)

    ensure_output_dir(output_dir)
    use_opencv = False
    try:
        import cv2  # type: ignore
        use_opencv = True
    except ImportError:
        use_opencv = False

    if not use_opencv:
        print("INFO: OpenCV no está instalado, intentando ffmpeg...")
        if not shutil.which("ffmpeg"):
            print("ERROR: ni OpenCV ni ffmpeg están disponibles.")
            sys.exit(1)

    success_count = 0
    for video_path in mp4_files:
        output_path = output_dir / f"{video_path.stem}.png"
        print(f"Procesando {video_path.name} -> {output_path.name}")

        if use_opencv:
            success = extract_with_opencv(video_path, output_path)
            if not success:
                success = extract_with_ffmpeg(video_path, output_path)
        else:
            success = extract_with_ffmpeg(video_path, output_path)

        if success:
            success_count += 1

    print(f"\nImágenes generadas: {success_count}/{len(mp4_files)}")

    if success_count != len(mp4_files):
        sys.exit(1)


if __name__ == "__main__":
    import shutil

    main()
