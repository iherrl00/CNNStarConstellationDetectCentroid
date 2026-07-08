from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
TESTS_DIR = PACKAGE_DIR.parent
ISABELLA_DEVELOPMENTS_DIR = TESTS_DIR.parent
WORKSPACE_ROOT = ISABELLA_DEVELOPMENTS_DIR.parent
TRAINING_DIR = ISABELLA_DEVELOPMENTS_DIR / "src" / "training"

DEFAULT_MODEL = TRAINING_DIR / "saved_models" / "MobileUNet_2_40.pt"
DEFAULT_API_URL = "http://nova.astrometry.net/api"
DEFAULT_API_KEY = "wnmzxqdgrsmercxg"
DEFAULT_MEAN = 5.9355394
DEFAULT_STD = 12.10830327

MODEL_FILTER = "Modelos (*.pt *.pth);;Todos (*)"
IMAGE_FILTER = "Imagen/FITS/NPY (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.fits *.fit *.fts *.npy);;Todos (*)"
VIDEO_FILTER = "Videos (*.mp4 *.avi *.mov *.mkv *.wmv);;Todos (*)"
