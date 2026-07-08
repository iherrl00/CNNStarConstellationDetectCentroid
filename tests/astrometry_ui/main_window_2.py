import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .astrometry_service import AstrometryConfig, AstrometryWorker
from .constants import (
    DEFAULT_API_KEY,
    DEFAULT_API_URL,
    DEFAULT_MEAN,
    DEFAULT_MODEL,
    DEFAULT_STD,
    IMAGE_FILTER,
    MODEL_FILTER,
    TRAINING_DIR,
    VIDEO_FILTER,
    WORKSPACE_ROOT,
)
from .inference_engine import ModelInferenceEngine
from .io_utils import ensure_bgr_from_gray, load_env_file, ndarray_to_pixmap, read_image_to_gray
from .result_parser import parse_astrometry_result
from .results_widget_2 import AstrometryResultsPanel
from .style import APP_STYLE
from .webview import AnnotatedWebViewDialog


class InferenceAstrometryWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        env_values = load_env_file(WORKSPACE_ROOT / ".env")

        self.engine = ModelInferenceEngine()
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.video_interval_ms = 33
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self._process_video_tick)
        self.astrometry_worker: Optional[AstrometryWorker] = None

        self.last_image_gray: Optional[np.ndarray] = None
        self.last_image_prob: Optional[np.ndarray] = None
        self.last_frame_gray: Optional[np.ndarray] = None
        self.last_frame_prob: Optional[np.ndarray] = None

        self.source_pixmap: Optional[QPixmap] = None
        self.pred_pixmap: Optional[QPixmap] = None
        self.webviews: List[AnnotatedWebViewDialog] = []
        self.latest_raw_result: Optional[Dict[str, Any]] = None

        self.setWindowTitle("Inference + Astrometry")
        self.resize(1540, 1000)
        self._build_ui(env_values)
        self.setStyleSheet(APP_STYLE)

        if DEFAULT_MODEL.exists():
            self.model_path_input.setText(str(DEFAULT_MODEL))

    def _build_ui(self, env_values: Dict[str, str]) -> None:
        root_scroll = QScrollArea()
        root_scroll.setWidgetResizable(True)

        container = QWidget()
        root_layout = QVBoxLayout(container)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        model_box = QGroupBox("Modelo")
        model_layout = QGridLayout(model_box)

        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("Ruta del .pt")
        model_browse_btn = QPushButton("Examinar")
        model_browse_btn.clicked.connect(self._browse_model)
        self.load_model_btn = QPushButton("Cargar")
        self.load_model_btn.clicked.connect(self._load_model)

        self.mean_input = QDoubleSpinBox()
        self.mean_input.setDecimals(7)
        self.mean_input.setRange(-1e6, 1e6)
        self.mean_input.setValue(DEFAULT_MEAN)
        self.mean_input.setToolTip("Media usada para normalización de entrada")

        self.std_input = QDoubleSpinBox()
        self.std_input.setDecimals(7)
        self.std_input.setRange(1e-7, 1e6)
        self.std_input.setValue(DEFAULT_STD)
        self.std_input.setToolTip("Desviación estándar para normalización")

        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setDecimals(3)
        self.threshold_input.setRange(0.01, 0.99)
        self.threshold_input.setSingleStep(0.01)
        self.threshold_input.setValue(0.5)
        self.threshold_input.setToolTip("Umbral binario sobre mapa de probabilidad")

        self.topk_input = QSpinBox()
        self.topk_input.setRange(1, 1000)
        self.topk_input.setValue(40)
        self.topk_input.setToolTip("Si hay 200 o más detecciones, limita las enviadas a Astrometry")

        self.device_label = QLabel(f"Dispositivo: {self.engine.device}")

        model_layout.addWidget(QLabel("Ruta"), 0, 0)
        model_layout.addWidget(self.model_path_input, 0, 1, 1, 4)
        model_layout.addWidget(model_browse_btn, 0, 5)
        model_layout.addWidget(self.load_model_btn, 0, 6)

        model_layout.addWidget(QLabel("Mean"), 1, 0)
        model_layout.addWidget(self.mean_input, 1, 1)
        model_layout.addWidget(QLabel("Std"), 1, 2)
        model_layout.addWidget(self.std_input, 1, 3)
        model_layout.addWidget(QLabel("Threshold"), 1, 4)
        model_layout.addWidget(self.threshold_input, 1, 5)
        model_layout.addWidget(QLabel("Top K"), 1, 6)
        model_layout.addWidget(self.topk_input, 1, 7)
        model_layout.addWidget(self.device_label, 2, 0, 1, 8)

        io_box = QGroupBox("Imagen y video")
        io_layout = QGridLayout(io_box)

        self.image_path_input = QLineEdit()
        self.image_path_input.setPlaceholderText("Ruta de imagen")
        image_browse_btn = QPushButton("Examinar")
        image_browse_btn.clicked.connect(self._browse_image)
        self.infer_image_btn = QPushButton("Inferir imagen")
        self.infer_image_btn.clicked.connect(self._infer_image)
        self.solve_image_btn = QPushButton("Astrometry imagen")
        self.solve_image_btn.clicked.connect(self._solve_from_image)

        self.video_path_input = QLineEdit()
        self.video_path_input.setPlaceholderText("Ruta de video")
        video_browse_btn = QPushButton("Examinar")
        video_browse_btn.clicked.connect(self._browse_video)
        self.start_video_btn = QPushButton("Iniciar")
        self.start_video_btn.clicked.connect(self._start_video)
        self.pause_video_btn = QPushButton("Pausar")
        self.pause_video_btn.clicked.connect(self._pause_video)
        self.stop_video_btn = QPushButton("Detener")
        self.stop_video_btn.clicked.connect(self._stop_video)
        self.solve_frame_btn = QPushButton("Astrometry frame")
        self.solve_frame_btn.clicked.connect(self._solve_from_current_frame)

        self.video_stride_input = QSpinBox()
        self.video_stride_input.setRange(1, 30)
        self.video_stride_input.setValue(1)
        self.video_stride_input.setToolTip("Procesa un frame cada N frames para acelerar inferencia")

        io_layout.addWidget(QLabel("Imagen"), 0, 0)
        io_layout.addWidget(self.image_path_input, 0, 1, 1, 5)
        io_layout.addWidget(image_browse_btn, 0, 6)
        io_layout.addWidget(self.infer_image_btn, 0, 7)

        io_layout.addWidget(QLabel("Video"), 1, 0)
        io_layout.addWidget(self.video_path_input, 1, 1, 1, 5)
        io_layout.addWidget(video_browse_btn, 1, 6)
        io_layout.addWidget(self.start_video_btn, 1, 7)
        io_layout.addWidget(self.pause_video_btn, 1, 8)
        io_layout.addWidget(self.stop_video_btn, 1, 9)

        io_layout.addWidget(QLabel("Stride"), 2, 0)
        io_layout.addWidget(self.video_stride_input, 2, 1)

        preview_box = QGroupBox("Vista")
        preview_layout = QHBoxLayout(preview_box)

        self.source_label = QLabel("Sin entrada")
        self.source_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.source_label.setMinimumSize(680, 380)

        self.pred_label = QLabel("Sin predicción")
        self.pred_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pred_label.setMinimumSize(680, 380)

        preview_layout.addWidget(self.source_label)
        preview_layout.addWidget(self.pred_label)

        astrometry_box = QGroupBox("Astrometry")
        astrometry_layout = QGridLayout(astrometry_box)

        self.api_url_input = QLineEdit(env_values.get("ASTROMETRY_API_URL", DEFAULT_API_URL))
        api_key_default = env_values.get("ASTROMETRY_API_KEY") or DEFAULT_API_KEY
        self.api_key_input = QLineEdit(api_key_default)
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        show_key_check = QCheckBox("Mostrar key")
        show_key_check.toggled.connect(self._toggle_api_key_visibility)

        self.public_visibility_combo = QComboBox()
        self.public_visibility_combo.addItems(["n", "y"])
        self.public_visibility_combo.setCurrentText("n")
        self.public_visibility_combo.setToolTip("Controla visibilidad pública del envío")

        self.commercial_combo = QComboBox()
        self.commercial_combo.addItems(["d", "y", "n"])
        self.commercial_combo.setCurrentText("d")
        self.commercial_combo.setToolTip("Política de uso comercial en Astrometry")

        self.modifications_combo = QComboBox()
        self.modifications_combo.addItems(["d", "y", "n", "sa"])
        self.modifications_combo.setCurrentText("d")
        self.modifications_combo.setToolTip("Política de modificaciones para el resultado")

        self.sub_timeout_input = QSpinBox()
        self.sub_timeout_input.setRange(30, 1800)
        self.sub_timeout_input.setValue(300)
        self.sub_timeout_input.setToolTip("Tiempo máximo esperando asignación de job")

        self.job_timeout_input = QSpinBox()
        self.job_timeout_input.setRange(30, 3600)
        self.job_timeout_input.setValue(600)
        self.job_timeout_input.setToolTip("Tiempo máximo esperando resolución final del job")

        astrometry_layout.addWidget(QLabel("API URL"), 0, 0)
        astrometry_layout.addWidget(self.api_url_input, 0, 1, 1, 5)
        astrometry_layout.addWidget(QLabel("API Key"), 1, 0)
        astrometry_layout.addWidget(self.api_key_input, 1, 1, 1, 4)
        astrometry_layout.addWidget(show_key_check, 1, 5)

        astrometry_layout.addWidget(QLabel("Visible"), 2, 0)
        astrometry_layout.addWidget(self.public_visibility_combo, 2, 1)
        astrometry_layout.addWidget(QLabel("Comercial"), 2, 2)
        astrometry_layout.addWidget(self.commercial_combo, 2, 3)
        astrometry_layout.addWidget(QLabel("Modificaciones"), 2, 4)
        astrometry_layout.addWidget(self.modifications_combo, 2, 5)

        astrometry_layout.addWidget(QLabel("Timeout sub"), 3, 0)
        astrometry_layout.addWidget(self.sub_timeout_input, 3, 1)
        astrometry_layout.addWidget(QLabel("Timeout job"), 3, 2)
        astrometry_layout.addWidget(self.job_timeout_input, 3, 3)
        astrometry_layout.addWidget(self.solve_image_btn, 3, 4)
        astrometry_layout.addWidget(self.solve_frame_btn, 3, 5)

        self.astrometry_progress = QProgressBar()
        self.astrometry_progress.setVisible(False)
        astrometry_layout.addWidget(self.astrometry_progress, 4, 0, 1, 6)

        results_box = QGroupBox("Resultados Astrometry")
        results_layout = QVBoxLayout(results_box)
        self.results_panel = AstrometryResultsPanel()
        self.results_panel.open_webview_requested.connect(self._open_webview)
        results_layout.addWidget(self.results_panel)

        logs_box = QGroupBox("Registro")
        logs_layout = QVBoxLayout(logs_box)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(170)
        logs_layout.addWidget(self.log_output)

        root_layout.addWidget(model_box)
        root_layout.addWidget(io_box)
        root_layout.addWidget(preview_box)
        root_layout.addWidget(astrometry_box)
        root_layout.addWidget(results_box)
        root_layout.addWidget(logs_box)

        root_scroll.setWidget(container)
        self.setCentralWidget(root_scroll)

    def _toggle_api_key_visibility(self, checked: bool) -> None:
        mode = QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
        self.api_key_input.setEchoMode(mode)

    def _browse_model(self) -> None:
        model_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar modelo", str(TRAINING_DIR), MODEL_FILTER)
        if model_path:
            self.model_path_input.setText(model_path)

    def _browse_image(self) -> None:
        image_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar imagen", str(WORKSPACE_ROOT), IMAGE_FILTER)
        if image_path:
            self.image_path_input.setText(image_path)

    def _browse_video(self) -> None:
        video_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar video", str(WORKSPACE_ROOT), VIDEO_FILTER)
        if video_path:
            self.video_path_input.setText(video_path)

    def _log(self, text: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_output.appendPlainText(f"[{timestamp}] {text}")

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)
        self._log(f"ERROR: {message}")

    def _load_model(self) -> None:
        model_path = Path(self.model_path_input.text().strip())
        if not model_path.exists():
            self._show_error("La ruta del modelo no existe")
            return
        try:
            self.engine.update_normalization(self.mean_input.value(), self.std_input.value())
            self.engine.load_model(model_path)
            self._log(f"Modelo cargado: {model_path.name}")
        except Exception as exc:
            self._show_error(str(exc))

    def _infer_image(self) -> None:
        if self.engine.model is None:
            self._show_error("Primero carga un modelo")
            return
        image_path = Path(self.image_path_input.text().strip())
        if not image_path.exists():
            self._show_error("La imagen no existe")
            return
        try:
            gray = read_image_to_gray(image_path)
            prob_map = self.engine.infer_prob_map(gray)
            overlay, n_stars, max_prob = self.engine.build_overlay(gray, prob_map, self.threshold_input.value())
            self.last_image_gray = gray
            self.last_image_prob = prob_map
            source_bgr = ensure_bgr_from_gray(gray)
            self._set_source_image(source_bgr)
            self._set_pred_image(overlay)
            self._log(f"Inferencia imagen lista | detecciones={n_stars} | max_prob={max_prob:.4f}")
        except Exception as exc:
            self._show_error(str(exc))

    def _start_video(self) -> None:
        if self.engine.model is None:
            self._show_error("Primero carga un modelo")
            return
        video_path = Path(self.video_path_input.text().strip())
        if not video_path.exists():
            self._show_error("El video no existe")
            return
        self._stop_video()
        self.video_capture = cv2.VideoCapture(str(video_path))
        if not self.video_capture.isOpened():
            self.video_capture = None
            self._show_error("No se pudo abrir el video")
            return
        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25
        self.video_interval_ms = int(max(15, 1000.0 / fps))
        self.video_timer.start(self.video_interval_ms)
        self._log(f"Video en tiempo real iniciado | FPS fuente={fps:.2f}")

    def _pause_video(self) -> None:
        if self.video_timer.isActive():
            self.video_timer.stop()
            self._log("Video pausado")
        elif self.video_capture is not None:
            self.video_timer.start(self.video_interval_ms)
            self._log("Video reanudado")

    def _stop_video(self) -> None:
        self.video_timer.stop()
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

    def _process_video_tick(self) -> None:
        if self.video_capture is None:
            return
        stride = max(1, self.video_stride_input.value())
        for _ in range(stride - 1):
            if not self.video_capture.grab():
                self._stop_video()
                self._log("Video finalizado")
                return
        ok, frame_bgr = self.video_capture.read()
        if not ok or frame_bgr is None:
            self._stop_video()
            self._log("Video finalizado")
            return
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        try:
            prob_map = self.engine.infer_prob_map(gray)
            overlay, n_stars, max_prob = self.engine.build_overlay(gray, prob_map, self.threshold_input.value())
        except Exception as exc:
            self._stop_video()
            self._show_error(str(exc))
            return
        self.last_frame_gray = gray
        self.last_frame_prob = prob_map
        source_bgr = ensure_bgr_from_gray(gray)
        cv2.putText(source_bgr, "Video", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"Detecciones: {n_stars} | Max: {max_prob:.4f}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 230, 120), 2, cv2.LINE_AA)
        self._set_source_image(source_bgr)
        self._set_pred_image(overlay)

    def _set_source_image(self, bgr: np.ndarray) -> None:
        self.source_pixmap = ndarray_to_pixmap(bgr)
        self._refresh_label_pixmap(self.source_label, self.source_pixmap)

    def _set_pred_image(self, bgr: np.ndarray) -> None:
        self.pred_pixmap = ndarray_to_pixmap(bgr)
        self._refresh_label_pixmap(self.pred_label, self.pred_pixmap)

    def _refresh_label_pixmap(self, label: QLabel, pixmap: Optional[QPixmap]) -> None:
        if pixmap is None:
            return
        target = pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        label.setPixmap(target)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_label_pixmap(self.source_label, self.source_pixmap)
        self._refresh_label_pixmap(self.pred_label, self.pred_pixmap)

    def _build_astrometry_config(self) -> Optional[AstrometryConfig]:
        api_url = self.api_url_input.text().strip()
        api_key = self.api_key_input.text().strip()
        if not api_url:
            self._show_error("Define la API URL de Astrometry")
            return None
        if not api_key:
            self._show_error("Define la API key de Astrometry")
            return None
        return AstrometryConfig(
            api_url=api_url,
            api_key=api_key,
            publicly_visible=self.public_visibility_combo.currentText(),
            allow_commercial_use=self.commercial_combo.currentText(),
            allow_modifications=self.modifications_combo.currentText(),
            submission_timeout_sec=int(self.sub_timeout_input.value()),
            job_timeout_sec=int(self.job_timeout_input.value()),
        )

    def _solve_from_image(self) -> None:
        if self.engine.model is None:
            self._show_error("Primero carga un modelo")
            return
        if self.last_image_gray is None or self.last_image_prob is None:
            self._infer_image()
            if self.last_image_gray is None or self.last_image_prob is None:
                return
        self._solve_with_astrometry(self.last_image_gray, self.last_image_prob, source_label="imagen")

    def _solve_from_current_frame(self) -> None:
        if self.engine.model is None:
            self._show_error("Primero carga un modelo")
            return
        if self.last_frame_gray is None or self.last_frame_prob is None:
            self._show_error("No hay frame disponible")
            return
        self._solve_with_astrometry(self.last_frame_gray, self.last_frame_prob, source_label="video_frame")

    def _solve_with_astrometry(self, gray: np.ndarray, prob: np.ndarray, source_label: str) -> None:
        if self.astrometry_worker is not None and self.astrometry_worker.isRunning():
            self._show_error("Ya existe una solicitud Astrometry en curso")
            return

        config = self._build_astrometry_config()
        if config is None:
            return

        try:
            threshold = float(self.threshold_input.value())
            stars = self.engine.extract_stars(gray, prob, threshold)
            clean_sky, selected = self.engine.render_clean_sky(stars, gray.shape, self.topk_input.value())
            fits_bytes = self.engine.to_fits_bytes(clean_sky)
        except Exception as exc:
            self._show_error(str(exc))
            return

        self._log(f"Astrometry {source_label}: detecciones={len(stars)} | enviadas={len(selected)}")

        self.astrometry_worker = AstrometryWorker(config=config, fits_bytes=fits_bytes, filename=f"{source_label}_clean_sky.fits")
        self.astrometry_worker.log_message.connect(self._log)
        self.astrometry_worker.completed.connect(self._on_astrometry_completed)
        self.astrometry_worker.failed.connect(self._on_astrometry_failed)

        self.astrometry_progress.setRange(0, 0)
        self.astrometry_progress.setVisible(True)
        self._set_astrometry_buttons_enabled(False)
        self.astrometry_worker.start()

    def _set_astrometry_buttons_enabled(self, enabled: bool) -> None:
        self.solve_image_btn.setEnabled(enabled)
        self.solve_frame_btn.setEnabled(enabled)

    def _on_astrometry_completed(self, result: Dict[str, Any]) -> None:
        self.latest_raw_result = result
        self.astrometry_progress.setVisible(False)
        self._set_astrometry_buttons_enabled(True)
        view_data = parse_astrometry_result(result)
        # Cambio: Pasar tamaño de imagen al panel para conversión WCS
        gray = self.last_image_gray if self.last_image_gray is not None else self.last_frame_gray
        if gray is not None:
            self.results_panel.set_image_size(float(gray.shape[1]), float(gray.shape[0]))
        self.results_panel.set_result(view_data)
        self._log(f"Astrometry completado | status={view_data.status} | subid={view_data.subid} | job={view_data.job_id}")

    def _on_astrometry_failed(self, error_message: str) -> None:
        self.astrometry_progress.setVisible(False)
        self._set_astrometry_buttons_enabled(True)
        self._show_error(error_message)

    def _open_webview(self, title: str, url: str) -> None:
        dialog = AnnotatedWebViewDialog(url=url, title=title, parent=self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.webviews.append(dialog)
        dialog.finished.connect(lambda _result, dlg=dialog: self._remove_webview(dlg))
        dialog.show()

    def _remove_webview(self, dialog: AnnotatedWebViewDialog) -> None:
        self.webviews = [dlg for dlg in self.webviews if dlg is not dialog]

    def closeEvent(self, event) -> None:
        self._stop_video()
        super().closeEvent(event)


def run() -> None:
    app = QApplication(sys.argv)
    window = InferenceAstrometryWindow()
    window.show()
    sys.exit(app.exec())