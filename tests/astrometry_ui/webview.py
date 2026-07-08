import webbrowser
from typing import Optional

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
except Exception:
    QWebEngineView = None


class AnnotatedWebViewDialog(QDialog):
    def __init__(self, url: str, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.url = url
        self.zoom = 1.0
        self.view: Optional[QWebEngineView] = None

        self.setWindowTitle(title)
        self.resize(1280, 900)

        root = QVBoxLayout(self)

        row_top = QHBoxLayout()
        self.url_line = QLineEdit(url)
        self.url_line.setReadOnly(True)
        open_external_btn = QPushButton("Abrir externo")
        open_external_btn.clicked.connect(self._open_external)
        reload_btn = QPushButton("Recargar")
        reload_btn.clicked.connect(self._reload)
        row_top.addWidget(self.url_line)
        row_top.addWidget(open_external_btn)
        row_top.addWidget(reload_btn)

        row_controls = QGridLayout()
        zoom_out_btn = QPushButton("-")
        zoom_out_btn.clicked.connect(lambda: self._change_zoom(-0.1))
        zoom_out_btn.setToolTip("Reducir zoom")
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.clicked.connect(lambda: self._change_zoom(0.1))
        zoom_in_btn.setToolTip("Aumentar zoom")
        zoom_reset_btn = QPushButton("100%")
        zoom_reset_btn.clicked.connect(self._reset_zoom)
        zoom_reset_btn.setToolTip("Restablecer zoom")

        pan_up_btn = QPushButton("↑")
        pan_up_btn.clicked.connect(lambda: self._pan(0, -220))
        pan_down_btn = QPushButton("↓")
        pan_down_btn.clicked.connect(lambda: self._pan(0, 220))
        pan_left_btn = QPushButton("←")
        pan_left_btn.clicked.connect(lambda: self._pan(-220, 0))
        pan_right_btn = QPushButton("→")
        pan_right_btn.clicked.connect(lambda: self._pan(220, 0))
        for btn in (pan_up_btn, pan_down_btn, pan_left_btn, pan_right_btn):
            btn.setToolTip("Mover vista")

        self.zoom_label = QLabel("Zoom: 100%")
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        row_controls.addWidget(zoom_out_btn, 0, 0)
        row_controls.addWidget(zoom_in_btn, 0, 1)
        row_controls.addWidget(zoom_reset_btn, 0, 2)
        row_controls.addWidget(self.zoom_label, 0, 3)
        row_controls.addWidget(pan_left_btn, 0, 4)
        row_controls.addWidget(pan_up_btn, 0, 5)
        row_controls.addWidget(pan_down_btn, 0, 6)
        row_controls.addWidget(pan_right_btn, 0, 7)

        root.addLayout(row_top)
        root.addLayout(row_controls)

        if QWebEngineView is None:
            fallback = QLabel("PyQt6-WebEngine no está disponible en este entorno")
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            root.addWidget(fallback)
        else:
            self.view = QWebEngineView(self)
            self.view.setUrl(QUrl(url))
            self.view.setZoomFactor(self.zoom)
            root.addWidget(self.view)

    def _change_zoom(self, delta: float) -> None:
        self.zoom = max(0.2, min(3.0, self.zoom + delta))
        self._apply_zoom()

    def _reset_zoom(self) -> None:
        self.zoom = 1.0
        self._apply_zoom()

    def _apply_zoom(self) -> None:
        self.zoom_label.setText(f"Zoom: {int(self.zoom * 100)}%")
        if self.view is not None:
            self.view.setZoomFactor(self.zoom)

    def _pan(self, dx: int, dy: int) -> None:
        if self.view is None:
            return
        self.view.page().runJavaScript(f"window.scrollBy({dx}, {dy});")

    def _reload(self) -> None:
        if self.view is not None:
            self.view.reload()

    def _open_external(self) -> None:
        try:
            webbrowser.open(self.url, new=2)
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))
