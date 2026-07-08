import webbrowser
from typing import Dict, Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .result_parser import AstrometryViewData


class AstrometryResultsPanel(QWidget):
    open_webview_requested = pyqtSignal(str, str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.links: Dict[str, str] = {}

        root = QVBoxLayout(self)

        self.tabs = QTabWidget()

        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)

        summary_group = QGroupBox("Resumen")
        summary_form = QFormLayout(summary_group)
        self.status_value = self._value_label()
        self.subid_value = self._value_label()
        self.job_id_value = self._value_label()
        self.session_value = self._value_label()
        summary_form.addRow("Estado", self.status_value)
        summary_form.addRow("Submission ID", self.subid_value)
        summary_form.addRow("Job ID", self.job_id_value)
        summary_form.addRow("Session", self.session_value)

        calibration_group = QGroupBox("Calibración")
        calibration_form = QFormLayout(calibration_group)
        self.ra_value = self._value_label()
        self.dec_value = self._value_label()
        self.pixscale_value = self._value_label()
        self.orientation_value = self._value_label()
        self.parity_value = self._value_label()
        self.radius_value = self._value_label()
        self.width_value = self._value_label()
        self.height_value = self._value_label()
        self.orientation_value.setToolTip("Ángulo de orientación del campo resuelto")
        self.parity_value.setToolTip("Paridad del WCS devuelta por Astrometry")
        calibration_form.addRow("RA", self.ra_value)
        calibration_form.addRow("Dec", self.dec_value)
        calibration_form.addRow("Escala", self.pixscale_value)
        calibration_form.addRow("Orientación", self.orientation_value)
        calibration_form.addRow("Paridad", self.parity_value)
        calibration_form.addRow("Radio", self.radius_value)
        calibration_form.addRow("Ancho", self.width_value)
        calibration_form.addRow("Alto", self.height_value)

        counts_group = QGroupBox("Conteos")
        counts_form = QFormLayout(counts_group)
        self.const_count_value = self._value_label()
        self.objects_count_value = self._value_label()
        self.tags_count_value = self._value_label()
        self.stars_count_value = self._value_label()
        self.annotations_count_value = self._value_label()
        counts_form.addRow("Constelaciones", self.const_count_value)
        counts_form.addRow("Objetos", self.objects_count_value)
        counts_form.addRow("Tags", self.tags_count_value)
        counts_form.addRow("Estrellas", self.stars_count_value)
        counts_form.addRow("Anotaciones", self.annotations_count_value)

        links_group = QGroupBox("Visualización")
        links_layout = QGridLayout(links_group)
        self.annotated_btn = QPushButton("Annotated")
        self.extraction_btn = QPushButton("Extraction")
        self.wcs_btn = QPushButton("WCS")
        self.new_fits_btn = QPushButton("New FITS")
        self.annotated_btn.clicked.connect(lambda: self._open_in_webview("annotated_display", "Annotated display"))
        self.extraction_btn.clicked.connect(lambda: self._open_in_webview("extraction_image_display", "Extraction display"))
        self.wcs_btn.clicked.connect(lambda: self._open_external("wcs_file"))
        self.new_fits_btn.clicked.connect(lambda: self._open_external("new_fits_file"))
        self.annotated_btn.setToolTip("Abre la vista anotada dentro de la app")
        self.extraction_btn.setToolTip("Abre la extracción dentro de la app")
        links_layout.addWidget(self.annotated_btn, 0, 0)
        links_layout.addWidget(self.extraction_btn, 0, 1)
        links_layout.addWidget(self.wcs_btn, 0, 2)
        links_layout.addWidget(self.new_fits_btn, 0, 3)

        overview_layout.addWidget(summary_group)
        overview_layout.addWidget(calibration_group)
        overview_layout.addWidget(counts_group)
        overview_layout.addWidget(links_group)

        objects_tab = QWidget()
        objects_layout = QVBoxLayout(objects_tab)
        splitter = QSplitter()

        const_group = QGroupBox("Constelaciones")
        const_layout = QVBoxLayout(const_group)
        self.constellations_list = QListWidget()
        const_layout.addWidget(self.constellations_list)

        obj_group = QGroupBox("Objetos en campo")
        obj_layout = QVBoxLayout(obj_group)
        self.objects_list = QListWidget()
        obj_layout.addWidget(self.objects_list)

        stars_group = QGroupBox("Estrellas relevantes")
        stars_layout = QVBoxLayout(stars_group)
        self.stars_list = QListWidget()
        stars_layout.addWidget(self.stars_list)

        splitter.addWidget(const_group)
        splitter.addWidget(obj_group)
        splitter.addWidget(stars_group)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)
        objects_layout.addWidget(splitter)

        annotations_tab = QWidget()
        annotations_layout = QVBoxLayout(annotations_tab)
        self.annotations_table = QTableWidget(0, 5)
        self.annotations_table.setHorizontalHeaderLabels(["Nombre", "Vmag", "X", "Y", "Tipo"])
        self.annotations_table.horizontalHeader().setStretchLastSection(True)
        self.annotations_table.setSortingEnabled(True)
        annotations_layout.addWidget(self.annotations_table)

        self.tabs.addTab(overview_tab, "Resumen")
        self.tabs.addTab(objects_tab, "Objetos")
        self.tabs.addTab(annotations_tab, "Anotaciones")

        root.addWidget(self.tabs)

    def set_result(self, data: AstrometryViewData) -> None:
        self.links = data.links

        self.status_value.setText(data.status)
        self.subid_value.setText(self._fmt_int(data.subid))
        self.job_id_value.setText(self._fmt_int(data.job_id))
        self.session_value.setText(data.session)

        self.ra_value.setText(self._fmt_float(data.ra, "°", 6))
        self.dec_value.setText(self._fmt_float(data.dec, "°", 6))
        self.pixscale_value.setText(self._fmt_float(data.pixscale, " arcsec/px", 4))
        self.orientation_value.setText(self._fmt_float(data.orientation, "°", 4))
        self.parity_value.setText(self._fmt_float(data.parity, "", 1))
        self.radius_value.setText(self._fmt_float(data.radius, "°", 4))
        self.width_value.setText(self._fmt_float(data.width_arcsec, " arcsec", 3))
        self.height_value.setText(self._fmt_float(data.height_arcsec, " arcsec", 3))

        self.const_count_value.setText(str(len(data.constellations)))
        self.objects_count_value.setText(str(len(data.objects)))
        self.tags_count_value.setText(str(len(data.tags)))
        self.stars_count_value.setText(str(len(data.stars)))
        self.annotations_count_value.setText(str(len(data.bright_annotations)))

        self._fill_list(self.constellations_list, data.constellations)
        self._fill_list(self.objects_list, data.objects)
        self._fill_list(self.stars_list, data.stars)

        self.annotations_table.setSortingEnabled(False)
        self.annotations_table.setRowCount(len(data.bright_annotations))
        for row, item in enumerate(data.bright_annotations):
            self.annotations_table.setItem(row, 0, QTableWidgetItem(item.name))
            self.annotations_table.setItem(row, 1, QTableWidgetItem(self._fmt_float(item.vmag, "", 3)))
            self.annotations_table.setItem(row, 2, QTableWidgetItem(self._fmt_float(item.x, "", 2)))
            self.annotations_table.setItem(row, 3, QTableWidgetItem(self._fmt_float(item.y, "", 2)))
            self.annotations_table.setItem(row, 4, QTableWidgetItem(item.kind))
        self.annotations_table.resizeColumnsToContents()
        self.annotations_table.setSortingEnabled(True)

        self.annotated_btn.setEnabled("annotated_display" in self.links)
        self.extraction_btn.setEnabled("extraction_image_display" in self.links)
        self.wcs_btn.setEnabled("wcs_file" in self.links)
        self.new_fits_btn.setEnabled("new_fits_file" in self.links)

    def _open_in_webview(self, key: str, title: str) -> None:
        url = self.links.get(key)
        if url:
            self.open_webview_requested.emit(title, url)

    def _open_external(self, key: str) -> None:
        url = self.links.get(key)
        if url:
            webbrowser.open(url, new=2)

    def _fill_list(self, widget: QListWidget, values: list[str]) -> None:
        widget.clear()
        widget.addItems(values)

    def _value_label(self) -> QLabel:
        label = QLabel("-")
        return label

    def _fmt_float(self, value: Optional[float], suffix: str, precision: int) -> str:
        if value is None:
            return "-"
        return f"{value:.{precision}f}{suffix}"

    def _fmt_int(self, value: Optional[int]) -> str:
        if value is None:
            return "-"
        return str(value)
