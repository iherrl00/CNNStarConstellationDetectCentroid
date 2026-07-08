APP_STYLE = """
QWidget {
    background: #10141a;
    color: #e6e9ef;
    font-size: 12px;
}
QGroupBox {
    border: 1px solid #2d3642;
    border-radius: 6px;
    margin-top: 8px;
    padding: 8px;
    background: #141b23;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 3px;
}
QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget, QTableWidget, QTabWidget::pane {
    background: #0d1117;
    border: 1px solid #2b3340;
    border-radius: 4px;
    padding: 4px;
}
QPushButton {
    background: #1f6feb;
    border: none;
    border-radius: 5px;
    padding: 6px 12px;
    color: #ffffff;
}
QPushButton:hover { background: #2c7df7; }
QPushButton:disabled { background: #3b4453; color: #b4bdc8; }
QLabel {
    border: 1px solid #2b3340;
    border-radius: 4px;
    padding: 2px;
    background: #0d1117;
}
QHeaderView::section {
    background: #1a2230;
    color: #e6e9ef;
    border: 1px solid #2b3340;
    padding: 4px;
}
QProgressBar {
    border: 1px solid #2b3340;
    border-radius: 5px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #1f6feb;
}
"""
