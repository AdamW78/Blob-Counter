from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QSlider, QLineEdit
from PySide6.QtCore import Qt

class UIUtils:
    @staticmethod
    def create_slider_with_input(name, min_value, max_value, initial_value, tooltip):
        layout = QVBoxLayout()

        # Create HBox for label and input field
        hbox = QHBoxLayout()
        label = QLabel(name)
        input_field = QLineEdit(str(initial_value))
        input_field.setFixedWidth(50)

        # Add label and input field to HBox, anchoring them at opposite ends
        hbox.addWidget(label)
        hbox.addStretch()
        hbox.addWidget(input_field)

        # Add HBox and slider to the main VBox layout
        layout.addLayout(hbox)
        slider_hbox = QHBoxLayout()
        min_label = QLabel(str(min_value))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_value, max_value)
        slider.setValue(initial_value)
        slider.setToolTip(tooltip)
        max_label = QLabel(str(max_value))

        # Add min label, slider, and max label to HBox
        slider_hbox.addWidget(min_label)
        slider_hbox.addWidget(slider)
        slider_hbox.addWidget(max_label)

        # Add slider HBox to the main VBox layout
        layout.addLayout(slider_hbox)

        slider.valueChanged.connect(lambda value: input_field.setText(str(value)))
        input_field.editingFinished.connect(
            lambda: slider.setValue(int(input_field.text()) if input_field.text().isdigit() else min_value))

        return layout, input_field