import os
from collections import namedtuple

from PySide6.QtCore import Qt, QEvent
from PySide6.QtWidgets import QListWidget, QFileDialog, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, \
    QApplication, QLabel, QSlider, QLineEdit, QGroupBox, QStackedWidget, QCheckBox

import logger
from blob_detector_logic import BlobDetectorLogic
from blob_detector_ui import BlobDetectorUI
from excel_output import ExcelOutput
from utils import DEFAULT_DILUTION, IMAGE_LIST_WIDGET_WIDTH

Timepoint = namedtuple("Timepoint", ["image_path", "keypoints", "day", "dilution"])

class ImageSetBlobDetector(QWidget):
    def __init__(self):
        super().__init__()
        self.image_paths = []
        self.timepoints = []
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Set Blob Detector")
        self.layout = QHBoxLayout(self)

        # Add image list widget
        self.create_image_list_widget()

        # Add blob detector UI
        self.blob_detector_stack = QStackedWidget()
        self.layout.addWidget(self.blob_detector_stack)

        # Add universal blob detector settings
        self.create_universal_blob_detector_settings()

        # Add a blank BlobDetectorUI widget initially
        blank_blob_detector_logic = BlobDetectorLogic()
        blank_blob_detector_ui = BlobDetectorUI(blank_blob_detector_logic)
        self.blob_detector_stack.addWidget(blank_blob_detector_ui)

        # Install event filters for zooming
        self.installEventFilter(self)


    def eventFilter(self, source, event):
        if event.type() == QEvent.Wheel:
            current_widget = self.blob_detector_stack.currentWidget()
            if isinstance(current_widget, BlobDetectorUI):
                current_widget.handle_wheel_zoom(event)
                return True
        elif event.type() == QEvent.KeyPress:
            current_widget = self.blob_detector_stack.currentWidget()
            if isinstance(current_widget, BlobDetectorUI):
                current_widget.handle_key_zoom(event)
                return True
        return super().eventFilter(source, event)

    def create_slider_with_input(self, name, min_value, max_value, initial_value):
        layout = QHBoxLayout()
        label = QLabel(name)
        layout.addWidget(label)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_value, max_value)
        slider.setValue(initial_value)
        layout.addWidget(slider)
        input_field = QLineEdit(str(initial_value))
        input_field.setFixedWidth(50)
        layout.addWidget(input_field)

        slider.valueChanged.connect(lambda value: input_field.setText(str(value)))
        input_field.editingFinished.connect(lambda: slider.setValue(int(input_field.text()) if input_field.text().isdigit() else min_value))

        return layout, input_field

    def create_universal_blob_detector_settings(self):
        # Add universal sliders and text inputs in a group box
        self.controls_group_box = QGroupBox("Blob Detection Parameters - All Images")
        self.controls_layout = QVBoxLayout()
        self.min_area_slider, self.min_area_input = self.create_slider_with_input('Min Area', 0, 5000, 144)
        self.max_area_slider, self.max_area_input = self.create_slider_with_input('Max Area', 0, 5000, 5000)
        self.min_circularity_slider, self.min_circularity_input = self.create_slider_with_input('Min Circularity', 0, 100, 60)
        self.min_dist_between_blobs_slider, self.min_dist_between_blobs_input = self.create_slider_with_input('Min Dist Between Blobs', 0, 100, 10)
        self.min_threshold_slider, self.min_threshold_input = self.create_slider_with_input('Min Threshold', 0, 255, 10)
        self.max_threshold_slider, self.max_threshold_input = self.create_slider_with_input('Max Threshold', 0, 255, 200)

        self.controls_layout.addLayout(self.min_area_slider)
        self.controls_layout.addLayout(self.max_area_slider)
        self.controls_layout.addLayout(self.min_circularity_slider)
        self.controls_layout.addLayout(self.min_dist_between_blobs_slider)
        self.controls_layout.addLayout(self.min_threshold_slider)
        self.controls_layout.addLayout(self.max_threshold_slider)

        # Add checkboxes for preprocessing features
        self.gaussian_blur_checkbox = QCheckBox('Apply Gaussian Blur')
        self.morphological_operations_checkbox = QCheckBox('Apply Morphological Operations')
        self.controls_layout.addWidget(self.gaussian_blur_checkbox)
        self.controls_layout.addWidget(self.morphological_operations_checkbox)

        # Add button to update blob count for all images
        self.update_all_button = QPushButton('Update Blob Count for All Images')
        self.update_all_button.clicked.connect(self.update_blob_count_for_all_images)
        self.controls_layout.addWidget(self.update_all_button)

        # Add button to open folder below the scroll pane
        self.open_folder_button = QPushButton('Open Image Set Folder...')
        self.open_folder_button.clicked.connect(self.open_folder)
        self.controls_layout.addWidget(self.open_folder_button)

        # Add export button below the open folder button
        self.export_button = QPushButton('Export Blob Counts')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_blob_counts)
        self.controls_layout.addWidget(self.export_button)

        self.controls_group_box.setLayout(self.controls_layout)
        self.layout.addWidget(self.controls_group_box)

    def create_image_list_widget(self):
        self.image_list_group_box = QGroupBox("Image List")
        self.image_list_layout = QVBoxLayout()
        self.image_list_widget = QListWidget()
        self.image_list_widget.setMinimumWidth(IMAGE_LIST_WIDGET_WIDTH)
        self.image_list_widget.setMaximumWidth(IMAGE_LIST_WIDGET_WIDTH)
        self.image_list_widget.itemClicked.connect(self.display_selected_image)
        self.image_list_layout.addWidget(self.image_list_widget)
        self.image_list_group_box.setLayout(self.image_list_layout)
        self.layout.addWidget(self.image_list_group_box)

    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.load_images_from_folder(folder_path)

    def load_images_from_folder(self, folder_path):
        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                            f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_paths.sort(key=self.sort_key)
        self.image_list_widget.clear()
        self.timepoints.clear()

        # Remove all widgets from the stack
        while self.blob_detector_stack.count() > 0:
            self.blob_detector_stack.removeWidget(self.blob_detector_stack.widget(0))

        for image_path in self.image_paths:
            blob_detector_logic = BlobDetectorLogic(image_path)
            blob_detector_ui = BlobDetectorUI(blob_detector_logic)
            blob_detector_ui.keypoints_changed.connect(self.update_image_list)
            self.blob_detector_stack.addWidget(blob_detector_ui)
            self.add_to_image_list(blob_detector_logic)

        # Automatically select the top image
        if self.image_list_widget.count() > 0:
            self.image_list_widget.setCurrentRow(0)
            self.display_selected_image(self.image_list_widget.item(0))

        i = 0
        while i < len(self.timepoints):
            if self.timepoints[i] is None:
                self.timepoints.pop(i)
            else:
                i += 1
        self.timepoints.sort(key=lambda x: x.sample_number, reverse=False)
        if self.timepoints:
            self.export_button.setEnabled(True)

    def sort_key(self, filename):
        # Implement your sorting logic here
        return filename

    def display_selected_image(self, item):
        selected_index = self.image_list_widget.row(item)
        self.blob_detector_stack.setCurrentIndex(selected_index)
        current_widget = self.blob_detector_stack.currentWidget()
        if isinstance(current_widget, BlobDetectorUI):
            current_widget.update_display_image()

    def update_blob_count_for_all_images(self):
        for i in range(self.blob_detector_stack.count()):
            widget = self.blob_detector_stack.widget(i)
            if isinstance(widget, BlobDetectorUI):
                widget.update_blob_count()
        self.update_image_list()

    def add_to_image_list(self, blob_detector_logic):
        list_name = blob_detector_logic.get_custom_name(DEFAULT_DILUTION)
        self.image_list_widget.addItem(f"{list_name} - Keypoints: {len(blob_detector_logic.keypoints)}")
        self.timepoints.append(blob_detector_logic.get_timepoint())

    def update_image_list(self):
        self.image_list_widget.clear()
        for i in range(self.blob_detector_stack.count()):
            widget = self.blob_detector_stack.widget(i)
            if isinstance(widget, BlobDetectorUI):
                list_name = widget.blob_detector_logic.get_custom_name(DEFAULT_DILUTION)
                self.image_list_widget.addItem(f"{list_name} - Keypoints: {len(widget.blob_detector_logic.keypoints)}")
        if self.image_list_widget.count() > 0:
            self.image_list_widget.setCurrentRow(0)

    def export_blob_counts(self):
        if not self.timepoints:
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Blob Counts", "", "Excel Files (*.xlsx)")
        if not file_path:
            return

        excel_output = ExcelOutput(file_path)
        for timepoint in self.timepoints:
            excel_output.write_blob_counts(timepoint.day, timepoint.sample_number, timepoint.num_keypoints)
        excel_output.save()
        logger.LOGGER().info("Blob counts exported to Excel.")

if __name__ == "__main__":
    app = QApplication([])
    image_set_blob_detector = ImageSetBlobDetector()
    image_set_blob_detector.show()
    app.exec()