import os
import re

import cv2
from PySide6.QtWidgets import QListWidget, QFileDialog, QScrollArea, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QApplication, QLabel, QSlider, QLineEdit, QGroupBox
from PySide6.QtCore import Qt

from image_reader import BlobDetector

IMAGE_LIST_WIDGET_WIDTH = 300

class ImageSetBlobDetector(QWidget):
    def __init__(self):
        super().__init__()
        self.image_paths = []
        self.blob_detectors = []
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Set Blob Detector")
        self.layout = QHBoxLayout(self)

        # Add universal blob detector settings
        self.create_universal_blob_detector_settings()

        # Add image list widget
        self.create_image_list_widget()

        # Add blob detector UI
        self.blob_detector_widget = BlobDetector()
        self.layout.addWidget(self.blob_detector_widget)

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

        self.controls_layout.addLayout(self.min_area_slider)
        self.controls_layout.addLayout(self.max_area_slider)
        self.controls_layout.addLayout(self.min_circularity_slider)

        # Add button to update blob count for all images
        self.update_all_button = QPushButton('Update Blob Count for All Images')
        self.update_all_button.clicked.connect(self.update_blob_count_for_all_images)
        self.controls_layout.addWidget(self.update_all_button)

        # Add button to open folder below the scroll pane
        self.open_folder_button = QPushButton('Open Image Set Folder...')
        self.open_folder_button.clicked.connect(self.open_folder)
        self.controls_layout.addWidget(self.open_folder_button)

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
        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_paths.sort(key=self.sort_key)
        self.image_list_widget.clear()
        self.blob_detectors = []
        for image_path in self.image_paths:
            detector = BlobDetector(image_path)
            self.blob_detectors.append(detector)
            self.image_list_widget.addItem(f"{os.path.basename(image_path)} - Keypoints: {len(detector.keypoints)}")

        # Automatically select the top image
        if self.image_list_widget.count() > 0:
            self.image_list_widget.setCurrentRow(0)
            self.display_selected_image(self.image_list_widget.item(0))

    def sort_key(self, filename):
        base_name = os.path.basename(filename)
        parts = re.split(r'[_\s]', base_name)
        numbers = [int(part) for part in parts if part.isdigit()]
        if len(numbers) >= 2:
            return (numbers[0], numbers[1])
        elif len(numbers) == 1:
            return (numbers[0],)
        else:
            return base_name

    def display_selected_image(self, item):
        selected_image_path = os.path.join(os.path.dirname(self.image_paths[0]), item.text().split(' - ')[0])
        for detector in self.blob_detectors:
            if detector.image_path == selected_image_path:
                self.layout.removeWidget(self.blob_detector_widget)
                self.blob_detector_widget.deleteLater()
                self.blob_detector_widget = detector
                self.layout.addWidget(self.blob_detector_widget)
                break

    def update_blob_count_for_all_images(self):
        for detector in self.blob_detectors:
            detector.params.minArea = int(self.min_area_input.text())
            detector.params.maxArea = int(self.max_area_input.text())
            detector.params.minCircularity = int(self.min_circularity_input.text()) / 100.0
            detector.detector = cv2.SimpleBlobDetector_create(detector.params)
            detector.keypoints = list(detector.detect_blobs())
            detector.update_display_image()
        self.update_image_list()

    def update_image_list(self):
        self.image_list_widget.clear()
        for detector in self.blob_detectors:
            self.image_list_widget.addItem(f"{os.path.basename(detector.image_path)} - Keypoints: {len(detector.keypoints)}")

if __name__ == "__main__":
    app = QApplication([])
    image_set_blob_detector = ImageSetBlobDetector()
    image_set_blob_detector.show()
    app.exec()