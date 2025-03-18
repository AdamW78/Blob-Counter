import concurrent.futures
import logging

from PySide6.QtGui import QWheelEvent

from ui_utils import UIUtils

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
import math
import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
from collections import namedtuple
import cv2
from PySide6.QtCore import Qt, QEvent, QThread, Signal, QObject
from PySide6.QtWidgets import QListWidget, QFileDialog, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, \
    QApplication, QLabel, QSlider, QLineEdit, QGroupBox, QStackedWidget, QCheckBox, QProgressDialog, QMessageBox

import logger
from blob_detector_logic import BlobDetectorLogic
from blob_detector_ui import BlobDetectorUI
from excel_output import ExcelOutput
from logger import LOGGER
from utils import DEFAULT_DILUTION, IMAGE_LIST_WIDGET_WIDTH, TOOLTIP_MIN_AREA, TOOLTIP_MAX_AREA, \
    TOOLTIP_MIN_CIRCULARITY, TOOLTIP_MIN_CONVEXITY, TOOLTIP_MIN_INERTIA_RATIO, TOOLTIP_MIN_THRESHOLD, \
    TOOLTIP_MAX_THRESHOLD, TOOLTIP_MIN_DIST_BETWEEN_BLOBS

Timepoint = namedtuple("Timepoint", ["image_path", "keypoints", "day", "dilution", "sample_number"])

# image_set_reader.py
class BlobCounterWorker(QObject):
    finished = Signal()
    error = Signal(str)

    def __init__(self, blob_detector_logic, min_area, max_area, min_circularity, min_dist_between_blobs,
                 min_threshold, max_threshold, apply_gaussian_blur, apply_morphological_operations):
        super().__init__()
        self.blob_detector_logic = blob_detector_logic
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.min_dist_between_blobs = min_dist_between_blobs
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.apply_gaussian_blur = apply_gaussian_blur
        self.apply_morphological_operations = apply_morphological_operations

    def run(self):
        try:
            self.blob_detector_logic.update_blob_count(self.min_area, self.max_area, self.min_circularity, self.min_convexity, self.min_inertia_ratio,
                                  self.min_dist_between_blobs, self.apply_gaussian_blur, self.apply_morphological_operations)
            self.blob_detector_logic.update_blob_count(
                self.min_area, self.max_area, self.min_circularity, self.min_dist_between_blobs,
                self.min_threshold, self.max_threshold, self.apply_gaussian_blur,
                self.apply_morphological_operations
            )
            self.blob_detector_logic.update_timepoint()
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

class ImageSetBlobDetector(QWidget):
    def __init__(self):
        super().__init__()
        self.blob_detector_logic_list= []
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
        self.progress_dialog = None

        # Add universal blob detector settings
        self.create_universal_blob_detector_settings()

        # Add a blank BlobDetectorUI widget initially
        blank_blob_detector_logic = BlobDetectorLogic()
        blank_blob_detector_ui = BlobDetectorUI(blank_blob_detector_logic)
        self.blob_detector_stack.addWidget(blank_blob_detector_ui)

        # Install event filters for zooming
        self.installEventFilter(self)

    # noinspection PyTypeChecker
    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.Gesture:
            return self.handle_gesture_event(event)
        elif event.type() == QEvent.Type.Wheel:
            event = event if isinstance(event, QWheelEvent) else None
            current_widget = self.blob_detector_stack.currentWidget()
            if isinstance(current_widget, BlobDetectorUI):
                current_widget.handle_wheel_zoom(event)
                return True
        elif event.type() == QEvent.Type.KeyPress:
            current_widget = self.blob_detector_stack.currentWidget()
            if isinstance(current_widget, BlobDetectorUI):
                current_widget.handle_key_zoom(event)
                return True
        return super().eventFilter(source, event)

    def handle_gesture_event(self, event):
        gesture = event.gesture(Qt.PanGesture)
        if gesture:
            delta = gesture.delta()
            self.blob_detector_stack.horizontalScrollBar().setValue(
                self.blob_detector_stack.horizontalScrollBar().value() - delta.x()
            )
            self.blob_detector_stack.verticalScrollBar().setValue(
                self.blob_detector_stack.verticalScrollBar().value() - delta.y()
            )
            return True
        return False

        return layout, input_field

    def create_universal_blob_detector_settings(self):
        self.controls_group_box = QGroupBox("Blob Detection Parameters - All Images")
        self.controls_layout = QVBoxLayout()
        self.min_area_slider, self.min_area_input = UIUtils.create_slider_with_input('Min Area', 0, 5000, 144, TOOLTIP_MIN_AREA)
        self.max_area_slider, self.max_area_input = UIUtils.create_slider_with_input('Max Area', 0, 5000, 5000, TOOLTIP_MAX_AREA)
        self.min_circularity_slider, self.min_circularity_input = UIUtils.create_slider_with_input('Min Circularity', 0, 100, 60, TOOLTIP_MIN_CIRCULARITY)
        self.min_convexity_slider, self.min_convexity_input = UIUtils.create_slider_with_input('Min Convexity', 0, 100, 50, TOOLTIP_MIN_CONVEXITY)
        self.min_inertia_ratio_slider, self.min_inertia_ratio_input = UIUtils.create_slider_with_input('Min Inertia Ratio', 0, 100, 50, TOOLTIP_MIN_INERTIA_RATIO)
        self.min_dist_between_blobs_slider, self.min_dist_between_blobs_input = UIUtils.create_slider_with_input('Min Dist Between Blobs', 0, 100, 10, TOOLTIP_MIN_DIST_BETWEEN_BLOBS)
        self.min_threshold_slider, self.min_threshold_input = UIUtils.create_slider_with_input('Min Threshold', 0, 255, 10, TOOLTIP_MIN_THRESHOLD)
        self.max_threshold_slider, self.max_threshold_input = UIUtils.create_slider_with_input('Max Threshold', 0, 255, 200, TOOLTIP_MAX_THRESHOLD)

        self.controls_layout.addLayout(self.min_area_slider)
        self.controls_layout.addLayout(self.max_area_slider)
        self.controls_layout.addLayout(self.min_circularity_slider)
        self.controls_layout.addLayout(self.min_convexity_slider)
        self.controls_layout.addLayout(self.min_inertia_ratio_slider)
        self.controls_layout.addLayout(self.min_dist_between_blobs_slider)
        self.controls_layout.addLayout(self.min_threshold_slider)
        self.controls_layout.addLayout(self.max_threshold_slider)

        self.gaussian_blur_checkbox = QCheckBox('Apply Gaussian Blur')
        self.morphological_operations_checkbox = QCheckBox('Apply Morphological Operations')
        self.controls_layout.addWidget(self.gaussian_blur_checkbox)
        self.controls_layout.addWidget(self.morphological_operations_checkbox)

        self.update_all_button = QPushButton('Update Blob Count for All Images')
        self.update_all_button.clicked.connect(self.count_all_blobs)
        self.controls_layout.addWidget(self.update_all_button)

        self.open_folder_button = QPushButton('Open Image Set Folder...')
        self.open_folder_button.clicked.connect(self.open_folder)
        self.controls_layout.addWidget(self.open_folder_button)

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
        if not self.image_paths:
            LOGGER().warning("No images found in the selected folder.")
            QMessageBox.warning(self, "No Images Found", "No images found in the selected folder.")
            return
        self.image_list_widget.clear()
        self.timepoints.clear()

        # Remove all widgets from the stack
        while self.blob_detector_stack.count() > 0:
            self.blob_detector_stack.removeWidget(self.blob_detector_stack.widget(0))

        progress_dialog = self.show_progress_dialog(len(self.image_paths), "Loading Images...", "Cancel")
        QApplication.processEvents()

        for i, image_path in enumerate(self.image_paths):
            blob_detector_ui = BlobDetectorUI(BlobDetectorLogic(image_path))
            blob_detector_logic = blob_detector_ui.blob_detector_logic
            blob_detector_ui.keypoints_changed.connect(self.update_displayed_blob_counts)
            self.blob_detector_stack.addWidget(blob_detector_ui)
            self.add_to_image_list(blob_detector_logic)
            self.blob_detector_logic_list.append(blob_detector_logic)

            progress_dialog.setValue(i + 1)
            QApplication.processEvents()

        progress_dialog.close()
        self.blob_detector_logic_list.sort(key=lambda x: x.get_timepoint().sample_number if x.get_timepoint() else -1)

        # Automatically select the top image
        if self.image_list_widget.count() > 0:
            self.image_list_widget.setCurrentRow(0)
            self.display_selected_image(self.image_list_widget.item(0))

        self.update_image_list()
        if self.timepoints:
            self.export_button.setEnabled(True)

    def display_selected_image(self, item):
        selected_index = self.image_list_widget.row(item)
        self.blob_detector_stack.setCurrentIndex(selected_index)
        current_widget = self.blob_detector_stack.currentWidget()
        if isinstance(current_widget, BlobDetectorUI):
            current_widget.update_display_image()

    def show_progress_dialog(self, max_value, message, cancel_button_text):
        progress_dialog = QProgressDialog(message, cancel_button_text, 0, max_value, self)
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setValue(0)
        progress_dialog.show()
        logging.debug("Progress dialog initialized and shown")
        return progress_dialog

    def count_all_blobs(self):
        # Update universal settings
        min_area = int(self.min_area_input.text())
        max_area = int(self.max_area_input.text())
        min_circularity = int(self.min_circularity_input.text()) / 100.0
        min_dist_between_blobs = int(self.min_dist_between_blobs_input.text())
        min_threshold = int(self.min_threshold_input.text())
        max_threshold = int(self.max_threshold_input.text())
        apply_gaussian_blur = self.gaussian_blur_checkbox.isChecked()
        apply_morphological_operations = self.morphological_operations_checkbox.isChecked()

        progress_dialog = self.show_progress_dialog(len(self.blob_detector_logic_list), "Counting Blobs...", "Cancel")
        progress_dialog.setValue(0)
        QApplication.processEvents()

        self.threads = []
        self.workers = []

        def update_progress():
            progress_dialog.setValue(progress_dialog.value() + 1)
            if progress_dialog.value() == len(self.blob_detector_logic_list):
                progress_dialog.close()
            QApplication.processEvents()

        for logic in self.blob_detector_logic_list:
            worker = BlobCounterWorker(logic, min_area, max_area, min_circularity, min_dist_between_blobs,
                                       min_threshold, max_threshold, apply_gaussian_blur,
                                       apply_morphological_operations)
            thread = QThread()
            worker.moveToThread(thread)
            worker.finished.connect(update_progress)
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            worker.error.connect(lambda e: logging.error(f"Error counting blobs: {e}"))
            thread.started.connect(worker.run)
            self.threads.append(thread)
            self.workers.append(worker)
            thread.start()

        self.update_displayed_blob_counts()

    def update_displayed_blob_counts(self):
        for i in range(self.blob_detector_stack.count()):
            widget = self.blob_detector_stack.widget(i)
            if isinstance(widget, BlobDetectorUI):
                widget.update_display_image()

    def add_to_image_list(self, blob_detector_logic):
        list_name = blob_detector_logic.get_custom_name(DEFAULT_DILUTION)
        self.image_list_widget.addItem(f"{list_name} - Keypoints: {len(blob_detector_logic.keypoints)}")
        self.timepoints.append(blob_detector_logic.get_timepoint())

    def extract_sample_number(self, item_text):
        parts = item_text.split(' ')
        for part in parts:
            if part.startswith("Sample") and part[6:].isdigit():
                return int(part[6:])
        return -1  # Default value if no sample number is found

    def update_image_list(self):
        old_selected_index = self.image_list_widget.selectedIndexes()[
            0].row() if self.image_list_widget.selectedIndexes() else 0
        self.image_list_widget.clear()
        timepoints_with_widgets = []
        for i in range(self.blob_detector_stack.count()):
            widget = self.blob_detector_stack.widget(i)
            if isinstance(widget, BlobDetectorUI):
                widget.blob_detector_logic.update_timepoint()
                timepoints_with_widgets.append((widget.blob_detector_logic.get_timepoint(), widget))



        # Sort timepoints and widgets by sample number
        timepoints_with_widgets.sort(key=lambda x: x[0].sample_number if x[0] else -1)

        # Re-order widgets in the stack
        for index, (timepoint, widget) in enumerate(timepoints_with_widgets):
            self.blob_detector_stack.removeWidget(widget)
            self.blob_detector_stack.insertWidget(index, widget)
            list_name = widget.blob_detector_logic.get_custom_name(DEFAULT_DILUTION)
            self.image_list_widget.addItem(f"{list_name} - Keypoints: {len(widget.blob_detector_logic.keypoints)}")

        # Ensure the correct widget is selected
        self.image_list_widget.setCurrentRow(old_selected_index)
        self.blob_detector_stack.setCurrentIndex(old_selected_index)

    def export_blob_counts(self):
        if not self.timepoints:
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Blob Counts", "", "Excel Files (*.xlsx)")
        if not file_path:
            return

        # Update timepoints with the latest keypoint counts
        self.timepoints.clear()
        for i in range(self.blob_detector_stack.count()):
            widget = self.blob_detector_stack.widget(i)
            if isinstance(widget, BlobDetectorUI):
                widget.blob_detector_logic.update_timepoint()
                self.timepoints.append(widget.blob_detector_logic.get_timepoint())

        self.timepoints.sort(
            key=lambda x: x.sample_number if x else math.inf)  # Ensure timepoints are sorted by sample number
        excel_output = ExcelOutput(file_path)
        for timepoint in self.timepoints:
            if timepoint is not None:
                excel_output.write_blob_counts(timepoint.day, timepoint.sample_number, timepoint.num_keypoints)
        excel_output.save()

        self.save_all_keypoints_as_xml()

        for timepoint in self.timepoints:
            if timepoint is not None:
                self.save_image_with_keypoints(timepoint)
        LOGGER().info("Images with counted keypoints saved to disk.")

        logger.LOGGER().info("Blob counts exported to Excel.")

    def save_image_with_keypoints(self, timepoint):
        for i in range(self.blob_detector_stack.count()):
            widget = self.blob_detector_stack.widget(i)
            if isinstance(widget, BlobDetectorUI) and widget.blob_detector_logic.get_timepoint() == timepoint:
                image_with_keypoints = widget.blob_detector_logic.get_display_image()
                day_folder = os.path.join("counted_images", f"Day {timepoint.day}")
                os.makedirs(day_folder, exist_ok=True)
                image_path = os.path.join(day_folder, f"Sample_{timepoint.sample_number}.png")
                cv2.imwrite(image_path, cv2.cvtColor(image_with_keypoints, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR
                break


    def save_all_keypoints_as_xml(self):
        root = ET.Element("Keypoints")
        day_element = None
        for i in range(self.blob_detector_stack.count()):
            widget = self.blob_detector_stack.widget(i)
            if isinstance(widget, BlobDetectorUI):
                keypoints = widget.blob_detector_logic.keypoints
                timepoint = widget.blob_detector_logic.get_timepoint()
                if timepoint is None:
                    continue
                if day_element is None or day_element.get("number") != str(timepoint.day):
                    day_element = ET.SubElement(root, "Day")
                    day_element.set("number", str(timepoint.day))
                sample_element = ET.SubElement(day_element, "Sample")
                sample_element.set("number", str(timepoint.sample_number))
                for keypoint in keypoints:
                    kp_element = ET.SubElement(sample_element, "Keypoint")
                    ET.SubElement(kp_element, "X").text = str(keypoint.pt[0])
                    ET.SubElement(kp_element, "Y").text = str(keypoint.pt[1])
                    ET.SubElement(kp_element, "Size").text = str(keypoint.size)

        xml_path = os.path.join("counted_images", "keypoints.xml")

        # Pretty print the XML
        xml_str = ET.tostring(root, encoding='utf-8')
        parsed_xml = xml.dom.minidom.parseString(xml_str)
        pretty_xml_str = parsed_xml.toprettyxml(indent="  ")

        with open(xml_path, "w") as f:
            f.write(pretty_xml_str)
        LOGGER().debug("Keypoints exported to XML and saved to disk.")

if __name__ == "__main__":
    app = QApplication([])
    image_set_blob_detector = ImageSetBlobDetector()
    image_set_blob_detector.show()
    app.exec()