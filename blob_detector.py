import os
import cv2
import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import Qt, QPointF, QEvent, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QLineEdit, QPushButton, QCheckBox, \
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QApplication

import logger
from undo_redo_tracker import UndoRedoTracker, ActionType
from utils import *

class BlobDetector(QWidget):
    keypoints_changed = Signal(int)

    def __init__(self, image_path=None, display_scale_factor=DEFAULT_DISPLAY_SCALE_FACTOR):
        super().__init__()
        self.contours = []
        self.custom_name = None
        self.image_path = image_path
        self.display_scale_factor = display_scale_factor
        self.current_scale_factor = 1.0
        self.undo_redo_tracker = UndoRedoTracker()
        self.keypoints = None
        self.image = None
        self.initUI()

        if image_path is not None:
            self.image = self.load_image()
            self.gray_image = self.convert_to_grayscale()
            self.params = self.create_blob_detector_params()
            self.detector = cv2.SimpleBlobDetector_create(self.params)
            self.detect_blobs()
            self.blobs = self.draw_blobs()
            self.timepoint = None
            self.sample_number = -1
            self.day_num = -1
            self.dilution = None
            self.get_custom_name()
            self.update_timepoint()
            self.keypoint_count_label.setText(f'Keypoints: {len(self.keypoints) + len(self.contours)}')
            self.update_display_image()
        else:
            self.image = None
            self.gray_image = None
            self.params = None
            self.detector = None
            self.keypoints = None
            self.blobs = None

        self.is_dragging = False
        self.mouse_is_pressed = False
        self.mouse_press_position = QPointF()

    def initUI(self):
        self.setWindowTitle("Blob Detector")
        self.layout = QVBoxLayout(self)

        if self.keypoints is not None:
            self.keypoint_count_label = QLabel(f'Keypoints: {len(self.keypoints)}')
        else:
            self.keypoint_count_label = QLabel('Keypoints: 0')
        self.layout.addWidget(self.keypoint_count_label)

        self.min_area_slider, self.min_area_input = self.create_slider_with_input('Min Area', MIN_AREA, MAX_AREA, DEFAULT_MIN_AREA)
        self.max_area_slider, self.max_area_input = self.create_slider_with_input('Max Area', MIN_AREA, MAX_AREA, DEFAULT_MAX_AREA)
        self.min_circularity_slider, self.min_circularity_input = self.create_slider_with_input('Min Circularity', 0, 100, int(DEFAULT_MIN_CIRCULARITY * 100))
        self.min_convexity_slider, self.min_convexity_input = self.create_slider_with_input('Min Convexity', 0, 100, int(DEFAULT_MIN_CONVEXITY * 100))
        self.min_inertia_ratio_slider, self.min_inertia_ratio_input = self.create_slider_with_input('Min Inertia Ratio', 0, 100, int(DEFAULT_MIN_INERTIA_RATIO * 100))
        self.min_dist_between_blobs_slider, self.min_dist_between_blobs_input = self.create_slider_with_input('Min Dist Between Blobs', MIN_DISTANCE_BETWEEN_BLOBS, MAX_DISTANCE_BETWEEN_BLOBS, DEFAULT_MIN_DISTANCE_BETWEEN_BLOBS)

        self.layout.addLayout(self.min_area_slider)
        self.layout.addLayout(self.max_area_slider)
        self.layout.addLayout(self.min_circularity_slider)
        self.layout.addLayout(self.min_convexity_slider)
        self.layout.addLayout(self.min_inertia_ratio_slider)
        self.layout.addLayout(self.min_dist_between_blobs_slider)

        self.gaussian_blur_checkbox = QCheckBox('Apply Gaussian Blur')
        self.morphological_operations_checkbox = QCheckBox('Apply Morphological Operations')
        self.layout.addWidget(self.gaussian_blur_checkbox)
        self.layout.addWidget(self.morphological_operations_checkbox)

        self.recount_button = QPushButton('Recount Blobs')
        self.recount_button.clicked.connect(self.update_blob_count)
        self.layout.addWidget(self.recount_button)

        self.graphics_view = QGraphicsView(self)
        self.graphics_view.setFixedSize(GRAPHICS_VIEW_WIDTH, GRAPHICS_VIEW_HEIGHT)
        self.graphics_scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.graphics_scene)
        self.layout.addWidget(self.graphics_view)

        self.pixmap_item = QGraphicsPixmapItem()
        self.graphics_scene.addItem(self.pixmap_item)

        if self.image is not None:
            self.update_display_image()

        self.graphics_view.setDragMode(QGraphicsView.NoDrag)
        self.graphics_view.viewport().installEventFilter(self)
        self.installEventFilter(self)

    def get_dilution_string(self, dilution_str: str, default_dilution: str):
        if dilution_str.find("1st") != -1:
            self.dilution = "x10"
            return "x10 dilution"
        elif dilution_str.find("2nd") != -1:
            self.dilution = "x100"
            return "x100 dilution"
        elif dilution_str.find("3rd") != -1:
            self.dilution = "x1000"
            return "x1000 dilution"
        else:
            return 0, self.get_dilution_string(default_dilution, default_dilution)

    def handle_dilution_string(self, dilution_str: str, default_dilution: str) -> str:
        if USE_DILUTION:
            dilution_string = self.get_dilution_string(dilution_str, default_dilution)
            if isinstance(dilution_string, tuple):
                return f" - {dilution_string[1]}"
            else:
                return f" - {dilution_string}"
        else:
            return ""

    def update_timepoint(self):
        if self.day_num != -1 and self.sample_number != -1 and self.dilution is not None:
            self.timepoint = Timepoint(day=self.day_num, sample_number=self.sample_number, dilution=self.dilution, num_keypoints=len(self.keypoints))

    def get_timepoint(self):
        return self.timepoint

    def parse_day_number(self, day_num_str: str):
        index = 0
        while day_num_str[:index + 1].isdigit():
            index += 1
        self.day_num = int(day_num_str[:index])
        return self.day_num


    def check_if_int(self, string):
        try:
            int(string)
            return True
        except ValueError:
            return False

    def handle_sample_number(self, sample_number_str: str):
        if sample_number_str.isdigit():
            self.sample_number = int(sample_number_str)
            return f"Sample {self.sample_number}"


    def handle_day_src(self, day_source) -> str:
        day_string = ""
        if USE_DAY:
            if isinstance(day_source, list):
                day_string = day_source[0]
            elif isinstance(day_source, str):
                find_day = day_source.find(r"Day [0-9]{1,3}")
                if find_day != -1:
                    day_string_parts = day_source[find_day:].split(' ')
                    next_string = False
                    for index, part in enumerate(day_string_parts):
                        if next_string:
                            day_string = self.parse_day_number(part)
                        elif part.find("Day") != -1 and day_string_parts[index + 1][0].isdigit():
                            next_string = True
            else:
                logger.LOGGER().warning("Day source is incorrect type - expected list or string - not including day in timepoint display name...")
                logger.LOGGER().warning("Day source type: %s", type(day_source))
                return day_string
            if self.check_if_int(day_string):
                self.day_num = int(day_string)
                return f"Day {day_string} - "
            else:
                logger.LOGGER().warning("Unable to parse day string - NOT AN INTEGER - not including day in timepoint display name...")
                return ""
        return day_string

    def get_custom_name(self, default_dilution=DEFAULT_DILUTION):
        if self.custom_name is None:
            basename = os.path.basename(self.image_path)
            if basename.upper().endswith("LABEL.JPG"):
                logger.LOGGER().info("Skipping label image...")
                return basename
            parts = basename.split('_')
            folder_name = os.path.basename(os.path.dirname(self.image_path))
            str_val = ""
            if len(parts) == 2 or (len(parts) == 3 and parts[2].find("dilution") != -1):
                str_val += self.handle_day_src(folder_name)
                str_val += self.handle_sample_number(parts[0])
                str_val += self.handle_dilution_string(parts[1], default_dilution)
    
            elif len(parts) == 3 or (len(parts) == 4 and parts[3].find("dilution") != -1):
                day_string = self.handle_day_src(parts)
                if day_string == "":
                    day_string = self.handle_day_src(folder_name)
                str_val += day_string
                str_val += self.handle_sample_number(parts[1])
                str_val += self.handle_dilution_string(parts[2], default_dilution)
            else:
                logger.LOGGER().warning("Unable to parse image name - using file name...")
                str_val = os.path.basename(self.image_path)
            self.custom_name = str_val
            return str_val
        else:
            return self.custom_name

    def update_slider_input(self, slider, input_field, value):
        input_field.setText(str(value))

    def update_input_slider(self, input_field, slider, min_value):
        slider.setValue(int(input_field.text()) if input_field.text().isdigit() else min_value)

    def update_blob_count(self):
        # Update blob detection parameters
        self.params.minArea = int(self.min_area_input.text())
        self.params.maxArea = int(self.max_area_input.text())
        self.params.minCircularity = int(self.min_circularity_input.text()) / 100.0
        self.params.minConvexity = int(self.min_convexity_input.text()) / 100.0
        self.params.minInertiaRatio = int(self.min_inertia_ratio_input.text()) / 100.0
        self.params.minDistBetweenBlobs = int(self.min_dist_between_blobs_input.text())
        self.detector = cv2.SimpleBlobDetector_create(self.params)

        # Reload the image to reset any previous preprocessing
        self.gray_image = self.convert_to_grayscale()

        # Apply preprocessing if checkboxes are checked
        if self.gaussian_blur_checkbox.isChecked():
            self.gray_image = cv2.GaussianBlur(self.gray_image, (5, 5), 0)
        if self.morphological_operations_checkbox.isChecked():
            kernel = np.ones((5, 5), np.uint8)
            self.gray_image = cv2.erode(self.gray_image, kernel, iterations=1)
            self.gray_image = cv2.dilate(self.gray_image, kernel, iterations=1)

        # Detect blobs and update the display
        self.keypoints = list(self.detect_blobs())
        self.update_display_image()
        self.update_timepoint()
        self.keypoint_count_label.setText(f'Keypoints: {len(self.keypoints) + len(self.contours)}')

    def update_display_image(self):
        self.blobs = self.draw_blobs()
        image = cv2.cvtColor(self.blobs, cv2.COLOR_BGR2RGB)
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.pixmap_item.setPixmap(pixmap)
        self.graphics_scene.setSceneRect(0, 0, width, height)
        self.keypoint_count_label.setText(f'Keypoints: {len(self.keypoints)}')

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

    def load_image(self):
        image = cv2.imread(self.image_path, 1)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {self.image_path}")
        return image

    def convert_to_grayscale(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def create_blob_detector_params(self):
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = DEFAULT_MIN_AREA
        params.maxArea = DEFAULT_MAX_AREA
        params.filterByCircularity = True
        params.minCircularity = DEFAULT_MIN_CIRCULARITY
        params.filterByConvexity = True
        params.minConvexity = DEFAULT_MIN_CONVEXITY
        params.filterByInertia = True
        params.minInertiaRatio = DEFAULT_MIN_INERTIA_RATIO
        params.filterByColor = False
        params.blobColor = DEFAULT_BLOB_COLOR
        params.minDistBetweenBlobs = MIN_DISTANCE_BETWEEN_BLOBS
        params.minThreshold = DEFAULT_MIN_THRESHOLD
        params.maxThreshold = DEFAULT_MAX_THRESHOLD
        return params

    def preprocess_image(self, image):
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
        kernel = np.ones((5, 5), np.uint8)
        eroded_image = cv2.erode(blurred_image, kernel, iterations=1)
        dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)
        return dilated_image


    def apply_brightness_mask(self, image):
        _, mask = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)
        return cv2.bitwise_and(image, image, mask=mask)

    def detect_blobs(self):
        self.contours = []
        # Define kernel for morphological operations
        kernel = np.ones((5, 5), np.uint8)

        # Apply Gaussian Blur if the checkbox is checked
        if self.gaussian_blur_checkbox.isChecked():
            self.gray_image = cv2.GaussianBlur(self.gray_image, (5, 5), 0)
            print("Applied Gaussian Blur")

        # Apply morphological operations if the checkbox is checked
        if self.morphological_operations_checkbox.isChecked():
            self.gray_image = cv2.erode(self.gray_image, kernel, iterations=1)
            self.gray_image = cv2.dilate(self.gray_image, kernel, iterations=1)
            print("Applied Morphological Operations")

        # Remove small noise
        self.gray_image = cv2.morphologyEx(self.gray_image, cv2.MORPH_OPEN, kernel, iterations=2)
        print("Removed small noise")

        # Apply binary threshold
        _, binary_image = cv2.threshold(self.gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        print("Applied binary threshold")

        # Remove small white regions
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
        print("Removed small white regions")

        # Sure background area
        sure_bg = cv2.dilate(binary_image, kernel, iterations=3)
        print("Found sure background area")

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        print("Found sure foreground area")

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        print("Found unknown region")

        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 0] = 0
        print("Marker labelling completed")

        # Apply the Watershed algorithm
        try:
            markers = cv2.watershed(self.image, markers)
            self.image[markers == -1] = [255, 0, 0]
            print("Applied Watershed algorithm")
        except Exception as e:
            print(f"Error during Watershed algorithm: {e}")
            return []

        # Detect blobs using SimpleBlobDetector
        self.keypoints = [*self.detector.detect(self.gray_image)]
        print(f"Number of Blobs found: {len(self.keypoints)}")

        # Create a mask from the detected blobs
        mask = np.zeros_like(self.gray_image)
        for keypoint in self.keypoints:
            x, y = keypoint.pt
            radius = int(keypoint.size / 2)
            cv2.circle(mask, (int(x), int(y)), radius, (255), thickness=-1)

        # Invert the mask to get the areas not covered by blobs
        mask_inv = cv2.bitwise_not(mask)

        # Use the contour detector on the areas not covered by blobs
        contours, _ = cv2.findContours(mask_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.params.minArea <= area <= self.params.maxArea:
                # Filter by shape properties
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 0.8 <= aspect_ratio <= 1.2:  # Example: filter by aspect ratio
                    self.contours.append(contour)

        # Debug: Print the number of contours found and their areas
        print(f"Number of contours found: {len(self.contours)}")
        # for i, contour in enumerate(self.contours):
        #     print(f"Contour {i} area: {cv2.contourArea(contour)}")
        print(f"Number of keypoints found: {len(self.keypoints)}")

    def draw_blobs(self):
        image_with_keypoints = self.image.copy()
        for keypoint in self.keypoints:
            x, y = keypoint.pt
            radius = int(keypoint.size / 2)
            cv2.circle(image_with_keypoints, (int(x), int(y)), radius, CIRCLE_COLOR, CIRCLE_THICKNESS)

        # Draw circles around contours
        cv2.drawContours(image_with_keypoints, self.contours, -1, CIRCLE_COLOR, CIRCLE_THICKNESS)
        return image_with_keypoints

    def wheelEvent(self, event):
        zoom_in_factor = 1.1
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        new_scale_factor = self.current_scale_factor * zoom_factor
        if MIN_SCALE_FACTOR <= new_scale_factor <= MAX_SCALE_FACTOR:
            self.graphics_view.scale(zoom_factor, zoom_factor)
            self.current_scale_factor = new_scale_factor

    def eventFilter(self, source, event):
        if self.image is None:
            return super().eventFilter(source, event)
        if event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                self.mouse_press_position = event.position()
                if self.graphics_view.viewport().rect().contains(
                        self.graphics_view.mapFromGlobal(event.globalPosition().toPoint())):
                    self.mouse_is_pressed = True
                    self.is_dragging = False
                    return True
        elif event.type() == QEvent.MouseMove:
            if self.mouse_is_pressed and self.graphics_view.viewport().rect().contains(
                    self.graphics_view.mapFromGlobal(event.globalPosition().toPoint())):
                self.is_dragging = True
                delta = event.position() - self.mouse_press_position
                self.graphics_view.horizontalScrollBar().setValue(
                    self.graphics_view.horizontalScrollBar().value() - delta.x())
                self.graphics_view.verticalScrollBar().setValue(
                    self.graphics_view.verticalScrollBar().value() - delta.y())
                self.mouse_press_position = event.position()
                return True
        elif event.type() == QEvent.MouseButtonRelease:
            if event.button() == Qt.LeftButton:
                if not self.is_dragging and self.graphics_view.viewport().rect().contains(
                        self.graphics_view.mapFromGlobal(event.globalPosition().toPoint())):
                    self.add_or_remove_keypoint(event.position())
                    self.is_dragging = False
                    self.mouse_is_pressed = False
                    return True
        if (event.type() == QtCore.QEvent.ShortcutOverride and
                (event.modifiers() & (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier)) and
                event.key() == QtCore.Qt.Key_Z):
            return self.handle_undo_redo(event)
        return super().eventFilter(source, event)

    def handle_undo_redo(self, event):
        undo = False
        if event.modifiers() & QtCore.Qt.ShiftModifier:
            action = self.undo_redo_tracker.redo()
            undo = False
        else:
            action = self.undo_redo_tracker.undo()
            undo = True
        if action is not None:
            keypoint, action_type = action
            if undo:
                if action_type == ActionType.ADD:
                    self.keypoints.remove(keypoint)
                else:
                    self.keypoints.append(keypoint)
            else:
                if action_type == ActionType.ADD:
                    self.keypoints.append(keypoint)
                else:
                    self.keypoints.remove(keypoint)
            self.update_display_image()
            self.update_timepoint()
            return True
        else:
            logger.LOGGER().info("No more actions to undo/redo.")
            return False

    def add_or_remove_keypoint(self, position):
        scene_pos = self.graphics_view.mapToScene(position.toPoint())
        x, y = scene_pos.x(), scene_pos.y()

        # Check if the position is within any keypoint
        for keypoint in self.keypoints:
            kp_x, kp_y = keypoint.pt
            radius = keypoint.size / 2
            if (x - kp_x) ** 2 + (y - kp_y) ** 2 <= radius ** 2:
                self.undo_redo_tracker.push((keypoint, ActionType.REMOVE))
                self.keypoints.remove(keypoint)
                self.update_display_image()
                self.keypoints_changed.emit(len(self.keypoints))
                return

        # Check if the position is within any contour
        for contour in self.contours:
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                # Ensure the contour shapes match before removing
                for existing_contour in self.contours:
                    if np.array_equal(contour, existing_contour):
                        self.undo_redo_tracker.push((existing_contour, ActionType.REMOVE))
                        self.contours.remove(existing_contour)
                        self.update_display_image()
                        self.keypoints_changed.emit(len(self.keypoints))
                        return

        # If not within any keypoint or contour, add a new keypoint
        keypoint = cv2.KeyPoint(x, y, NEW_KEYPOINT_SIZE)
        self.undo_redo_tracker.push((keypoint, ActionType.ADD))
        self.keypoints.append(keypoint)
        self.update_display_image()
        self.keypoints_changed.emit(len(self.keypoints))
        self.update_timepoint()

        # If not within any keypoint or contour, add a new keypoint
        keypoint = cv2.KeyPoint(x, y, NEW_KEYPOINT_SIZE)
        self.undo_redo_tracker.push((keypoint, ActionType.ADD))
        self.keypoints.append(keypoint)
        self.update_display_image()
        self.keypoints_changed.emit(len(self.keypoints))
        self.update_timepoint()

    def fit_image_to_view(self):
        self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.KeepAspectRatio)
        self.current_scale_factor = self.graphics_view.transform().m11()

if __name__ == "__main__":
    app = QApplication([])
    blob_detector = BlobDetector(DEFAULT_IMAGE_PATH)
    blob_detector.show()
    app.exec()