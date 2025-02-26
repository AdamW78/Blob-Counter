import cv2
import numpy as np
from PySide6 import QtCore
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QVBoxLayout, QWidget, QHBoxLayout, QLabel, QSlider, QLineEdit
from PySide6.QtCore import Qt, QPointF, QEvent

from logger import LOGGER
from undo_redo_tracker import UndoRedoTracker, ActionType

# Constants
DEFAULT_DISPLAY_SCALE_FACTOR = 0.5
DEFAULT_MIN_AREA = 144
DEFAULT_MAX_AREA = 5000
DEFAULT_MIN_CIRCULARITY = 0.6
DEFAULT_MIN_CONVEXITY = 0.1
DEFAULT_MIN_INERTIA_RATIO = 0.01
DEFAULT_BLOB_COLOR = 0
DEFAULT_IMAGE_PATH = r'images/Arginine CLS with pH/Day 3/3_1_3rd_dilution.JPG'
CIRCLE_COLOR = (0, 0, 255)
CIRCLE_THICKNESS = 10
MIN_SCALE_FACTOR = 0.1
MAX_SCALE_FACTOR = 5.0
NEW_KEYPOINT_SIZE = 40
GRAPHICS_VIEW_HEIGHT = 600
GRAPHICS_VIEW_WIDTH = 800

class BlobDetector(QWidget):
    def __init__(self, image_path, display_scale_factor=DEFAULT_DISPLAY_SCALE_FACTOR):
        super().__init__()
        self.image_path = image_path
        self.display_scale_factor = display_scale_factor
        self.image = self.load_image()
        self.gray_image = self.convert_to_grayscale()
        self.params = self.create_blob_detector_params()
        self.detector = cv2.SimpleBlobDetector_create(self.params)
        self.keypoints = list(self.detect_blobs())
        self.blobs = self.draw_blobs()
        self.current_scale_factor = 1.0
        self.undo_redo_tracker = UndoRedoTracker()
        self.initUI()

        # Mouse state variables
        self.is_dragging = False
        self.mouse_is_pressed = False
        self.mouse_press_position = QPointF()

    def initUI(self):
        self.setWindowTitle("Blob Detector")
        self.layout = QVBoxLayout(self)

        # Add label to display the number of keypoints
        self.keypoint_count_label = QLabel(f'Keypoints: {len(self.keypoints)}')
        self.layout.addWidget(self.keypoint_count_label)

        # Add sliders and text inputs
        self.min_area_slider, self.min_area_input = self.create_slider_with_input('Min Area', 0, 5000, DEFAULT_MIN_AREA)
        self.max_area_slider, self.max_area_input = self.create_slider_with_input('Max Area', 0, 5000, DEFAULT_MAX_AREA)
        self.min_circularity_slider, self.min_circularity_input = self.create_slider_with_input('Min Circularity', 0, 100, int(DEFAULT_MIN_CIRCULARITY * 100))

        self.layout.addLayout(self.min_area_slider)
        self.layout.addLayout(self.max_area_slider)
        self.layout.addLayout(self.min_circularity_slider)

        self.graphics_view = QGraphicsView(self)
        self.graphics_view.setFixedSize(GRAPHICS_VIEW_WIDTH, GRAPHICS_VIEW_HEIGHT)
        self.graphics_scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.graphics_scene)
        self.layout.addWidget(self.graphics_view)

        self.pixmap_item = QGraphicsPixmapItem()
        self.graphics_scene.addItem(self.pixmap_item)

        self.update_display_image()

        self.graphics_view.setDragMode(QGraphicsView.NoDrag)
        self.graphics_view.viewport().installEventFilter(self)
        self.installEventFilter(self)

        # Fit the image to the view
        # self.fit_image_to_view()

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
        params.filterByConvexity = False
        params.minConvexity = DEFAULT_MIN_CONVEXITY
        params.filterByInertia = False
        params.minInertiaRatio = DEFAULT_MIN_INERTIA_RATIO
        params.filterByColor = False
        params.blobColor = DEFAULT_BLOB_COLOR
        return params

    def detect_blobs(self):
        return self.detector.detect(self.gray_image)

    def draw_blobs(self):
        image_with_keypoints = self.image.copy()
        for keypoint in self.keypoints:
            x, y = keypoint.pt
            radius = int(keypoint.size / 2)
            cv2.circle(image_with_keypoints, (int(x), int(y)), radius, CIRCLE_COLOR, CIRCLE_THICKNESS)
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
            return True
        else:
            LOGGER.info("No more actions to undo/redo.")
            return False


    def add_or_remove_keypoint(self, position):
        scene_pos = self.graphics_view.mapToScene(position.toPoint())
        x, y = scene_pos.x(), scene_pos.y()
        for keypoint in self.keypoints:
            kp_x, kp_y = keypoint.pt
            radius = keypoint.size / 2
            if (x - kp_x) ** 2 + (y - kp_y) ** 2 <= radius ** 2:
                self.undo_redo_tracker.push((keypoint, ActionType.REMOVE))
                self.keypoints.remove(keypoint)
                self.update_display_image()
                return
        keypoint = cv2.KeyPoint(x, y, NEW_KEYPOINT_SIZE)
        self.undo_redo_tracker.push((keypoint, ActionType.ADD))
        self.keypoints.append(keypoint)
        self.update_display_image()

    def fit_image_to_view(self):
        self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.KeepAspectRatio)
        self.current_scale_factor = self.graphics_view.transform().m11()

if __name__ == "__main__":
    app = QApplication([])
    blob_detector = BlobDetector(DEFAULT_IMAGE_PATH)
    blob_detector.show()
    app.exec()