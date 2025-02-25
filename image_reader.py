import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap, QIntValidator
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QHBoxLayout, QLineEdit
from PySide6.QtCore import Qt

# Constants
DEFAULT_DISPLAY_SCALE_FACTOR = 0.5
DEFAULT_MIN_AREA = 144
DEFAULT_MAX_AREA = 5000
DEFAULT_MIN_CIRCULARITY = 0.6
DEFAULT_MIN_CONVEXITY = 0.1
DEFAULT_MIN_INERTIA_RATIO = 0.01
DEFAULT_BLOB_COLOR = 0
DEFAULT_IMAGE_PATH = r'images/Arginine CLS with pH/Day 3/3_1_3rd_dilution.JPG'
DOCK_SIZE = 400
TEXT_POSITION = (20, 200)
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 4
TEXT_COLOR = (0, 100, 255)
TEXT_THICKNESS = 10
CIRCLE_COLOR = (0, 0, 255)
CIRCLE_THICKNESS = 10
KEYPOINT_SIZE = 20
LINE_EDIT_WIDTH = 50
ZOOM_STEP = 0.1
MIN_SCALE_FACTOR = 0.1
MAX_SCALE_FACTOR = 5.0
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600

class BlobDetector:
    def __init__(self, image_path, display_scale_factor=DEFAULT_DISPLAY_SCALE_FACTOR):
        self.image_path = image_path
        self.display_scale_factor = display_scale_factor
        self.image = self.load_image()
        self.gray_image = self.convert_to_grayscale()
        self.params = self.create_blob_detector_params()
        self.detector = cv2.SimpleBlobDetector_create(self.params)
        self.keypoints = list(self.detect_blobs())
        self.blobs = self.draw_blobs()
        self.display_image = None
        self.zoom_center = (self.image.shape[1] // 2, self.image.shape[0] // 2)
        self.dragging = False
        self.last_mouse_position = None
        self.mouse_moved = False

    def load_image(self):
        image = cv2.imread(self.image_path, 1)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {self.image_path}")
        return image

    def update_display_image(self):
        self.blobs = self.draw_blobs()
        self.display_image = self.resize_image_for_display()

    def resize_image_for_display(self):
        image_height, image_width, _ = self.image.shape
        aspect_ratio = image_width / image_height

        if CANVAS_WIDTH / aspect_ratio <= CANVAS_HEIGHT:
            new_width = CANVAS_WIDTH
            new_height = int(CANVAS_WIDTH / aspect_ratio)
        else:
            new_height = CANVAS_HEIGHT
            new_width = int(CANVAS_HEIGHT * aspect_ratio)

        zoom_width = int(image_width / self.display_scale_factor)
        zoom_height = int(image_height / self.display_scale_factor)
        x_center, y_center = self.zoom_center

        x_start = max(0, int(x_center - zoom_width // 2))
        y_start = max(0, int(y_center - zoom_height // 2))
        x_end = min(image_width, int(x_center + zoom_width // 2))
        y_end = min(image_height, int(y_center + zoom_height // 2))

        cropped_image = self.blobs[y_start:y_end, x_start:x_end]
        resized_image = cv2.resize(cropped_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        return resized_image

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

    def resize_image_to_fit(self, width, height):
        aspect_ratio = self.image.shape[1] / self.image.shape[0]
        if width / aspect_ratio <= height:
            new_width = width
            new_height = int(width / aspect_ratio)
        else:
            new_height = height
            new_width = int(height * aspect_ratio)
        resized_image = cv2.resize(self.blobs, (new_width, new_height))
        return resized_image

    def detect_blobs(self):
        return self.detector.detect(self.gray_image)

    def draw_blobs(self):
        image_with_keypoints = self.image.copy()
        for keypoint in self.keypoints:
            x, y = keypoint.pt
            radius = int(keypoint.size / 2)
            cv2.circle(image_with_keypoints, (int(x), int(y)), radius, CIRCLE_COLOR, CIRCLE_THICKNESS)
        return image_with_keypoints

    def update_params(self, min_area, max_area, min_circularity):
        self.params.minArea = min_area
        self.params.maxArea = max_area
        self.params.minCircularity = min_circularity
        self.detector = cv2.SimpleBlobDetector_create(self.params)
        self.keypoints = list(self.detect_blobs())
        self.update_display_image()

    def get_display_image_pixmap(self):
        if self.display_image is None:
            self.update_display_image()
        image = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_image)

    # Other methods remain unchanged
    def run_with_gui(self, app=None):
        appWasNone = False
        if app is None:
            app = QApplication([])
            appWasNone = True

        screen = app.primaryScreen()
        screen_geometry = screen.availableGeometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height() - DOCK_SIZE

        image_height, image_width, _ = self.image.shape
        scale_width = screen_width / image_width
        scale_height = screen_height / image_height
        self.display_scale_factor = min(scale_width, scale_height, self.display_scale_factor)
        self.display_image = self.resize_image_for_display()

        window = QWidget()
        window.setWindowTitle("Blob Detector Parameters")
        layout = QVBoxLayout()

        min_area_layout, min_area_slider = self.create_slider_with_labels("Min Area", 0, 5000, self.params.minArea)
        max_area_layout, max_area_slider = self.create_slider_with_labels("Max Area", 0, 5000, self.params.maxArea)
        min_circularity_layout, min_circularity_slider = self.create_slider_with_labels("Min Circularity", 0, 100,
                                                                                        int(self.params.minCircularity * 100))

        layout.addLayout(min_area_layout)
        layout.addLayout(max_area_layout)
        layout.addLayout(min_circularity_layout)

        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        self.blob_count_label = QLabel()
        layout.addWidget(self.blob_count_label)

        def update_image():
            self.update_params(min_area_slider.value(), max_area_slider.value(), min_circularity_slider.value() / 100.0)
            self.update_image_label()
            self.blob_count_label.setText(f"Number of Circular Blobs: {len(self.keypoints)}")

        min_area_slider.valueChanged.connect(update_image)
        max_area_slider.valueChanged.connect(update_image)
        min_circularity_slider.valueChanged.connect(update_image)

        update_button = QPushButton("Update Parameters")
        update_button.clicked.connect(update_image)
        layout.addWidget(update_button)

        window.setLayout(layout)
        window.show()
        update_image()

        self.image_label.mousePressEvent = self.mousePressEvent
        self.image_label.mouseMoveEvent = self.mouseMoveEvent
        self.image_label.mouseReleaseEvent = self.mouseReleaseEvent
        window.wheelEvent = self.wheelEvent

        if appWasNone:
            app.exec()

if __name__ == "__main__":
    BLOB_DETECTOR = BlobDetector(DEFAULT_IMAGE_PATH)
    BLOB_DETECTOR.run_with_gui()