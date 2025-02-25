import re

from PySide6.QtCore import Qt, QThread, QEvent
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QListWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QWidget, QSlider, QLineEdit, \
    QFileDialog, QListWidgetItem, QApplication

from image_processor import ImageProcessor
from image_reader import BlobDetector, DEFAULT_MIN_AREA, DEFAULT_MAX_AREA, DEFAULT_MIN_CIRCULARITY, MIN_SCALE_FACTOR, \
    MAX_SCALE_FACTOR


class ImageSetReader(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image_items = []
        self.scale_factor = 1.0
        self.dragging = False
        self.last_mouse_position = None
        self.thread = None

    def initUI(self):
        self.setWindowTitle('Image Set Reader')
        main_layout = QHBoxLayout()

        controls_layout = QVBoxLayout()

        self.label = QLabel('Select a directory containing images:')
        controls_layout.addWidget(self.label)

        self.button = QPushButton('Select Directory')
        self.button.clicked.connect(self.select_directory)
        controls_layout.addWidget(self.button)

        self.min_area_slider, self.min_area_input = self.create_slider_with_input('Min Area', 0, 5000, DEFAULT_MIN_AREA)
        self.max_area_slider, self.max_area_input = self.create_slider_with_input('Max Area', 0, 5000, DEFAULT_MAX_AREA)
        self.min_circularity_slider, self.min_circularity_input = self.create_slider_with_input('Min Circularity', 0, 100, int(DEFAULT_MIN_CIRCULARITY * 100))

        controls_layout.addLayout(self.min_area_slider)
        controls_layout.addLayout(self.max_area_slider)
        controls_layout.addLayout(self.min_circularity_slider)

        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.on_item_clicked)
        controls_layout.addWidget(self.image_list)

        main_layout.addLayout(controls_layout)

        self.image_label = QLabel()
        self.image_label.setFixedSize(800, 600)
        self.image_label.setMouseTracking(True)
        self.image_label.installEventFilter(self)
        main_layout.addWidget(self.image_label)

        self.setLayout(main_layout)

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

    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, 'Select Directory')
        if directory:
            self.process_images(directory)

    def process_images(self, directory):
        self.thread = QThread()
        self.worker = ImageProcessor(directory, int(self.min_area_input.text()), int(self.max_area_input.text()), int(self.min_circularity_input.text()) / 100.0)
        self.worker.moveToThread(self.thread)
        self.worker.image_processed.connect(self.add_image_item)
        self.thread.started.connect(self.worker.process_images)
        self.thread.start()

    def add_image_item(self, filename, blob_count, image_path):
        self.image_items.append((blob_count, filename, image_path))

        def extract_number(item):
            match = re.search(r'_(\d+)', item[1])
            return int(match.group(1)) if match else 0

        self.image_items.sort(key=extract_number, reverse=True)

        self.image_list.clear()
        for blob_count, filename, image_path in self.image_items:
            item = QListWidgetItem(f'{filename}: {blob_count} blobs')
            item.setData(Qt.UserRole, (blob_count, image_path))
            self.image_list.addItem(item)

        if self.image_items:
            self.image_list.setCurrentItem(self.image_list.item(0))
            self.show_image_with_blobs(self.image_items[0][2])

    def on_item_clicked(self, item):
        image_path = item.data(Qt.UserRole)[1]
        self.show_image_with_blobs(image_path)

    def show_image_with_blobs(self, image_path):
        self.blob_detector = BlobDetector(image_path)
        self.blob_detector.update_params(
            int(self.min_area_input.text()),
            int(self.max_area_input.text()),
            int(self.min_circularity_input.text()) / 100.0
        )
        self.update_image_label()

    def update_image_label(self):
        self.blob_detector.update_display_image()
        pixmap = self.blob_detector.get_display_image_pixmap()
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)
        self.image_label.adjustSize()

    def eventFilter(self, source, event):
        if source == self.image_label:
            if event.type() == QEvent.Wheel:
                self.handle_wheel_event(event)
                return True
            elif event.type() == QEvent.MouseButtonPress:
                self.handle_mouse_press_event(event)
                return True
            elif event.type() == QEvent.MouseMove:
                self.handle_mouse_move_event(event)
                return True
            elif event.type() == QEvent.MouseButtonRelease:
                self.handle_mouse_release_event(event)
                return True
        return super().eventFilter(source, event)

    def handle_wheel_event(self, event):
        if isinstance(event, QWheelEvent):
            if event.angleDelta().y() > 0:
                self.blob_detector.display_scale_factor = min(self.blob_detector.display_scale_factor * 1.1, MAX_SCALE_FACTOR)
            else:
                self.blob_detector.display_scale_factor = max(self.blob_detector.display_scale_factor / 1.1, MIN_SCALE_FACTOR)
            self.update_image_label()

    def handle_mouse_press_event(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.last_mouse_position = event.position()

    def handle_mouse_move_event(self, event):
        if self.dragging:
            delta = event.position() - self.last_mouse_position
            self.last_mouse_position = event.position()
            self.blob_detector.zoom_center = (
                self.blob_detector.zoom_center[0] - delta.x(),
                self.blob_detector.zoom_center[1] - delta.y()
            )
            self.update_image_label()

    def handle_mouse_release_event(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.update_blob_count()

    def update_blob_count(self):
        self.blob_detector.update_params(
            int(self.min_area_input.text()),
            int(self.max_area_input.text()),
            int(self.min_circularity_input.text()) / 100.0
        )
        self.update_image_label()
        selected_item = self.image_list.currentItem()
        if selected_item:
            blob_count = len(self.blob_detector.keypoints)
            filename = selected_item.text().split(':')[0]
            selected_item.setText(f'{filename}: {blob_count} blobs')

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        event.accept()
if __name__ == '__main__':
    app = QApplication([])
    reader = ImageSetReader()
    reader.show()
    app.exec()