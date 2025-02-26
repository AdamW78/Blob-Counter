import os
from PySide6.QtWidgets import QListWidget, QFileDialog, QScrollArea, QWidget, QHBoxLayout, QPushButton, QApplication

from image_reader import BlobDetector


class ImageSetBlobDetector(QWidget):
    def __init__(self):
        super().__init__()
        self.image_paths = []
        self.blob_detectors = []
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Set Blob Detector")
        self.layout = QHBoxLayout(self)

        # Add scroll area for image list
        self.scroll_area = QScrollArea(self)
        self.image_list_widget = QListWidget(self)
        self.image_list_widget.itemClicked.connect(self.display_selected_image)
        self.scroll_area.setWidget(self.image_list_widget)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedWidth(200)
        self.layout.addWidget(self.scroll_area)

        # Add blob detector UI
        self.blob_detector_widget = BlobDetector()
        self.layout.addWidget(self.blob_detector_widget)

        # Add button to open folder
        self.open_folder_button = QPushButton('Open Folder')
        self.open_folder_button.clicked.connect(self.open_folder)
        self.layout.addWidget(self.open_folder_button)

    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.load_images_from_folder(folder_path)

    def load_images_from_folder(self, folder_path):
        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_list_widget.clear()
        for image_path in self.image_paths:
            self.image_list_widget.addItem(os.path.basename(image_path))
        self.blob_detectors = [BlobDetector(image_path) for image_path in self.image_paths]

    def display_selected_image(self, item):
        selected_image_path = os.path.join(os.path.dirname(self.image_paths[0]), item.text())
        for detector in self.blob_detectors:
            if detector.image_path == selected_image_path:
                self.layout.removeWidget(self.blob_detector_widget)
                self.blob_detector_widget.deleteLater()
                self.blob_detector_widget = detector
                self.layout.addWidget(self.blob_detector_widget)
                break

    def update_blob_count_for_all_images(self):
        for detector in self.blob_detectors:
            detector.update_blob_count()

if __name__ == "__main__":
    app = QApplication([])
    image_set_blob_detector = ImageSetBlobDetector()
    image_set_blob_detector.show()
    app.exec()