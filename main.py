from PySide6.QtWidgets import QApplication
from blob_detector_ui import BlobDetectorUI
from blob_detector_logic import BlobDetectorLogic
from utils import DEFAULT_IMAGE_PATH

if __name__ == "__main__":
    app = QApplication([])
    blob_detector_logic = BlobDetectorLogic(DEFAULT_IMAGE_PATH)
    blob_detector_ui = BlobDetectorUI(blob_detector_logic)
    blob_detector_ui.update_display_image()  # Ensure the image is loaded and displayed
    blob_detector_ui.show()
    app.exec()