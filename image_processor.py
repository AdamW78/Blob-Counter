import concurrent.futures
import os
import re

from PySide6.QtCore import Signal, QObject

from image_reader import BlobDetector


class ImageProcessor(QObject):
    image_processed = Signal(str, int, str)

    def __init__(self, directory, min_area, max_area, min_circularity):
        super().__init__()
        self.directory = directory
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity

    def process_images(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for filename in os.listdir(self.directory):
                if re.search(r'\.(jpg|jpeg|png|bmp)$', filename, re.IGNORECASE):
                    image_path = os.path.join(self.directory, filename)
                    futures.append(executor.submit(self.process_single_image, filename, image_path))
            for future in concurrent.futures.as_completed(futures):
                filename, blob_count, image_path = future.result()
                self.image_processed.emit(filename, blob_count, image_path)

    def process_single_image(self, filename, image_path):
        blob_detector = BlobDetector(image_path)
        blob_detector.update_params(self.min_area, self.max_area, self.min_circularity)
        blob_detector.update_display_image()
        blob_count = len(blob_detector.keypoints)
        return filename, blob_count, image_path