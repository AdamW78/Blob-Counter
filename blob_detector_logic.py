import os

import cv2
import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import Signal, QObject

import logger
from undo_redo_tracker import ActionType, UndoRedoTracker
from utils import DEFAULT_MIN_AREA, DEFAULT_MIN_CIRCULARITY, DEFAULT_MAX_AREA, DEFAULT_MIN_CONVEXITY, \
    DEFAULT_MIN_INERTIA_RATIO, DEFAULT_BLOB_COLOR, MIN_DISTANCE_BETWEEN_BLOBS, DEFAULT_MIN_THRESHOLD, \
    DEFAULT_MAX_THRESHOLD, CIRCLE_COLOR, CIRCLE_THICKNESS, DEFAULT_DILUTION, USE_DAY, Timepoint, USE_DILUTION, \
    NEW_KEYPOINT_SIZE


class BlobDetectorLogic(QObject):
    keypoints_changed = Signal(int)

    def __init__(self, image_path=None):
        super().__init__()
        self.undo_redo_tracker = UndoRedoTracker()
        self.timepoint = None
        self.image_path = image_path
        self.image = None
        self.gray_image = None
        self.keypoints = []
        self.contours = []
        self.params = self.create_blob_detector_params()
        self.detector = cv2.SimpleBlobDetector_create(self.params)
        self.custom_name = None
        self.day_num = -1
        self.sample_number = -1
        self.dilution = None
        if image_path is not None:
            self.load_image()
            self.convert_to_grayscale()
            self.detect_blobs()
            self.custom_name = self.get_custom_name()
            self.update_timepoint()

    def load_image(self):
        self.image = cv2.imread(self.image_path, 1)
        if self.image is None:
            raise FileNotFoundError(f"Image not found at path: {self.image_path}")

    def update_timepoint(self):
        if self.day_num != -1 and self.sample_number != -1 and self.dilution is not None:
            self.timepoint = Timepoint(day=self.day_num, sample_number=self.sample_number, dilution=self.dilution,
                                       num_keypoints=len(self.keypoints))
    def get_timepoint(self):
        return self.timepoint

    def convert_to_grayscale(self):
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

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

    def detect_blobs(self):
        self.keypoints = list(self.detector.detect(self.gray_image))

    def update_blob_count(self, min_area, max_area, min_circularity, min_convexity, min_inertia_ratio,
                          min_dist_between_blobs, apply_gaussian_blur, apply_morphological_operations):
        self.params.minArea = min_area
        self.params.maxArea = max_area
        self.params.minCircularity = min_circularity
        self.params.minConvexity = min_convexity
        self.params.minInertiaRatio = min_inertia_ratio
        self.params.minDistBetweenBlobs = min_dist_between_blobs
        self.detector = cv2.SimpleBlobDetector_create(self.params)
        self.convert_to_grayscale()
        if apply_gaussian_blur:
            self.gray_image = cv2.GaussianBlur(self.gray_image, (5, 5), 0)
        if apply_morphological_operations:
            kernel = np.ones((5, 5), np.uint8)
            self.gray_image = cv2.erode(self.gray_image, kernel, iterations=1)
            self.gray_image = cv2.dilate(self.gray_image, kernel, iterations=1)
        self.update_timepoint()
        self.keypoints_changed.emit(len(self.keypoints))

        # Detect blobs and update the display
        self.keypoints_changed.emit(len(self.keypoints))

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
                logger.LOGGER().warning(
                    "Day source is incorrect type - expected list or string - not including day in timepoint display name...")
                logger.LOGGER().warning("Day source type: %s", type(day_source))
                return day_string
            if self.check_if_int(day_string):
                self.day_num = int(day_string)
                return f"Day {day_string} - "
            else:
                logger.LOGGER().warning(
                    "Unable to parse day string - NOT AN INTEGER - not including day in timepoint display name...")
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

    def get_display_image(self):
        image_with_keypoints = self.image.copy()
        for keypoint in self.keypoints:
            x, y = keypoint.pt
            radius = int(keypoint.size / 2)
            cv2.circle(image_with_keypoints, (int(x), int(y)), radius, CIRCLE_COLOR, CIRCLE_THICKNESS)
        return image_with_keypoints

    def get_keypoint_count(self):
        return len(self.keypoints)

    def add_or_remove_keypoint(self, x, y):
        for keypoint in self.keypoints:
            kp_x, kp_y = keypoint.pt
            radius = keypoint.size / 2
            if (x - kp_x) ** 2 + (y - kp_y) ** 2 <= radius ** 2:
                self.keypoints.remove(keypoint)
                self.undo_redo_tracker.push((keypoint, ActionType.REMOVE))
                self.update_timepoint()
                self.keypoints_changed.emit(len(self.keypoints))
                return
        keypoint = cv2.KeyPoint(x, y, NEW_KEYPOINT_SIZE)
        self.keypoints.append(keypoint)
        self.undo_redo_tracker.push((keypoint, ActionType.ADD))
        self.update_timepoint()
        self.keypoints_changed.emit(len(self.keypoints))

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
            self.update_timepoint()
            return True
        else:
            logger.LOGGER().info("No more actions to undo/redo.")
            return False