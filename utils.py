from collections import namedtuple

# Constants
DEFAULT_DISPLAY_SCALE_FACTOR = 0.5
MIN_DISTANCE_BETWEEN_BLOBS = 1
MAX_DISTANCE_BETWEEN_BLOBS = 100
MIN_THRESHOLD = 1
MAX_THRESHOLD = 255
MIN_AREA = 1
MAX_AREA = 8000
DEFAULT_MIN_THRESHOLD = 100
DEFAULT_MAX_THRESHOLD = 160
DEFAULT_MIN_DISTANCE_BETWEEN_BLOBS = 1
DEFAULT_MIN_AREA = 144
DEFAULT_MAX_AREA = 5000
DEFAULT_MIN_CIRCULARITY = 0.4
DEFAULT_MIN_CONVEXITY = 0.80
DEFAULT_MIN_INERTIA_RATIO = 0.01
DEFAULT_BLOB_COLOR = 1
DEFAULT_IMAGE_PATH = r'images/Arginine CLS with pH/Day 19/19_23_2nd_dilution.JPG'
CIRCLE_COLOR = (0, 0, 255)
CIRCLE_THICKNESS = 10
MIN_SCALE_FACTOR = 0.1
MAX_SCALE_FACTOR = 5.0
NEW_KEYPOINT_SIZE = 40
GRAPHICS_VIEW_HEIGHT = 600
GRAPHICS_VIEW_WIDTH = 800
DEFAULT_DILUTION = "3rd"
USE_DILUTION = True
USE_DAY = True
IMAGE_LIST_WIDGET_WIDTH = 300

# Tooltips
TOOLTIP_MIN_AREA = "Minimum area of blobs to detect"
TOOLTIP_MAX_AREA = "Maximum area of blobs to detect"
TOOLTIP_MIN_CIRCULARITY = "Minimum circularity of blobs to detect"
TOOLTIP_MIN_CONVEXITY = "Minimum convexity of blobs to detect"
TOOLTIP_MIN_INERTIA_RATIO = "Minimum inertia ratio of blobs to detect"
TOOLTIP_MIN_DIST_BETWEEN_BLOBS = "Minimum distance between blobs"
TOOLTIP_MIN_THRESHOLD = "Minimum threshold for blob detection"
TOOLTIP_MAX_THRESHOLD = "Maximum threshold for blob detection"

Timepoint = namedtuple("Timepoint", ["day", "sample_number", "dilution", "num_keypoints"])