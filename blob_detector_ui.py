import cv2
from PySide6.QtCore import Qt, QEvent, Signal, QPointF
from PySide6.QtGui import QImage, QPixmap, QWheelEvent, QKeyEvent, QIcon
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QCheckBox, \
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGestureEvent, QGesture

from ui_utils import UIUtils
from utils import GRAPHICS_VIEW_WIDTH, GRAPHICS_VIEW_HEIGHT, MIN_SCALE_FACTOR, MAX_SCALE_FACTOR


class BlobDetectorUI(QWidget):
    keypoints_changed = Signal(int)

    def __init__(self, blob_detector_logic):
        super().__init__()
        self.fitted = False
        self.blob_detector_logic = blob_detector_logic
        self.current_scale_factor = 1.0
        self.is_dragging = False
        self.mouse_is_pressed = False
        self.mouse_press_position = QPointF()
        self.initUI()
        if not blob_detector_logic.image_path:
            self.disable_all_widgets()

        # Connect the keypoints_changed signal to the update_keypoint_count_label and update_display_image slots
        self.blob_detector_logic.keypoints_changed.connect(self.update_keypoint_count_label)
        self.blob_detector_logic.keypoints_changed.connect(self.update_display_image)

    def initUI(self):
        self.setWindowTitle("Blob Detector")
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(0)  # Set vertical spacing to zero

        self.keypoint_count_label = QLabel('Keypoints: 0')
        self.layout.addWidget(self.keypoint_count_label)

        sliders = UIUtils.create_blob_detector_sliders()
        self.min_area_group_box, self.min_area_slider, self.min_area_input = sliders['min_area']
        self.max_area_group_box, self.max_area_slider, self.max_area_input = sliders['max_area']
        self.min_circularity_group_box, self.min_circularity_slider, self.min_circularity_input = sliders[
            'min_circularity']
        self.min_convexity_group_box, self.min_convexity_slider, self.min_convexity_input = sliders['min_convexity']
        self.min_inertia_ratio_group_box, self.min_inertia_ratio_slider, self.min_inertia_ratio_input = sliders[
            'min_inertia_ratio']
        self.min_dist_between_blobs_group_box, self.min_dist_between_blobs_slider, self.min_dist_between_blobs_input = \
        sliders['min_dist_between_blobs']

        self.layout.addWidget(self.min_area_group_box)
        self.layout.addWidget(self.max_area_group_box)
        self.layout.addWidget(self.min_circularity_group_box)
        self.layout.addWidget(self.min_convexity_group_box)
        self.layout.addWidget(self.min_inertia_ratio_group_box)
        self.layout.addWidget(self.min_dist_between_blobs_group_box)

        self.gaussian_blur_checkbox = QCheckBox('Apply Gaussian Blur')
        self.morphological_operations_checkbox = QCheckBox('Apply Morphological Operations')
        self.layout.addWidget(self.gaussian_blur_checkbox)
        self.layout.addWidget(self.morphological_operations_checkbox)

        self.recount_button = QPushButton('Recount Blobs')
        self.recount_button.setIcon(QIcon('icons/recount.svg'))
        self.recount_button.clicked.connect(self.update_blob_count)
        self.layout.addWidget(self.recount_button)

        self.graphics_view = QGraphicsView(self)
        self.graphics_view.setFixedSize(GRAPHICS_VIEW_WIDTH, GRAPHICS_VIEW_HEIGHT)
        self.graphics_scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.graphics_scene)
        self.layout.addWidget(self.graphics_view)

        self.pixmap_item = QGraphicsPixmapItem()
        self.graphics_scene.addItem(self.pixmap_item)

        self.graphics_view.setDragMode(QGraphicsView.NoDrag)
        self.graphics_view.viewport().installEventFilter(self)
        self.installEventFilter(self)
        self.graphics_view.viewport().grabGesture(Qt.GestureType.SwipeGesture)
        self.graphics_view.viewport().grabGesture(Qt.GestureType.PinchGesture)

    def fit_in_view(self):
        rect = self.graphics_scene.itemsBoundingRect()
        self.graphics_view.setSceneRect(rect)
        self.graphics_view.resetTransform()
        scale_x = GRAPHICS_VIEW_WIDTH * 1.5 / rect.width()
        scale_y = GRAPHICS_VIEW_HEIGHT * 1.5 / rect.height()
        scale = min(scale_x, scale_y)
        self.graphics_view.scale(scale, scale)
        # Center the view
        self.graphics_view.centerOn(rect.center())

    def disable_all_widgets(self):
        self.keypoint_count_label.setEnabled(False)
        self.min_area_slider.setEnabled(False)
        self.min_area_input.setEnabled(False)
        self.max_area_slider.setEnabled(False)
        self.max_area_input.setEnabled(False)
        self.min_circularity_slider.setEnabled(False)
        self.min_circularity_input.setEnabled(False)
        self.min_convexity_slider.setEnabled(False)
        self.min_convexity_input.setEnabled(False)
        self.min_inertia_ratio_slider.setEnabled(False)
        self.min_inertia_ratio_input.setEnabled(False)
        self.min_dist_between_blobs_slider.setEnabled(False)
        self.min_dist_between_blobs_input.setEnabled(False)
        self.gaussian_blur_checkbox.setEnabled(False)
        self.morphological_operations_checkbox.setEnabled(False)
        self.recount_button.setEnabled(False)
        self.graphics_view.setEnabled(False)

    def update_blob_count(self):
        min_area = int(self.min_area_input.text())
        max_area = int(self.max_area_input.text())
        min_circularity = float(self.min_circularity_input.text()) / 100.0
        min_convexity = float(self.min_convexity_input.text()) / 100.0
        min_inertia_ratio = float(self.min_inertia_ratio_input.text()) / 100.0
        min_dist_between_blobs = int(self.min_dist_between_blobs_input.text())
        apply_gaussian_blur = self.gaussian_blur_checkbox.isChecked()
        apply_morphological_operations = self.morphological_operations_checkbox.isChecked()

        self.blob_detector_logic.update_blob_count(
            min_area, max_area, min_circularity, min_convexity, min_inertia_ratio, min_dist_between_blobs,
            apply_gaussian_blur, apply_morphological_operations
        )
        self.update_display_image()
        QApplication.processEvents()  # Process the event loop to update the UI

    def update_display_image(self):
        image = self.blob_detector_logic.get_display_image()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.pixmap_item.setPixmap(pixmap)
        self.graphics_scene.setSceneRect(0, 0, width, height)
        if not self.fitted:
            self.fit_in_view()
            self.fitted = True


    def update_keypoint_count_label(self, count):
        self.keypoint_count_label.setText(f'Keypoints: {count}')

    def eventFilter(self, source, event):
        if self.blob_detector_logic.image_path is None:
            return super().eventFilter(source, event)
        if event.type() == QEvent.Gesture:
            return self.handle_gesture_event(event)
        elif event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                self.mouse_press_position = event.position()
                if self.graphics_view.viewport().rect().contains(self.graphics_view.mapFromGlobal(event.globalPosition().toPoint())):
                    self.mouse_is_pressed = True
                    self.is_dragging = False
                    return True
        elif event.type() == QEvent.Type.MouseMove:
            if self.mouse_is_pressed and self.graphics_view.viewport().rect().contains(self.graphics_view.mapFromGlobal(event.globalPosition().toPoint())):
                self.is_dragging = True
                delta = event.position() - self.mouse_press_position
                self.graphics_view.horizontalScrollBar().setValue(self.graphics_view.horizontalScrollBar().value() - delta.x())
                self.graphics_view.verticalScrollBar().setValue(self.graphics_view.verticalScrollBar().value() - delta.y())
                self.mouse_press_position = event.position()
                return True
        elif event.type() == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.LeftButton:
                if not self.is_dragging and self.graphics_view.viewport().rect().contains(self.graphics_view.mapFromGlobal(event.globalPosition().toPoint())):
                    self.add_or_remove_keypoint(event.position())
                    self.is_dragging = False
                    self.mouse_is_pressed = False
                    return True
        elif event.type() == QEvent.Type.Wheel:
            self.handle_wheel_zoom(event)
            return True
        elif event.type() == QEvent.Type.KeyPress:
            self.handle_key_zoom(event)
            return True
        return super().eventFilter(source, event)

    def handle_wheel_zoom(self, event: QWheelEvent):
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

    def handle_key_zoom(self, event: QKeyEvent):
        zoom_in_factor = 1.1
        zoom_out_factor = 1 / zoom_in_factor

        if event.modifiers() & Qt.KeyboardModifier.ControlModifier or event.modifiers() & Qt.KeyboardModifier.MetaModifier:
            if event.key() == Qt.Key.Key_Equal:
                zoom_factor = zoom_in_factor
            elif event.key() == Qt.Key.Key_Minus:
                zoom_factor = zoom_out_factor
            else:
                return

            new_scale_factor = self.current_scale_factor * zoom_factor
            if MIN_SCALE_FACTOR <= new_scale_factor <= MAX_SCALE_FACTOR:
                self.graphics_view.scale(zoom_factor, zoom_factor)
                self.current_scale_factor = new_scale_factor

    def handle_pinch_gesture(self, gesture: QGesture):
        if gesture is None or gesture.gestureType() is not Qt.GestureType.PinchGesture:
            return False
        pinch_gesture = gesture
        scale_factor = pinch_gesture.scaleFactor()
        new_scale_factor = self.current_scale_factor * scale_factor
        if MIN_SCALE_FACTOR <= new_scale_factor <= MAX_SCALE_FACTOR:
            self.graphics_view.scale(scale_factor, scale_factor)
            self.current_scale_factor = new_scale_factor
        return True

    def handle_swipe_gesture(self, gesture: QGesture):
        if gesture is None or gesture.gestureType() is not Qt.GestureType.SwipeGesture:
            return False
        swipe_gesture = gesture
        delta = swipe_gesture.delta()
        self.graphics_view.horizontalScrollBar().setValue(self.graphics_view.horizontalScrollBar().value() - delta.x())
        self.graphics_view.verticalScrollBar().setValue(self.graphics_view.verticalScrollBar().value() - delta.y())
        return True

    def handle_gesture_event(self, event: QGestureEvent):
        for gesture in event.gestures():
            if gesture.gestureType() is Qt.GestureType.PinchGesture:
                return self.handle_pinch_gesture(gesture)
            elif gesture.gestureType() is Qt.GestureType.SwipeGesture:
                return self.handle_swipe_gesture(gesture)
            else:
                return False

    def add_or_remove_keypoint(self, position):
        scene_pos = self.graphics_view.mapToScene(position.toPoint())
        self.blob_detector_logic.add_or_remove_keypoint(scene_pos.x(), scene_pos.y())
        self.keypoints_changed.emit(self.blob_detector_logic.get_keypoint_count())
        self.update_display_image()