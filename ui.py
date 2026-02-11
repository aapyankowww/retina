from __future__ import annotations

import copy
import multiprocessing as mp
import os
import queue as queue_module
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QLineF, QObject, QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QShortcut,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QInputDialog,
)

import backend


class ToolMode(str, Enum):
    NONE = "none"
    RECTANGLE = "rectangle"
    POLYGON = "polygon"
    LINE = "line"


PREPROCESS_MODE_LABELS_RU = {
    "rgb": "Исходные цвета (RGB)",
}
PREPROCESS_MODE_KEYS_BY_LABEL_RU = {v: k for k, v in PREPROCESS_MODE_LABELS_RU.items()}

DETECTOR_BACKEND_LABELS_RU = {
    "cellpose_nuclei": "Cellpose Nuclei (рекомендуется для CPU)",
    "stardist": "StarDist (классический вариант)",
}
DETECTOR_BACKEND_KEYS_BY_LABEL_RU = {v: k for k, v in DETECTOR_BACKEND_LABELS_RU.items()}


def preprocess_mode_to_russian(mode_key: str) -> str:
    return PREPROCESS_MODE_LABELS_RU.get(mode_key, mode_key)


BATCH_MODE_FULL = "full_frame"
BATCH_MODE_RECT = "rectangles"
BATCH_MODE_POLY = "polygons"

BATCH_MODE_LABELS_RU = {
    BATCH_MODE_FULL: "По всему кадру",
    BATCH_MODE_RECT: "По прямоугольникам",
    BATCH_MODE_POLY: "По полигонам",
}


def enhancement_to_slider_values(params: dict) -> tuple[int, int, int, int]:
    saturation = int(round(float(params.get("saturation", 1.0)) * 100.0))
    brightness = int(round(float(params.get("brightness", 0.0))))
    contrast = int(round(float(params.get("contrast", 1.0)) * 100.0))
    sharpness = int(round(float(params.get("sharpness", 1.0)) * 100.0))
    return saturation, brightness, contrast, sharpness


def slider_values_to_enhancement(
    saturation_value: int,
    brightness_value: int,
    contrast_value: int,
    sharpness_value: int,
) -> dict:
    return backend.normalize_enhancement_params(
        {
            "saturation": float(saturation_value) / 100.0,
            "brightness": float(brightness_value),
            "contrast": float(contrast_value) / 100.0,
            "sharpness": float(sharpness_value) / 100.0,
        }
    )


class DetectionWorker(QObject):
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(
        self,
        image_path: str,
        enhancement_params: dict | None = None,
        detection_params: dict | None = None,
        custom_model_path: str | None = None,
    ) -> None:
        super().__init__()
        self.image_path = image_path
        self.enhancement_params = backend.normalize_enhancement_params(enhancement_params)
        self.detection_params = dict(detection_params or backend.get_detection_params())
        self.custom_model_path = str(custom_model_path).strip() if custom_model_path else ""

    @pyqtSlot()
    def run(self) -> None:
        proc = None
        event_queue = None
        try:
            ctx = mp.get_context("spawn")
            event_queue = ctx.Queue()
            payload = {
                "mode": "single",
                "image_path": self.image_path,
                "enhancement_params": self.enhancement_params,
                "detection_params": self.detection_params,
                "custom_model_path": self.custom_model_path,
            }
            proc = ctx.Process(target=backend.run_detection_job, args=(payload, event_queue))
            proc.start()

            result_nuclei: list[dict] | None = None
            error_message: str | None = None

            while True:
                try:
                    message = event_queue.get(timeout=0.2)
                except queue_module.Empty:
                    message = None

                if isinstance(message, dict):
                    mtype = message.get("type")
                    if mtype == "result":
                        result_nuclei = message.get("nuclei", [])
                        break
                    if mtype == "error":
                        error_message = str(message.get("message", "Ошибка детекции в дочернем процессе"))
                        break

                if proc is not None and not proc.is_alive():
                    break

            while event_queue is not None:
                try:
                    message = event_queue.get_nowait()
                except queue_module.Empty:
                    break
                if isinstance(message, dict):
                    mtype = message.get("type")
                    if mtype == "result" and result_nuclei is None:
                        result_nuclei = message.get("nuclei", [])
                    elif mtype == "error" and error_message is None:
                        error_message = str(message.get("message", "Ошибка детекции в дочернем процессе"))

            if proc is not None:
                proc.join(timeout=1.0)

            if result_nuclei is not None:
                self.finished.emit(result_nuclei)
                return

            if error_message is not None:
                self.failed.emit(error_message)
                return

            exit_code = proc.exitcode if proc is not None else None
            if exit_code is None:
                self.failed.emit("Дочерний процесс детекции завершился без кода выхода")
            elif exit_code != 0:
                self.failed.emit(
                    "Дочерний процесс детекции аварийно завершился "
                    f"(код {exit_code}). Проверьте совместимость библиотек TensorFlow/LLVM."
                )
            else:
                self.failed.emit("Дочерний процесс детекции завершился без результата")
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if proc is not None and proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)


class BatchDetectionWorker(QObject):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)
    canceled = pyqtSignal()

    def __init__(
        self,
        image_paths: list[str],
        rois_by_file: dict[str, list[dict]],
        pixels_per_mm: float,
        enhancement_params: dict | None = None,
        detection_params: dict | None = None,
        custom_model_path: str | None = None,
    ) -> None:
        super().__init__()
        self.image_paths = image_paths
        self.rois_by_file = rois_by_file
        self.pixels_per_mm = float(pixels_per_mm)
        self.enhancement_params = backend.normalize_enhancement_params(enhancement_params)
        self.detection_params = dict(detection_params or backend.get_detection_params())
        self.custom_model_path = str(custom_model_path).strip() if custom_model_path else ""
        self._cancel_requested = False

    @pyqtSlot()
    def request_cancel(self) -> None:
        self._cancel_requested = True

    @pyqtSlot()
    def run(self) -> None:
        proc = None
        event_queue = None
        try:
            ctx = mp.get_context("spawn")
            event_queue = ctx.Queue()
            payload = {
                "mode": "batch",
                "image_paths": self.image_paths,
                "rois_by_file": self.rois_by_file,
                "pixels_per_mm": self.pixels_per_mm,
                "enhancement_params": self.enhancement_params,
                "detection_params": self.detection_params,
                "custom_model_path": self.custom_model_path,
            }
            proc = ctx.Process(target=backend.run_detection_job, args=(payload, event_queue))
            proc.start()

            batch_rows: list[dict] | None = None
            error_message: str | None = None

            while True:
                if self._cancel_requested:
                    if proc is not None and proc.is_alive():
                        proc.terminate()
                        proc.join(timeout=1.0)
                    self.canceled.emit()
                    return

                try:
                    message = event_queue.get(timeout=0.2)
                except queue_module.Empty:
                    message = None

                if isinstance(message, dict):
                    mtype = message.get("type")
                    if mtype == "progress":
                        current = int(message.get("current", 0))
                        total = int(message.get("total", 0))
                        file_name = str(message.get("file_name", ""))
                        self.progress.emit(current, total, file_name)
                    elif mtype == "result":
                        batch_rows = message.get("rows", [])
                        break
                    elif mtype == "error":
                        error_message = str(
                            message.get("message", "Ошибка пакетной детекции в дочернем процессе")
                        )
                        break

                if proc is not None and not proc.is_alive():
                    break

            while event_queue is not None:
                try:
                    message = event_queue.get_nowait()
                except queue_module.Empty:
                    break
                if isinstance(message, dict):
                    mtype = message.get("type")
                    if mtype == "progress":
                        current = int(message.get("current", 0))
                        total = int(message.get("total", 0))
                        file_name = str(message.get("file_name", ""))
                        self.progress.emit(current, total, file_name)
                    elif mtype == "result" and batch_rows is None:
                        batch_rows = message.get("rows", [])
                    elif mtype == "error" and error_message is None:
                        error_message = str(
                            message.get("message", "Ошибка пакетной детекции в дочернем процессе")
                        )

            if proc is not None:
                proc.join(timeout=1.0)

            if self._cancel_requested:
                self.canceled.emit()
                return

            if batch_rows is not None:
                self.finished.emit(batch_rows)
                return

            if error_message is not None:
                self.failed.emit(error_message)
                return

            exit_code = proc.exitcode if proc is not None else None
            if exit_code is None:
                self.failed.emit("Дочерний процесс пакетной детекции завершился без кода выхода")
            elif exit_code != 0:
                self.failed.emit(
                    "Дочерний процесс пакетной детекции аварийно завершился "
                    f"(код {exit_code}). Проверьте совместимость библиотек TensorFlow/LLVM."
                )
            else:
                self.failed.emit("Дочерний процесс пакетной детекции завершился без результата")
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if proc is not None and proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)


class ImageScene(QGraphicsScene):
    roi_created = pyqtSignal(dict)
    line_created = pyqtSignal(tuple)
    cursor_moved = pyqtSignal(float, float)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.tool_mode: ToolMode = ToolMode.NONE

        self._image_item = None

        self._start_point: QPointF | None = None
        self._temp_rect_item: QGraphicsRectItem | None = None
        self._temp_line_item: QGraphicsLineItem | None = None

        self._polygon_points: list[QPointF] = []
        self._polygon_item: QGraphicsPolygonItem | None = None
        self._poly_preview_line: QGraphicsLineItem | None = None

        self._roi_items: dict[int, QGraphicsPolygonItem] = {}
        self._item_to_roi: dict[QGraphicsItem, int] = {}
        self._nuclei_items: list[QGraphicsItem] = []

        self.setBackgroundBrush(QBrush(QColor("#151515")))

    def set_tool_mode(self, mode: ToolMode) -> None:
        self.tool_mode = mode
        if mode != ToolMode.POLYGON:
            self._clear_polygon_in_progress()

    def set_image_pixmap(self, pixmap: QPixmap | None) -> None:
        self.clear()
        self._reset_temp_state()
        self._roi_items.clear()
        self._item_to_roi.clear()
        self._nuclei_items.clear()

        if pixmap is None or pixmap.isNull():
            self._image_item = None
            self.setSceneRect(QRectF())
            return

        self._image_item = self.addPixmap(pixmap)
        self._image_item.setZValue(0)
        self.setSceneRect(self._image_item.boundingRect())

    def clear_roi_items(self) -> None:
        for item in list(self._roi_items.values()):
            self.removeItem(item)
        self._roi_items.clear()
        self._item_to_roi.clear()

    def clear_nuclei_items(self) -> None:
        for item in self._nuclei_items:
            self.removeItem(item)
        self._nuclei_items.clear()

    def add_roi_item(self, roi_id: int, roi_type: str, points: list[tuple[float, float]]) -> None:
        if len(points) < 3:
            return

        polygon = QPolygonF([QPointF(x, y) for x, y in points])
        item = QGraphicsPolygonItem(polygon)

        if roi_type == "rectangle":
            pen_color = QColor("#ffad33")
            fill_color = QColor(255, 173, 51, 30)
        else:
            pen_color = QColor("#5ec4ff")
            fill_color = QColor(94, 196, 255, 30)

        item.setPen(QPen(pen_color, 2))
        item.setBrush(QBrush(fill_color))
        item.setFlag(QGraphicsItem.ItemIsSelectable, True)
        item.setZValue(20)

        self.addItem(item)
        self._roi_items[roi_id] = item
        self._item_to_roi[item] = roi_id

    def remove_roi_item(self, roi_id: int) -> None:
        item = self._roi_items.pop(roi_id, None)
        if item is None:
            return
        self._item_to_roi.pop(item, None)
        self.removeItem(item)

    def selected_roi_ids(self) -> list[int]:
        result: list[int] = []
        for item in self.selectedItems():
            roi_id = self._item_to_roi.get(item)
            if roi_id is not None:
                result.append(roi_id)
        return result

    def set_nuclei_items(self, nuclei: list[dict]) -> None:
        self.clear_nuclei_items()

        contour_pen = QPen(QColor(50, 230, 120, 180), 1)
        center_pen = QPen(QColor(30, 255, 110, 230), 1)
        center_brush = QBrush(QColor(30, 255, 110, 150))

        for nuc in nuclei:
            contour = nuc.get("contour", [])
            if len(contour) >= 3:
                poly = QPolygonF([QPointF(float(x), float(y)) for x, y in contour])
                contour_item = QGraphicsPolygonItem(poly)
                contour_item.setPen(contour_pen)
                contour_item.setBrush(QBrush(Qt.NoBrush))
                contour_item.setZValue(30)
                self.addItem(contour_item)
                self._nuclei_items.append(contour_item)

            center = nuc.get("center")
            if center is not None:
                cx, cy = float(center[0]), float(center[1])
                radius = 2.0
                center_item = QGraphicsEllipseItem(cx - radius, cy - radius, radius * 2, radius * 2)
                center_item.setPen(center_pen)
                center_item.setBrush(center_brush)
                center_item.setZValue(31)
                self.addItem(center_item)
                self._nuclei_items.append(center_item)

    def cancel_current_drawing(self) -> None:
        self._reset_temp_state()

    def mousePressEvent(self, event) -> None:
        if self._image_item is None:
            super().mousePressEvent(event)
            return

        pos = self._clamp_to_image(event.scenePos())
        button = event.button()

        if self.tool_mode == ToolMode.POLYGON and button == Qt.RightButton and self._polygon_points:
            self._finalize_polygon()
            event.accept()
            return

        if button != Qt.LeftButton:
            super().mousePressEvent(event)
            return

        if self.tool_mode == ToolMode.RECTANGLE:
            self._start_point = pos
            self._temp_rect_item = QGraphicsRectItem(QRectF(pos, pos))
            self._temp_rect_item.setPen(QPen(QColor("#ffad33"), 2, Qt.DashLine))
            self._temp_rect_item.setZValue(50)
            self.addItem(self._temp_rect_item)
            event.accept()
            return

        if self.tool_mode == ToolMode.LINE:
            self._start_point = pos
            self._temp_line_item = QGraphicsLineItem(pos.x(), pos.y(), pos.x(), pos.y())
            self._temp_line_item.setPen(QPen(QColor("#ffd966"), 2))
            self._temp_line_item.setZValue(50)
            self.addItem(self._temp_line_item)
            event.accept()
            return

        if self.tool_mode == ToolMode.POLYGON:
            self._polygon_points.append(pos)
            self._update_polygon_preview(pos)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._image_item is None:
            super().mouseMoveEvent(event)
            return

        pos = self._clamp_to_image(event.scenePos())
        self.cursor_moved.emit(pos.x(), pos.y())

        if self.tool_mode == ToolMode.RECTANGLE and self._temp_rect_item and self._start_point:
            rect = QRectF(self._start_point, pos).normalized()
            self._temp_rect_item.setRect(rect)
            event.accept()
            return

        if self.tool_mode == ToolMode.LINE and self._temp_line_item and self._start_point:
            self._temp_line_item.setLine(self._start_point.x(), self._start_point.y(), pos.x(), pos.y())
            event.accept()
            return

        if self.tool_mode == ToolMode.POLYGON and self._polygon_points:
            self._update_polygon_preview(pos)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._image_item is None:
            super().mouseReleaseEvent(event)
            return

        pos = self._clamp_to_image(event.scenePos())

        if self.tool_mode == ToolMode.RECTANGLE and event.button() == Qt.LeftButton:
            if self._start_point is not None:
                rect = QRectF(self._start_point, pos).normalized()
                self._finalize_rectangle(rect)
                event.accept()
                return

        if self.tool_mode == ToolMode.LINE and event.button() == Qt.LeftButton:
            if self._start_point is not None:
                p1 = (self._start_point.x(), self._start_point.y())
                p2 = (pos.x(), pos.y())
                self._finalize_line(p1, p2)
                event.accept()
                return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if self.tool_mode == ToolMode.POLYGON and event.button() == Qt.LeftButton:
            self._finalize_polygon()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _finalize_rectangle(self, rect: QRectF) -> None:
        if self._temp_rect_item is not None:
            self.removeItem(self._temp_rect_item)
            self._temp_rect_item = None

        self._start_point = None

        if rect.width() < 3 or rect.height() < 3:
            return

        points = [
            (rect.left(), rect.top()),
            (rect.right(), rect.top()),
            (rect.right(), rect.bottom()),
            (rect.left(), rect.bottom()),
        ]
        self.roi_created.emit({"type": "rectangle", "points": points})

    def _finalize_line(self, p1: tuple[float, float], p2: tuple[float, float]) -> None:
        if self._temp_line_item is not None:
            self.removeItem(self._temp_line_item)
            self._temp_line_item = None

        self._start_point = None
        if abs(p1[0] - p2[0]) < 1e-6 and abs(p1[1] - p2[1]) < 1e-6:
            return

        self.line_created.emit((p1, p2))

    def _update_polygon_preview(self, current_pos: QPointF | None = None) -> None:
        if self._polygon_item is not None:
            self.removeItem(self._polygon_item)
            self._polygon_item = None
        if self._poly_preview_line is not None:
            self.removeItem(self._poly_preview_line)
            self._poly_preview_line = None

        if not self._polygon_points:
            return

        poly = QPolygonF(self._polygon_points)
        self._polygon_item = QGraphicsPolygonItem(poly)
        self._polygon_item.setPen(QPen(QColor("#5ec4ff"), 2, Qt.DashLine))
        self._polygon_item.setBrush(QBrush(QColor(94, 196, 255, 18)))
        self._polygon_item.setZValue(50)
        self.addItem(self._polygon_item)

        if current_pos is not None:
            start = self._polygon_points[-1]
            self._poly_preview_line = QGraphicsLineItem(start.x(), start.y(), current_pos.x(), current_pos.y())
            self._poly_preview_line.setPen(QPen(QColor("#5ec4ff"), 1, Qt.DotLine))
            self._poly_preview_line.setZValue(51)
            self.addItem(self._poly_preview_line)

    def _finalize_polygon(self) -> None:
        points = [(p.x(), p.y()) for p in self._polygon_points]
        self._clear_polygon_in_progress()

        if len(points) < 3:
            return

        self.roi_created.emit({"type": "polygon", "points": points})

    def _clear_polygon_in_progress(self) -> None:
        self._polygon_points.clear()
        if self._polygon_item is not None:
            self.removeItem(self._polygon_item)
            self._polygon_item = None
        if self._poly_preview_line is not None:
            self.removeItem(self._poly_preview_line)
            self._poly_preview_line = None

    def _reset_temp_state(self) -> None:
        self._start_point = None
        if self._temp_rect_item is not None:
            self.removeItem(self._temp_rect_item)
            self._temp_rect_item = None
        if self._temp_line_item is not None:
            self.removeItem(self._temp_line_item)
            self._temp_line_item = None
        self._clear_polygon_in_progress()

    def _clamp_to_image(self, point: QPointF) -> QPointF:
        if self._image_item is None:
            return point
        rect = self._image_item.boundingRect()
        x = min(max(point.x(), rect.left()), rect.right())
        y = min(max(point.y(), rect.top()), rect.bottom())
        return QPointF(x, y)


class InvertedLineItem(QGraphicsLineItem):
    def paint(self, painter, option, widget=None) -> None:
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setCompositionMode(QPainter.CompositionMode_Difference)
        pen = QPen(self.pen())
        pen.setCosmetic(True)
        if pen.widthF() < 1.0:
            pen.setWidthF(1.0)
        painter.setPen(pen)
        painter.drawLine(self.line())
        painter.restore()


class CalibrationScene(QGraphicsScene):
    line_updated = pyqtSignal(float)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._image_item = None
        self._image_rect = QRectF()
        self._start_point: QPointF | None = None
        self._line_coords: tuple[tuple[float, float], tuple[float, float]] | None = None

        self._line_item = InvertedLineItem()
        self._line_item.setPen(QPen(QColor(255, 255, 255), 1))
        self._line_item.setZValue(50)
        self.addItem(self._line_item)

        self._cross_h = InvertedLineItem()
        self._cross_h.setPen(QPen(QColor(255, 255, 255), 1, Qt.DotLine))
        self._cross_h.setZValue(40)
        self.addItem(self._cross_h)

        self._cross_v = InvertedLineItem()
        self._cross_v.setPen(QPen(QColor(255, 255, 255), 1, Qt.DotLine))
        self._cross_v.setZValue(40)
        self.addItem(self._cross_v)

        self.setBackgroundBrush(QBrush(QColor("#141414")))

    def set_image_pixmap(self, pixmap: QPixmap | None) -> None:
        self.clear()
        self._line_coords = None
        self._start_point = None

        self._image_item = None
        self._image_rect = QRectF()
        if pixmap is None or pixmap.isNull():
            self.setSceneRect(QRectF())
            return

        self._image_item = self.addPixmap(pixmap)
        self._image_item.setZValue(0)
        self._image_rect = self._image_item.boundingRect()
        self.setSceneRect(self._image_rect)

        self._line_item = InvertedLineItem()
        self._line_item.setPen(QPen(QColor(255, 255, 255), 1))
        self._line_item.setZValue(50)
        self.addItem(self._line_item)

        self._cross_h = InvertedLineItem()
        self._cross_h.setPen(QPen(QColor(255, 255, 255), 1, Qt.DotLine))
        self._cross_h.setZValue(40)
        self.addItem(self._cross_h)

        self._cross_v = InvertedLineItem()
        self._cross_v.setPen(QPen(QColor(255, 255, 255), 1, Qt.DotLine))
        self._cross_v.setZValue(40)
        self.addItem(self._cross_v)

    def clear_line(self) -> None:
        self._line_coords = None
        self._start_point = None
        self._line_item.setLine(QLineF())
        self.line_updated.emit(0.0)

    def line_coords(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        return self._line_coords

    def mousePressEvent(self, event) -> None:
        if self._image_item is None or event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return

        pos = self._clamp_to_image(event.scenePos())
        self._start_point = pos
        self._line_coords = None
        self._line_item.setLine(QLineF(pos, pos))
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._image_item is None:
            super().mouseMoveEvent(event)
            return

        pos = self._clamp_to_image(event.scenePos())
        self._update_crosshair(pos)

        if self._start_point is not None:
            self._line_item.setLine(QLineF(self._start_point, pos))
            self.line_updated.emit(self._line_item.line().length())
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._image_item is None or event.button() != Qt.LeftButton:
            super().mouseReleaseEvent(event)
            return

        if self._start_point is None:
            super().mouseReleaseEvent(event)
            return

        end_pos = self._clamp_to_image(event.scenePos())
        line = QLineF(self._start_point, end_pos)
        self._line_item.setLine(line)
        self._start_point = None

        if line.length() <= 0.0:
            self._line_coords = None
            self.line_updated.emit(0.0)
        else:
            self._line_coords = (
                (line.x1(), line.y1()),
                (line.x2(), line.y2()),
            )
            self.line_updated.emit(line.length())
        event.accept()

    def _update_crosshair(self, pos: QPointF) -> None:
        if self._image_item is None:
            return
        rect = self._image_rect
        self._cross_h.setLine(QLineF(rect.left(), pos.y(), rect.right(), pos.y()))
        self._cross_v.setLine(QLineF(pos.x(), rect.top(), pos.x(), rect.bottom()))

    def _clamp_to_image(self, point: QPointF) -> QPointF:
        rect = self._image_rect
        x = min(max(point.x(), rect.left()), rect.right())
        y = min(max(point.y(), rect.top()), rect.bottom())
        return QPointF(x, y)


class CalibrationView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene, parent: QWidget | None = None) -> None:
        super().__init__(scene, parent)
        self.setRenderHints(self.renderHints() | self.renderHints())
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self.viewport().setCursor(Qt.CrossCursor)
        self._zoom = 1.0

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        self._zoom *= factor
        self.scale(factor, factor)
        event.accept()

    def zoom_in(self) -> None:
        self._zoom *= 1.2
        self.scale(1.2, 1.2)

    def zoom_out(self) -> None:
        self._zoom /= 1.2
        self.scale(1.0 / 1.2, 1.0 / 1.2)

    def zoom_fit(self) -> None:
        rect = self.scene().sceneRect()
        if rect.isNull():
            return
        self.resetTransform()
        self._zoom = 1.0
        self.fitInView(rect, Qt.KeepAspectRatio)

    def zoom_100(self) -> None:
        self.resetTransform()
        self._zoom = 1.0


class InvertedEllipseItem(QGraphicsEllipseItem):
    def paint(self, painter, option, widget=None) -> None:
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setCompositionMode(QPainter.CompositionMode_Difference)
        pen = QPen(self.pen())
        pen.setCosmetic(True)
        if pen.widthF() < 1.0:
            pen.setWidthF(1.0)
        painter.setPen(pen)
        painter.setBrush(self.brush())
        painter.drawEllipse(self.rect())
        painter.restore()


class CellSelectionScene(QGraphicsScene):
    circle_updated = pyqtSignal(float, float)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._image_item = None
        self._image_rect = QRectF()
        self._drag_start: QPointF | None = None
        self._selected_center: QPointF | None = None
        self._radius_px: float | None = None

        self._circle_item = InvertedEllipseItem()
        self._circle_item.setPen(QPen(QColor(255, 255, 255), 1))
        self._circle_item.setBrush(QBrush(Qt.NoBrush))
        self._circle_item.setZValue(40)
        self.addItem(self._circle_item)

        self.setBackgroundBrush(QBrush(QColor("#141414")))

    def set_image_pixmap(self, pixmap: QPixmap | None) -> None:
        self.clear()
        self._drag_start = None
        self._selected_center = None
        self._radius_px = None
        self._image_item = None
        self._image_rect = QRectF()

        if pixmap is None or pixmap.isNull():
            self.setSceneRect(QRectF())
            self.circle_updated.emit(0.0, 0.0)
            return

        self._image_item = self.addPixmap(pixmap)
        self._image_item.setZValue(0)
        self._image_rect = self._image_item.boundingRect()
        self.setSceneRect(self._image_rect)

        self._circle_item = InvertedEllipseItem()
        self._circle_item.setPen(QPen(QColor(255, 255, 255), 1))
        self._circle_item.setBrush(QBrush(Qt.NoBrush))
        self._circle_item.setZValue(40)
        self.addItem(self._circle_item)
        self.circle_updated.emit(0.0, 0.0)

    def clear_circle(self) -> None:
        self._drag_start = None
        self._selected_center = None
        self._radius_px = None
        self._circle_item.setRect(QRectF())
        self.circle_updated.emit(0.0, 0.0)

    def selected_radius_px(self) -> float | None:
        return self._radius_px

    def selected_center_xy(self) -> tuple[float, float] | None:
        if self._selected_center is None:
            return None
        return (float(self._selected_center.x()), float(self._selected_center.y()))

    def mousePressEvent(self, event) -> None:
        if self._image_item is None or event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return

        self._drag_start = self._clamp_to_image(event.scenePos())
        self._selected_center = None
        self._radius_px = None
        self._set_circle_rect(self._drag_start, 0.0)
        self.circle_updated.emit(0.0, 0.0)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._image_item is None or self._drag_start is None:
            super().mouseMoveEvent(event)
            return
        pos = self._clamp_to_image(event.scenePos())
        center, radius = self._circle_from_diameter_points(self._drag_start, pos)
        self._set_circle_rect(center, radius)
        area = float(np.pi * radius * radius)
        self.circle_updated.emit(radius * 2.0, area)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._image_item is None or event.button() != Qt.LeftButton:
            super().mouseReleaseEvent(event)
            return
        if self._drag_start is None:
            super().mouseReleaseEvent(event)
            return

        pos = self._clamp_to_image(event.scenePos())
        center, radius = self._circle_from_diameter_points(self._drag_start, pos)
        if radius < 1.0:
            self.clear_circle()
            event.accept()
            return

        self._drag_start = None
        self._selected_center = center
        self._radius_px = radius
        self._set_circle_rect(center, radius)
        area = float(np.pi * radius * radius)
        self.circle_updated.emit(radius * 2.0, area)
        event.accept()

    def _circle_from_diameter_points(self, p1: QPointF, p2: QPointF) -> tuple[QPointF, float]:
        center = QPointF((p1.x() + p2.x()) * 0.5, (p1.y() + p2.y()) * 0.5)
        radius = float(QLineF(p1, p2).length() * 0.5)
        return center, radius

    def _set_circle_rect(self, center: QPointF, radius: float) -> None:
        rect = QRectF(
            center.x() - radius,
            center.y() - radius,
            radius * 2.0,
            radius * 2.0,
        )
        self._circle_item.setRect(rect)

    def _clamp_to_image(self, point: QPointF) -> QPointF:
        rect = self._image_rect
        x = min(max(point.x(), rect.left()), rect.right())
        y = min(max(point.y(), rect.top()), rect.bottom())
        return QPointF(x, y)


class CellSelectionDialog(QDialog):
    def __init__(
        self,
        image_path: str,
        enhancement_params: dict | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Выбор средней клетки")
        self.resize(1100, 820)
        self.setWindowState(self.windowState() | Qt.WindowMaximized)

        self.image_path = image_path
        self.enhancement_params = backend.normalize_enhancement_params(enhancement_params)
        self.image_bgr = None
        self.selected_radius_px: float | None = None
        self.selected_diameter_px: float | None = None
        self.selected_center_xy: tuple[float, float] | None = None
        self.selected_preset: str = "точный"

        layout = QVBoxLayout(self)

        top_row = QHBoxLayout()
        self.btn_clear_circle = QPushButton("Сбросить круг")
        self.btn_zoom_in = QToolButton()
        self.btn_zoom_in.setText("+")
        self.btn_zoom_out = QToolButton()
        self.btn_zoom_out.setText("-")
        self.btn_zoom_fit = QToolButton()
        self.btn_zoom_fit.setText("По размеру")
        self.path_label = QLabel("Файл: -")
        top_row.addWidget(self.btn_clear_circle)
        top_row.addWidget(self.btn_zoom_in)
        top_row.addWidget(self.btn_zoom_out)
        top_row.addWidget(self.btn_zoom_fit)
        top_row.addWidget(self.path_label, 1)
        layout.addLayout(top_row)

        self.scene_cell = CellSelectionScene(self)
        self.view_cell = CalibrationView(self.scene_cell, self)
        layout.addWidget(self.view_cell, 1)

        form = QFormLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["точный", "чувствительный"])
        self.diameter_label = QLabel("0.0 px")
        self.area_label = QLabel("0.0 px²")
        form.addRow("Режим подбора:", self.preset_combo)
        form.addRow("Диаметр выбранной клетки:", self.diameter_label)
        form.addRow("Площадь выбранной клетки:", self.area_label)
        layout.addLayout(form)

        hint = QLabel(
            "Выделите среднюю по размеру клетку (ядро): "
            "нажмите на один край ядра и протяните до противоположного края. "
            "Колесо мыши, + и - изменяют масштаб для точного выделения."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(self.button_box)

        self.btn_clear_circle.clicked.connect(self.scene_cell.clear_circle)
        self.btn_zoom_in.clicked.connect(self.view_cell.zoom_in)
        self.btn_zoom_out.clicked.connect(self.view_cell.zoom_out)
        self.btn_zoom_fit.clicked.connect(self.view_cell.zoom_fit)
        self.scene_cell.circle_updated.connect(self._on_circle_updated)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self._load_image(image_path)

    def _load_image(self, image_path: str) -> None:
        try:
            self.image_bgr = backend.load_image(image_path)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))
            return
        self.path_label.setText(f"Файл: {Path(image_path).name}")
        preview = backend.apply_image_enhancement(self.image_bgr, self.enhancement_params)
        self.scene_cell.set_image_pixmap(self._bgr_to_pixmap(preview))
        self.view_cell.zoom_100()

    def _bgr_to_pixmap(self, image_bgr) -> QPixmap:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def _on_circle_updated(self, diameter_px: float, area_px: float) -> None:
        self.diameter_label.setText(f"{diameter_px:.1f} px")
        self.area_label.setText(f"{area_px:.1f} px²")

    def accept(self) -> None:
        center_xy = self.scene_cell.selected_center_xy()
        radius = self.scene_cell.selected_radius_px()
        if center_xy is None or radius is None or radius <= 0.0:
            QMessageBox.warning(self, "Нет выделения", "Сначала выделите среднюю клетку кругом")
            return
        self.selected_center_xy = (float(center_xy[0]), float(center_xy[1]))
        self.selected_radius_px = float(radius)
        self.selected_diameter_px = float(radius * 2.0)
        self.selected_preset = str(self.preset_combo.currentText()).strip().lower()
        super().accept()


class ScaleCalibrationDialog(QDialog):
    def __init__(self, parent: QWidget | None = None, initial_path: str | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Калибровка масштаба")
        self.resize(1200, 820)
        self.setWindowState(self.windowState() | Qt.WindowMaximized)

        self.image_path: str | None = None
        self.image_bgr = None
        self.pixels_per_mm: float | None = None
        self.line_coords_result: tuple[tuple[float, float], tuple[float, float]] | None = None
        self.real_length_mm: float | None = None

        layout = QVBoxLayout(self)

        top_row = QHBoxLayout()
        self.btn_open_scale_image = QPushButton("Открыть фото линейки")
        self.btn_clear_line = QPushButton("Сбросить линию")
        self.btn_zoom_in = QToolButton()
        self.btn_zoom_in.setText("+")
        self.btn_zoom_out = QToolButton()
        self.btn_zoom_out.setText("-")
        self.btn_zoom_fit = QToolButton()
        self.btn_zoom_fit.setText("По размеру")
        self.path_label = QLabel("Файл: не выбран")
        top_row.addWidget(self.btn_open_scale_image)
        top_row.addWidget(self.btn_clear_line)
        top_row.addWidget(self.btn_zoom_in)
        top_row.addWidget(self.btn_zoom_out)
        top_row.addWidget(self.btn_zoom_fit)
        top_row.addWidget(self.path_label, 1)
        layout.addLayout(top_row)

        self.scene_cal = CalibrationScene(self)
        self.view_cal = CalibrationView(self.scene_cal, self)
        layout.addWidget(self.view_cal, 1)

        form_row = QFormLayout()
        self.mm_spin = QDoubleSpinBox()
        self.mm_spin.setDecimals(4)
        self.mm_spin.setRange(0.0001, 1_000_000.0)
        self.mm_spin.setValue(1.0)
        self.px_len_label = QLabel("0.00 px")
        self.ppm_preview_label = QLabel("-")
        form_row.addRow("Реальная длина (мм):", self.mm_spin)
        form_row.addRow("Длина линии (px):", self.px_len_label)
        form_row.addRow("Коэффициент (px/mm):", self.ppm_preview_label)
        layout.addLayout(form_row)

        help_label = QLabel(
            "ЛКМ: задать отрезок. Колесо мыши / + / -: масштаб. "
            "Линия и прицел инвертируются относительно фона."
        )
        layout.addWidget(help_label)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(self.button_box)

        self.btn_open_scale_image.clicked.connect(self.open_scale_image)
        self.btn_clear_line.clicked.connect(self.scene_cal.clear_line)
        self.btn_zoom_in.clicked.connect(self.view_cal.zoom_in)
        self.btn_zoom_out.clicked.connect(self.view_cal.zoom_out)
        self.btn_zoom_fit.clicked.connect(self.view_cal.zoom_fit)
        self.scene_cal.line_updated.connect(self._on_line_updated)
        self.mm_spin.valueChanged.connect(self._update_preview)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        if initial_path and os.path.exists(initial_path):
            self.load_scale_image(initial_path)

    def open_scale_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Открыть фото линейки",
            "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg)",
        )
        if not path:
            return
        self.load_scale_image(path)

    def load_scale_image(self, path: str) -> None:
        try:
            image = backend.load_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))
            return

        self.image_path = path
        self.image_bgr = image
        self.path_label.setText(f"Файл: {Path(path).name}")
        self.scene_cal.set_image_pixmap(self._bgr_to_pixmap(image))
        self.view_cal.zoom_100()
        self._on_line_updated(0.0)

    def _bgr_to_pixmap(self, image_bgr) -> QPixmap:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def _on_line_updated(self, px_length: float) -> None:
        self.px_len_label.setText(f"{px_length:.2f} px")
        self._update_preview()

    def _update_preview(self) -> None:
        if self.image_bgr is None:
            self.ppm_preview_label.setText("-")
            return
        line_coords = self.scene_cal.line_coords()
        if line_coords is None:
            self.ppm_preview_label.setText("-")
            return
        try:
            ppm = backend.calibrate_scale(self.image_bgr, line_coords, float(self.mm_spin.value()))
        except Exception:
            self.ppm_preview_label.setText("-")
            return
        self.ppm_preview_label.setText(f"{ppm:.4f}")

    def accept(self) -> None:
        if self.image_bgr is None:
            QMessageBox.warning(self, "Нет изображения", "Сначала откройте фото линейки")
            return
        line_coords = self.scene_cal.line_coords()
        if line_coords is None:
            QMessageBox.warning(self, "Нет линии", "Выберите отрезок по масштабной линейке")
            return

        real_mm = float(self.mm_spin.value())
        try:
            ppm = backend.calibrate_scale(self.image_bgr, line_coords, real_mm)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка калибровки", str(exc))
            return

        self.line_coords_result = line_coords
        self.real_length_mm = real_mm
        self.pixels_per_mm = ppm
        super().accept()


class DetectionParamsDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Параметры детекции ядер")
        self.resize(560, 640)

        self._base_params = backend.get_detection_params()
        self.params_result: dict | None = None
        detector_backends = backend.get_detector_backends()
        model_names = backend.get_pretrained_model_names()
        preprocess_modes = backend.get_preprocess_modes()

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.detector_backend_combo = QComboBox()
        for backend_name in detector_backends:
            self.detector_backend_combo.addItem(
                DETECTOR_BACKEND_LABELS_RU.get(backend_name, backend_name),
                backend_name,
            )
        current_backend = str(self._base_params.get("detector_backend", "cellpose_nuclei"))
        backend_index = self.detector_backend_combo.findData(current_backend)
        if backend_index < 0:
            self.detector_backend_combo.addItem(
                DETECTOR_BACKEND_LABELS_RU.get(current_backend, current_backend),
                current_backend,
            )
            backend_index = self.detector_backend_combo.count() - 1
        self.detector_backend_combo.setCurrentIndex(backend_index)
        form.addRow("Встроенная модель:", self.detector_backend_combo)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        for model_name in model_names:
            self.model_combo.addItem(model_name)
        current_model = str(self._base_params.get("model_name", "2D_versatile_he"))
        if self.model_combo.findText(current_model) < 0:
            self.model_combo.addItem(current_model)
        self.model_combo.setCurrentText(current_model)
        form.addRow("Модель StarDist:", self.model_combo)

        self.prob_spin = QDoubleSpinBox()
        self.prob_spin.setDecimals(3)
        self.prob_spin.setRange(0.0, 1.0)
        self.prob_spin.setSingleStep(0.01)
        self.prob_spin.setValue(float(self._base_params.get("prob_thresh", 0.15)))
        form.addRow("Порог уверенности:", self.prob_spin)

        self.nms_spin = QDoubleSpinBox()
        self.nms_spin.setDecimals(3)
        self.nms_spin.setRange(0.0, 1.0)
        self.nms_spin.setSingleStep(0.01)
        self.nms_spin.setValue(float(self._base_params.get("nms_thresh", 0.55)))
        form.addRow("Порог разделения соседних ядер:", self.nms_spin)

        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setDecimals(3)
        self.scale_spin.setRange(0.2, 3.0)
        self.scale_spin.setSingleStep(0.05)
        self.scale_spin.setValue(float(self._base_params.get("scale", 1.0)))
        form.addRow("Масштаб для сети:", self.scale_spin)

        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(0, 100000)
        self.min_area_spin.setValue(int(self._base_params.get("min_area_px", 18)))
        form.addRow("Мин. площадь ядра (пикс):", self.min_area_spin)

        self.max_area_spin = QSpinBox()
        self.max_area_spin.setRange(0, 500000)
        self.max_area_spin.setSpecialValueText("без ограничения")
        self.max_area_spin.setValue(int(self._base_params.get("max_area_px", 0)))
        form.addRow("Макс. площадь ядра (пикс):", self.max_area_spin)

        self.tiles_x_spin = QSpinBox()
        self.tiles_x_spin.setRange(1, 12)
        self.tiles_x_spin.setValue(int(self._base_params.get("n_tiles_x", 3)))
        form.addRow("Тайлы по горизонтали:", self.tiles_x_spin)

        self.tiles_y_spin = QSpinBox()
        self.tiles_y_spin.setRange(1, 12)
        self.tiles_y_spin.setValue(int(self._base_params.get("n_tiles_y", 3)))
        form.addRow("Тайлы по вертикали:", self.tiles_y_spin)

        self.preprocess_combo = QComboBox()
        for mode in preprocess_modes:
            self.preprocess_combo.addItem(preprocess_mode_to_russian(mode), mode)
        current_mode_key = str(self._base_params.get("preprocess_mode", "rgb"))
        current_mode_index = self.preprocess_combo.findData(current_mode_key)
        if current_mode_index < 0:
            self.preprocess_combo.addItem(
                preprocess_mode_to_russian(current_mode_key),
                current_mode_key,
            )
            current_mode_index = self.preprocess_combo.count() - 1
        self.preprocess_combo.setCurrentIndex(current_mode_index)
        form.addRow("Подготовка изображения:", self.preprocess_combo)

        self.cellpose_model_combo = QComboBox()
        self.cellpose_model_combo.setEditable(False)
        self.cellpose_model_combo.addItem("nuclei", "nuclei")
        current_cellpose_model = str(self._base_params.get("cellpose_model_type", "nuclei"))
        cellpose_idx = self.cellpose_model_combo.findData(current_cellpose_model)
        if cellpose_idx < 0:
            self.cellpose_model_combo.addItem(current_cellpose_model, current_cellpose_model)
            cellpose_idx = self.cellpose_model_combo.count() - 1
        self.cellpose_model_combo.setCurrentIndex(cellpose_idx)
        form.addRow("Модель Cellpose:", self.cellpose_model_combo)

        self.cellpose_diameter_spin = QDoubleSpinBox()
        self.cellpose_diameter_spin.setDecimals(1)
        self.cellpose_diameter_spin.setRange(1.0, 200.0)
        self.cellpose_diameter_spin.setSingleStep(0.5)
        self.cellpose_diameter_spin.setValue(float(self._base_params.get("cellpose_diameter_px", 14.0)))
        form.addRow("Оценка диаметра ядра (пикс):", self.cellpose_diameter_spin)

        self.cellpose_flow_spin = QDoubleSpinBox()
        self.cellpose_flow_spin.setDecimals(2)
        self.cellpose_flow_spin.setRange(0.0, 2.0)
        self.cellpose_flow_spin.setSingleStep(0.05)
        self.cellpose_flow_spin.setValue(
            float(self._base_params.get("cellpose_flow_threshold", 0.40))
        )
        form.addRow("Cellpose: порог потока:", self.cellpose_flow_spin)

        self.cellpose_cellprob_spin = QDoubleSpinBox()
        self.cellpose_cellprob_spin.setDecimals(2)
        self.cellpose_cellprob_spin.setRange(-10.0, 10.0)
        self.cellpose_cellprob_spin.setSingleStep(0.1)
        self.cellpose_cellprob_spin.setValue(
            float(self._base_params.get("cellpose_cellprob_threshold", -0.50))
        )
        form.addRow("Cellpose: порог вероятности маски:", self.cellpose_cellprob_spin)

        self.norm_low_spin = QDoubleSpinBox()
        self.norm_low_spin.setDecimals(2)
        self.norm_low_spin.setRange(0.0, 99.0)
        self.norm_low_spin.setSingleStep(0.1)
        self.norm_low_spin.setValue(float(self._base_params.get("norm_p_low", 1.0)))
        form.addRow("Нижний перцентиль нормализации:", self.norm_low_spin)

        self.norm_high_spin = QDoubleSpinBox()
        self.norm_high_spin.setDecimals(2)
        self.norm_high_spin.setRange(1.0, 100.0)
        self.norm_high_spin.setSingleStep(0.1)
        self.norm_high_spin.setValue(float(self._base_params.get("norm_p_high", 99.8)))
        form.addRow("Верхний перцентиль нормализации:", self.norm_high_spin)

        self.purple_filter_check = QCheckBox(
            "Фильтровать объекты по окраске гематоксилином (фиолетовые ядра)"
        )
        self.purple_filter_check.setChecked(bool(self._base_params.get("purple_filter_enabled", True)))
        form.addRow(self.purple_filter_check)

        self.center_purple_check = QCheckBox("Требовать окраску в центре ядра")
        self.center_purple_check.setChecked(bool(self._base_params.get("require_center_purple", True)))
        form.addRow(self.center_purple_check)

        self.hue_min_spin = QSpinBox()
        self.hue_min_spin.setRange(0, 179)
        self.hue_min_spin.setValue(int(self._base_params.get("purple_h_min", 120)))
        form.addRow("Мин. оттенок фиолетового:", self.hue_min_spin)

        self.hue_max_spin = QSpinBox()
        self.hue_max_spin.setRange(0, 179)
        self.hue_max_spin.setValue(int(self._base_params.get("purple_h_max", 170)))
        form.addRow("Макс. оттенок фиолетового:", self.hue_max_spin)

        self.sat_min_spin = QSpinBox()
        self.sat_min_spin.setRange(0, 255)
        self.sat_min_spin.setValue(int(self._base_params.get("purple_s_min", 45)))
        form.addRow("Мин. насыщенность:", self.sat_min_spin)

        self.val_max_spin = QSpinBox()
        self.val_max_spin.setRange(0, 255)
        self.val_max_spin.setValue(int(self._base_params.get("purple_v_max", 200)))
        form.addRow("Макс. яркость:", self.val_max_spin)

        self.min_ratio_spin = QDoubleSpinBox()
        self.min_ratio_spin.setDecimals(3)
        self.min_ratio_spin.setRange(0.0, 1.0)
        self.min_ratio_spin.setSingleStep(0.01)
        self.min_ratio_spin.setValue(float(self._base_params.get("min_purple_ratio", 0.20)))
        form.addRow("Мин. доля окрашенных пикселей в ядре:", self.min_ratio_spin)

        layout.addLayout(form)

        preset_row = QHBoxLayout()
        self.btn_preset_precise = QPushButton("Точный")
        self.btn_preset_sensitive = QPushButton("Чувствительный")
        preset_row.addWidget(QLabel("Пресет:"))
        preset_row.addWidget(self.btn_preset_precise)
        preset_row.addWidget(self.btn_preset_sensitive)
        preset_row.addStretch(1)
        layout.addLayout(preset_row)

        note = QLabel(
            "Точный — меньше ложных срабатываний. "
            "Чувствительный — находит больше ядер, но может добавить лишние объекты."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.purple_filter_check.toggled.connect(self._toggle_purple_controls)
        self.detector_backend_combo.currentIndexChanged.connect(self._toggle_detector_controls)
        self.btn_preset_precise.clicked.connect(self._apply_precision_preset)
        self.btn_preset_sensitive.clicked.connect(self._apply_sensitive_preset)
        self._toggle_purple_controls(self.purple_filter_check.isChecked())
        self._toggle_detector_controls()

    def _toggle_purple_controls(self, enabled: bool) -> None:
        for widget in [
            self.center_purple_check,
            self.hue_min_spin,
            self.hue_max_spin,
            self.sat_min_spin,
            self.val_max_spin,
            self.min_ratio_spin,
        ]:
            widget.setEnabled(enabled)

    def _toggle_detector_controls(self, *_args) -> None:
        backend_name = self.detector_backend_combo.currentData()
        if backend_name is None:
            backend_name = DETECTOR_BACKEND_KEYS_BY_LABEL_RU.get(
                self.detector_backend_combo.currentText().strip(),
                "cellpose_nuclei",
            )
        backend_name = str(backend_name).strip().lower()
        use_stardist = backend_name == "stardist"

        for widget in [
            self.model_combo,
            self.prob_spin,
            self.nms_spin,
            self.scale_spin,
            self.tiles_x_spin,
            self.tiles_y_spin,
            self.preprocess_combo,
            self.norm_low_spin,
            self.norm_high_spin,
        ]:
            widget.setEnabled(use_stardist)

        for widget in [
            self.cellpose_model_combo,
            self.cellpose_diameter_spin,
            self.cellpose_flow_spin,
            self.cellpose_cellprob_spin,
        ]:
            widget.setEnabled(not use_stardist)

    def _apply_precision_preset(self) -> None:
        self.model_combo.setCurrentText("2D_versatile_he")
        self.prob_spin.setValue(0.15)
        self.nms_spin.setValue(0.55)
        self.scale_spin.setValue(1.0)
        self.min_area_spin.setValue(18)
        self.max_area_spin.setValue(0)
        self.tiles_x_spin.setValue(3)
        self.tiles_y_spin.setValue(3)
        self.preprocess_combo.setCurrentIndex(
            max(0, self.preprocess_combo.findData("rgb"))
        )
        self.norm_low_spin.setValue(1.0)
        self.norm_high_spin.setValue(99.8)
        self.purple_filter_check.setChecked(True)
        self.center_purple_check.setChecked(True)
        self.hue_min_spin.setValue(120)
        self.hue_max_spin.setValue(170)
        self.sat_min_spin.setValue(42)
        self.val_max_spin.setValue(200)
        self.min_ratio_spin.setValue(0.20)
        self.cellpose_model_combo.setCurrentIndex(max(0, self.cellpose_model_combo.findData("nuclei")))
        self.cellpose_flow_spin.setValue(0.50)
        self.cellpose_cellprob_spin.setValue(-0.20)

    def _apply_sensitive_preset(self) -> None:
        self.model_combo.setCurrentText("2D_versatile_he")
        self.prob_spin.setValue(0.15)
        self.nms_spin.setValue(0.70)
        self.scale_spin.setValue(1.0)
        self.min_area_spin.setValue(10)
        self.max_area_spin.setValue(0)
        self.tiles_x_spin.setValue(3)
        self.tiles_y_spin.setValue(3)
        self.preprocess_combo.setCurrentIndex(
            max(0, self.preprocess_combo.findData("rgb"))
        )
        self.norm_low_spin.setValue(1.0)
        self.norm_high_spin.setValue(99.8)
        self.purple_filter_check.setChecked(True)
        self.center_purple_check.setChecked(True)
        self.hue_min_spin.setValue(118)
        self.hue_max_spin.setValue(172)
        self.sat_min_spin.setValue(30)
        self.val_max_spin.setValue(215)
        self.min_ratio_spin.setValue(0.12)
        self.cellpose_model_combo.setCurrentIndex(max(0, self.cellpose_model_combo.findData("nuclei")))
        self.cellpose_flow_spin.setValue(0.30)
        self.cellpose_cellprob_spin.setValue(-1.00)

    def _collect_params(self) -> dict:
        detector_backend = self.detector_backend_combo.currentData()
        if detector_backend is None:
            detector_backend = DETECTOR_BACKEND_KEYS_BY_LABEL_RU.get(
                self.detector_backend_combo.currentText().strip(),
                "cellpose_nuclei",
            )
        preprocess_mode = self.preprocess_combo.currentData()
        if preprocess_mode is None:
            preprocess_mode = PREPROCESS_MODE_KEYS_BY_LABEL_RU.get(
                self.preprocess_combo.currentText().strip(),
                "rgb",
            )
        cellpose_model_type = self.cellpose_model_combo.currentData()
        if cellpose_model_type is None:
            cellpose_model_type = self.cellpose_model_combo.currentText().strip() or "nuclei"
        return {
            "detector_backend": str(detector_backend).strip(),
            "model_name": str(self.model_combo.currentText()).strip() or "2D_versatile_he",
            "prob_thresh": float(self.prob_spin.value()),
            "nms_thresh": float(self.nms_spin.value()),
            "scale": float(self.scale_spin.value()),
            "min_area_px": int(self.min_area_spin.value()),
            "max_area_px": int(self.max_area_spin.value()),
            "n_tiles_x": int(self.tiles_x_spin.value()),
            "n_tiles_y": int(self.tiles_y_spin.value()),
            "preprocess_mode": str(preprocess_mode).strip(),
            "norm_p_low": float(self.norm_low_spin.value()),
            "norm_p_high": float(self.norm_high_spin.value()),
            "purple_filter_enabled": bool(self.purple_filter_check.isChecked()),
            "purple_h_min": int(self.hue_min_spin.value()),
            "purple_h_max": int(self.hue_max_spin.value()),
            "purple_s_min": int(self.sat_min_spin.value()),
            "purple_v_max": int(self.val_max_spin.value()),
            "min_purple_ratio": float(self.min_ratio_spin.value()),
            "require_center_purple": bool(self.center_purple_check.isChecked()),
            "cellpose_model_type": str(cellpose_model_type).strip() or "nuclei",
            "cellpose_diameter_px": float(self.cellpose_diameter_spin.value()),
            "cellpose_flow_threshold": float(self.cellpose_flow_spin.value()),
            "cellpose_cellprob_threshold": float(self.cellpose_cellprob_spin.value()),
        }

    def accept(self) -> None:
        params = self._collect_params()
        try:
            backend.set_detection_params(params)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка параметров", str(exc))
            return
        self.params_result = params
        super().accept()


class ColorTuningDialog(QDialog):
    def __init__(
        self,
        image_path: str,
        initial_params: dict | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Настройка цветокоррекции")
        self.resize(1180, 860)

        self.image_path = image_path
        self.result_params: dict | None = None
        self._base_display_image: np.ndarray | None = None
        self._current_params = backend.normalize_enhancement_params(initial_params)
        self._preview_initialized = False

        layout = QVBoxLayout(self)

        top_row = QHBoxLayout()
        self.path_label = QLabel("Файл: -")
        self.btn_zoom_in = QToolButton()
        self.btn_zoom_in.setText("+")
        self.btn_zoom_out = QToolButton()
        self.btn_zoom_out.setText("-")
        self.btn_zoom_fit = QToolButton()
        self.btn_zoom_fit.setText("По размеру")
        top_row.addWidget(self.path_label, 1)
        top_row.addWidget(self.btn_zoom_in)
        top_row.addWidget(self.btn_zoom_out)
        top_row.addWidget(self.btn_zoom_fit)
        layout.addLayout(top_row)

        self.scene_preview = QGraphicsScene(self)
        self.view_preview = CalibrationView(self.scene_preview, self)
        layout.addWidget(self.view_preview, 1)

        sliders_group = QGroupBox("Параметры цветокоррекции")
        sliders_layout = QFormLayout(sliders_group)

        self.slider_saturation = QSlider(Qt.Horizontal)
        self.slider_saturation.setRange(0, 300)
        self.slider_brightness = QSlider(Qt.Horizontal)
        self.slider_brightness.setRange(-100, 100)
        self.slider_contrast = QSlider(Qt.Horizontal)
        self.slider_contrast.setRange(20, 300)
        self.slider_sharpness = QSlider(Qt.Horizontal)
        self.slider_sharpness.setRange(0, 300)

        sat, bri, con, sha = enhancement_to_slider_values(self._current_params)
        self.slider_saturation.setValue(sat)
        self.slider_brightness.setValue(bri)
        self.slider_contrast.setValue(con)
        self.slider_sharpness.setValue(sha)

        self.lbl_saturation = QLabel()
        self.lbl_brightness = QLabel()
        self.lbl_contrast = QLabel()
        self.lbl_sharpness = QLabel()
        self._update_slider_labels()

        sat_row = QHBoxLayout()
        sat_row.addWidget(self.slider_saturation, 1)
        sat_row.addWidget(self.lbl_saturation)
        sliders_layout.addRow("Цветность:", sat_row)

        bri_row = QHBoxLayout()
        bri_row.addWidget(self.slider_brightness, 1)
        bri_row.addWidget(self.lbl_brightness)
        sliders_layout.addRow("Яркость:", bri_row)

        con_row = QHBoxLayout()
        con_row.addWidget(self.slider_contrast, 1)
        con_row.addWidget(self.lbl_contrast)
        sliders_layout.addRow("Контрастность:", con_row)

        sha_row = QHBoxLayout()
        sha_row.addWidget(self.slider_sharpness, 1)
        sha_row.addWidget(self.lbl_sharpness)
        sliders_layout.addRow("Резкость:", sha_row)

        layout.addWidget(sliders_group)

        actions_row = QHBoxLayout()
        self.btn_reset = QPushButton("Сбросить")
        actions_row.addWidget(self.btn_reset)
        actions_row.addStretch(1)
        layout.addLayout(actions_row)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(self.button_box)

        self.slider_saturation.valueChanged.connect(self._on_sliders_changed)
        self.slider_brightness.valueChanged.connect(self._on_sliders_changed)
        self.slider_contrast.valueChanged.connect(self._on_sliders_changed)
        self.slider_sharpness.valueChanged.connect(self._on_sliders_changed)
        self.btn_reset.clicked.connect(self._reset_values)
        self.btn_zoom_in.clicked.connect(self.view_preview.zoom_in)
        self.btn_zoom_out.clicked.connect(self.view_preview.zoom_out)
        self.btn_zoom_fit.clicked.connect(self.view_preview.zoom_fit)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self._load_image()

    def _load_image(self) -> None:
        display_img, _ = backend.load_display_image(self.image_path, max_side=2200)
        self._base_display_image = display_img
        self.path_label.setText(f"Файл: {Path(self.image_path).name}")
        self._update_preview()

    def _current_from_sliders(self) -> dict:
        return slider_values_to_enhancement(
            int(self.slider_saturation.value()),
            int(self.slider_brightness.value()),
            int(self.slider_contrast.value()),
            int(self.slider_sharpness.value()),
        )

    def _update_slider_labels(self) -> None:
        params = self._current_from_sliders()
        self.lbl_saturation.setText(f"{params['saturation']:.2f}")
        self.lbl_brightness.setText(f"{params['brightness']:.0f}")
        self.lbl_contrast.setText(f"{params['contrast']:.2f}")
        self.lbl_sharpness.setText(f"{params['sharpness']:.2f}")

    def _on_sliders_changed(self) -> None:
        self._update_slider_labels()
        self._update_preview()

    def _update_preview(self) -> None:
        if self._base_display_image is None:
            return
        params = self._current_from_sliders()
        enhanced = backend.apply_image_enhancement(self._base_display_image, params)
        pixmap = self._bgr_to_pixmap(enhanced)
        self.scene_preview.clear()
        self.scene_preview.addPixmap(pixmap)
        self.scene_preview.setSceneRect(self.scene_preview.itemsBoundingRect())
        if not self._preview_initialized:
            self.view_preview.zoom_fit()
            self._preview_initialized = True

    def _reset_values(self) -> None:
        defaults = backend.get_default_enhancement_params()
        sat, bri, con, sha = enhancement_to_slider_values(defaults)
        self.slider_saturation.setValue(sat)
        self.slider_brightness.setValue(bri)
        self.slider_contrast.setValue(con)
        self.slider_sharpness.setValue(sha)

    def _bgr_to_pixmap(self, image_bgr) -> QPixmap:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def accept(self) -> None:
        self.result_params = self._current_from_sliders()
        super().accept()


class RoiNameDialog(QDialog):
    def __init__(
        self,
        suggestions: list[str],
        default_name: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Название ROI")
        self.resize(420, 120)
        self.result_name: str | None = None

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.combo_name = QComboBox()
        self.combo_name.setEditable(True)
        seen: set[str] = set()
        for value in suggestions:
            text = str(value).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            self.combo_name.addItem(text)
        if default_name and self.combo_name.findText(default_name) < 0:
            self.combo_name.addItem(default_name)
        self.combo_name.setCurrentText(default_name)
        form.addRow("Название:", self.combo_name)
        layout.addLayout(form)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(self.button_box)

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    def accept(self) -> None:
        text = self.combo_name.currentText().strip()
        if not text:
            QMessageBox.warning(self, "Пустое название", "Введите название ROI")
            return
        self.result_name = text
        super().accept()


class RoiAnnotationDialog(QDialog):
    def __init__(
        self,
        image_path: str,
        mode_key: str,
        enhancement_params: dict | None,
        name_suggestions: list[str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Разметка областей интереса")
        self.resize(1500, 900)

        self.image_path = image_path
        self.mode_key = mode_key
        self.enhancement_params = backend.normalize_enhancement_params(enhancement_params)
        self.name_suggestions = [str(x).strip() for x in name_suggestions if str(x).strip()]
        self.new_names: list[str] = []

        self.display_scale = 1.0
        self.rois: list[dict] = []
        self.next_roi_id = 1

        layout = QVBoxLayout(self)

        top_row = QHBoxLayout()
        self.path_label = QLabel(f"Файл: {Path(image_path).name}")
        self.btn_zoom_in = QToolButton()
        self.btn_zoom_in.setText("+")
        self.btn_zoom_out = QToolButton()
        self.btn_zoom_out.setText("-")
        self.btn_zoom_fit = QToolButton()
        self.btn_zoom_fit.setText("По размеру")
        top_row.addWidget(self.path_label, 1)
        top_row.addWidget(self.btn_zoom_in)
        top_row.addWidget(self.btn_zoom_out)
        top_row.addWidget(self.btn_zoom_fit)
        layout.addLayout(top_row)

        center = QHBoxLayout()
        layout.addLayout(center, 1)

        controls = QVBoxLayout()
        self.btn_rect_mode = QPushButton("Прямоугольник")
        self.btn_poly_mode = QPushButton("Полигон")
        self.btn_delete_roi = QPushButton("Удалить выбранные")
        self.btn_clear_roi = QPushButton("Удалить все ROI")
        self.btn_rect_mode.setCheckable(True)
        self.btn_poly_mode.setCheckable(True)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.btn_rect_mode)
        self.mode_group.addButton(self.btn_poly_mode)
        controls.addWidget(self.btn_rect_mode)
        controls.addWidget(self.btn_poly_mode)
        controls.addWidget(self.btn_delete_roi)
        controls.addWidget(self.btn_clear_roi)
        controls.addStretch(1)
        controls_widget = QWidget(self)
        controls_widget.setLayout(controls)
        controls_widget.setFixedWidth(220)
        center.addWidget(controls_widget)

        self.scene = ImageScene(self)
        self.view = CalibrationView(self.scene, self)
        center.addWidget(self.view, 1)

        right = QVBoxLayout()
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["ID", "Название", "Тип"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right.addWidget(self.table, 1)
        right_widget = QWidget(self)
        right_widget.setLayout(right)
        right_widget.setFixedWidth(320)
        center.addWidget(right_widget)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(self.button_box)

        self.btn_zoom_in.clicked.connect(self.view.zoom_in)
        self.btn_zoom_out.clicked.connect(self.view.zoom_out)
        self.btn_zoom_fit.clicked.connect(self.view.zoom_fit)
        self.btn_delete_roi.clicked.connect(self._delete_selected_rois)
        self.btn_clear_roi.clicked.connect(self._clear_rois)
        self.scene.roi_created.connect(self._on_roi_created)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.btn_rect_mode.clicked.connect(lambda: self._set_mode(ToolMode.RECTANGLE))
        self.btn_poly_mode.clicked.connect(lambda: self._set_mode(ToolMode.POLYGON))

        self._load_image()
        self._configure_mode()

    def _configure_mode(self) -> None:
        if self.mode_key == BATCH_MODE_RECT:
            self.btn_rect_mode.setChecked(True)
            self.btn_poly_mode.setEnabled(False)
            self._set_mode(ToolMode.RECTANGLE)
        else:
            self.btn_poly_mode.setChecked(True)
            self.btn_rect_mode.setEnabled(False)
            self._set_mode(ToolMode.POLYGON)

    def _set_mode(self, mode: ToolMode) -> None:
        self.scene.set_tool_mode(mode)

    def _load_image(self) -> None:
        display_img, self.display_scale = backend.load_display_image(self.image_path, max_side=2200)
        enhanced_display = backend.apply_image_enhancement(display_img, self.enhancement_params)
        pixmap = self._bgr_to_pixmap(enhanced_display)
        self.scene.set_image_pixmap(pixmap)
        self.view.zoom_fit()

    def _bgr_to_pixmap(self, image_bgr) -> QPixmap:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def _on_roi_created(self, payload: dict) -> None:
        points_display = payload.get("points", [])
        roi_type = payload.get("type", "polygon")
        if len(points_display) < 3:
            return

        name_dialog = RoiNameDialog(
            suggestions=self.name_suggestions,
            default_name=f"Область {self.next_roi_id}",
            parent=self,
        )
        if name_dialog.exec_() != QDialog.Accepted or not name_dialog.result_name:
            return

        roi_name = name_dialog.result_name
        points_orig = [
            (float(x) * self.display_scale, float(y) * self.display_scale)
            for x, y in points_display
        ]
        roi = {
            "id": self.next_roi_id,
            "name": roi_name,
            "type": roi_type,
            "points": points_orig,
        }
        self.rois.append(roi)
        self.scene.add_roi_item(self.next_roi_id, roi_type, points_display)
        self.next_roi_id += 1

        if roi_name not in self.name_suggestions:
            self.name_suggestions.append(roi_name)
        if roi_name not in self.new_names:
            self.new_names.append(roi_name)

        self._refresh_table()

    def _refresh_table(self) -> None:
        self.table.setRowCount(len(self.rois))
        for row_idx, roi in enumerate(self.rois):
            self.table.setItem(row_idx, 0, QTableWidgetItem(str(roi["id"])))
            self.table.setItem(row_idx, 1, QTableWidgetItem(str(roi.get("name", ""))))
            self.table.setItem(row_idx, 2, QTableWidgetItem(str(roi.get("type", ""))))

    def _delete_selected_rois(self) -> None:
        ids_from_scene = set(self.scene.selected_roi_ids())
        ids_from_table: set[int] = set()
        for idx in self.table.selectionModel().selectedRows():
            roi_item = self.table.item(idx.row(), 0)
            if roi_item is None:
                continue
            try:
                ids_from_table.add(int(roi_item.text()))
            except Exception:
                continue
        selected_ids = ids_from_scene | ids_from_table
        if not selected_ids:
            return

        self.rois = [roi for roi in self.rois if int(roi["id"]) not in selected_ids]
        for roi_id in selected_ids:
            self.scene.remove_roi_item(roi_id)
        self._refresh_table()

    def _clear_rois(self) -> None:
        self.rois.clear()
        self.scene.clear_roi_items()
        self._refresh_table()

    def accept(self) -> None:
        if not self.rois:
            QMessageBox.warning(
                self,
                "Нет ROI",
                "Добавьте хотя бы одну область интереса на изображении",
            )
            return
        super().accept()


class BatchSetupDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Пакетная обработка")
        self.resize(620, 220)

        self.folder_path: str | None = None
        self.mode_key: str = BATCH_MODE_FULL

        layout = QVBoxLayout(self)
        form = QFormLayout()

        folder_row = QHBoxLayout()
        self.input_folder = QLineEdit()
        self.input_folder.setPlaceholderText("Выберите папку с изображениями")
        self.btn_browse = QPushButton("Обзор")
        folder_row.addWidget(self.input_folder, 1)
        folder_row.addWidget(self.btn_browse)
        form.addRow("Папка:", folder_row)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem(BATCH_MODE_LABELS_RU[BATCH_MODE_FULL], BATCH_MODE_FULL)
        self.mode_combo.addItem(BATCH_MODE_LABELS_RU[BATCH_MODE_RECT], BATCH_MODE_RECT)
        self.mode_combo.addItem(BATCH_MODE_LABELS_RU[BATCH_MODE_POLY], BATCH_MODE_POLY)
        form.addRow("Режим:", self.mode_combo)

        layout.addLayout(form)

        self.hint = QLabel(
            "Режим по всему кадру не требует ручной разметки ROI. "
            "В режимах по прямоугольникам и полигонам разметка будет выполнена для каждого фото."
        )
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(self.button_box)

        self.btn_browse.clicked.connect(self._browse_folder)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
        if folder:
            self.input_folder.setText(folder)

    def accept(self) -> None:
        folder = self.input_folder.text().strip()
        if not folder:
            QMessageBox.warning(self, "Нет папки", "Укажите папку с изображениями")
            return

        path = Path(folder)
        if not path.exists() or not path.is_dir():
            QMessageBox.warning(self, "Ошибка пути", "Указанная папка не существует")
            return

        images = [
            p for p in sorted(path.iterdir())
            if p.is_file() and p.suffix.lower() in backend.SUPPORTED_IMAGE_FORMATS
        ]
        if not images:
            QMessageBox.warning(
                self,
                "Нет изображений",
                "В папке не найдено поддерживаемых изображений (TIFF/PNG/JPEG)",
            )
            return

        self.folder_path = str(path)
        self.mode_key = str(self.mode_combo.currentData())
        super().accept()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Анализ ядер и ROI")
        self.resize(1600, 900)

        self.image_path: str | None = None
        self.display_scale: float = 1.0  # orig_px / display_px
        self.pixels_per_mm: float | None = None

        self.rois: list[dict] = []
        self.nuclei: list[dict] = []
        self.next_roi_id = 1
        self.metrics_rows: list[dict] = []
        self.cell_diameter_px: float | None = None
        self.cell_tune_preset: str = "точный"
        self.enhancement_params = backend.get_default_enhancement_params()
        self.roi_name_history: list[str] = []
        self._initial_scale_completed = False

        self.undo_stack: list[dict] = []
        self.max_undo = 100
        self._restoring_state = False

        self._det_thread: QThread | None = None
        self._det_worker: DetectionWorker | None = None
        self._batch_thread: QThread | None = None
        self._batch_worker: BatchDetectionWorker | None = None
        self._batch_progress: QProgressDialog | None = None
        self._pending_batch_folder_name: str = ""
        self._pending_batch_total_files: int = 0

        self._build_ui()
        self._connect_signals()
        self._apply_dark_theme()
        self._update_detector_status()
        self._update_scale_status()
        QTimer.singleShot(0, self._enforce_initial_scale)

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        # Left panel
        left = QWidget(self)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        self.btn_open = QPushButton("Открыть изображение")
        self.btn_load_model = QPushButton("Загрузить модель (.pt/.onnx)")
        self.btn_reset_model = QPushButton("Сбросить модель")
        self.btn_calibrate = QPushButton("Калибровать масштаб")
        self.btn_detect = QPushButton("Детектировать ядра")
        self.btn_cell_tune = QPushButton("Подобрать по клетке")
        self.btn_detection_params = QPushButton("Параметры детекции")
        self.btn_batch = QPushButton("Пакетная обработка")

        self.btn_rect = QPushButton("Прямоугольник")
        self.btn_poly = QPushButton("Полигон")
        self.btn_line = QPushButton("Линия")
        self.btn_zoom_in_main = QPushButton("Увеличить (+)")
        self.btn_zoom_out_main = QPushButton("Уменьшить (-)")
        self.btn_zoom_100_main = QPushButton("Масштаб 100%")

        self.btn_rect.setCheckable(True)
        self.btn_poly.setCheckable(True)
        self.btn_line.setCheckable(True)

        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        self.tool_group.addButton(self.btn_rect)
        self.tool_group.addButton(self.btn_poly)
        self.tool_group.addButton(self.btn_line)

        left_layout.addWidget(self.btn_open)
        left_layout.addWidget(self.btn_load_model)
        left_layout.addWidget(self.btn_reset_model)
        left_layout.addWidget(self.btn_calibrate)
        left_layout.addWidget(self.btn_detect)
        left_layout.addWidget(self.btn_cell_tune)
        left_layout.addWidget(self.btn_detection_params)
        left_layout.addWidget(self.btn_batch)

        enhance_group = QGroupBox("Цветокоррекция")
        enhance_layout = QFormLayout(enhance_group)

        self.slider_saturation = QSlider(Qt.Horizontal)
        self.slider_saturation.setRange(0, 300)
        self.slider_brightness = QSlider(Qt.Horizontal)
        self.slider_brightness.setRange(-100, 100)
        self.slider_contrast = QSlider(Qt.Horizontal)
        self.slider_contrast.setRange(20, 300)
        self.slider_sharpness = QSlider(Qt.Horizontal)
        self.slider_sharpness.setRange(0, 300)

        sat, bri, con, sha = enhancement_to_slider_values(self.enhancement_params)
        self.slider_saturation.setValue(sat)
        self.slider_brightness.setValue(bri)
        self.slider_contrast.setValue(con)
        self.slider_sharpness.setValue(sha)

        self.lbl_saturation = QLabel()
        self.lbl_brightness = QLabel()
        self.lbl_contrast = QLabel()
        self.lbl_sharpness = QLabel()
        self._update_enhancement_labels()

        sat_row = QHBoxLayout()
        sat_row.addWidget(self.slider_saturation, 1)
        sat_row.addWidget(self.lbl_saturation)
        enhance_layout.addRow("Цветность:", sat_row)

        bri_row = QHBoxLayout()
        bri_row.addWidget(self.slider_brightness, 1)
        bri_row.addWidget(self.lbl_brightness)
        enhance_layout.addRow("Яркость:", bri_row)

        con_row = QHBoxLayout()
        con_row.addWidget(self.slider_contrast, 1)
        con_row.addWidget(self.lbl_contrast)
        enhance_layout.addRow("Контрастность:", con_row)

        sha_row = QHBoxLayout()
        sha_row.addWidget(self.slider_sharpness, 1)
        sha_row.addWidget(self.lbl_sharpness)
        enhance_layout.addRow("Резкость:", sha_row)

        self.btn_enhance_reset = QPushButton("Сбросить цветокоррекцию")
        enhance_layout.addRow(self.btn_enhance_reset)

        left_layout.addWidget(enhance_group)
        left_layout.addSpacing(12)
        left_layout.addWidget(self.btn_rect)
        left_layout.addWidget(self.btn_poly)
        left_layout.addWidget(self.btn_line)
        left_layout.addWidget(self.btn_zoom_in_main)
        left_layout.addWidget(self.btn_zoom_out_main)
        left_layout.addWidget(self.btn_zoom_100_main)
        left_layout.addStretch(1)

        left.setFixedWidth(320)

        # Center image area
        self.scene = ImageScene(self)
        self.view = CalibrationView(self.scene, self)

        # Right panel
        right = QWidget(self)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["ROI ID", "Тип", "Площадь (мм²)", "Ядра", "Плотность"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.btn_export = QPushButton("Экспортировать")

        right_layout.addWidget(self.table, 1)
        right_layout.addWidget(self.btn_export)
        right.setFixedWidth(430)

        root_layout.addWidget(left)
        root_layout.addWidget(self.view, 1)
        root_layout.addWidget(right)

        self.setCentralWidget(central)

        self.scale_label = QLabel("Масштаб: не задан")
        self.coord_label = QLabel("Курсор: x -, y -")
        self.proc_label = QLabel("Статус: готово")
        self.detector_label = QLabel("Детектор: -")

        self.statusBar().addPermanentWidget(self.scale_label)
        self.statusBar().addPermanentWidget(self.coord_label)
        self.statusBar().addPermanentWidget(self.proc_label)
        self.statusBar().addPermanentWidget(self.detector_label)

        self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.shortcut_delete = QShortcut(QKeySequence("Delete"), self)
        self.shortcut_escape = QShortcut(QKeySequence("Esc"), self)

    def _connect_signals(self) -> None:
        self.btn_open.clicked.connect(self.open_image)
        self.btn_load_model.clicked.connect(self.load_custom_model)
        self.btn_reset_model.clicked.connect(self.reset_custom_model)
        self.btn_calibrate.clicked.connect(self.activate_calibration)
        self.btn_detect.clicked.connect(self.detect_nuclei)
        self.btn_cell_tune.clicked.connect(self.run_cell_tuning_dialog)
        self.btn_detection_params.clicked.connect(self.open_detection_params)
        self.btn_batch.clicked.connect(self.run_batch_processing)
        self.btn_export.clicked.connect(self.export_results)

        self.btn_rect.clicked.connect(lambda: self.set_tool(ToolMode.RECTANGLE))
        self.btn_poly.clicked.connect(lambda: self.set_tool(ToolMode.POLYGON))
        self.btn_line.clicked.connect(lambda: self.set_tool(ToolMode.LINE))
        self.btn_zoom_in_main.clicked.connect(self.view.zoom_in)
        self.btn_zoom_out_main.clicked.connect(self.view.zoom_out)
        self.btn_zoom_100_main.clicked.connect(self.view.zoom_100)

        self.scene.roi_created.connect(self._on_roi_created)
        self.scene.line_created.connect(self._on_line_created)
        self.scene.cursor_moved.connect(self._on_cursor_moved)

        self.shortcut_undo.activated.connect(self.undo)
        self.shortcut_delete.activated.connect(self.delete_selected_rois)
        self.shortcut_escape.activated.connect(self.cancel_drawing)

        self.slider_saturation.valueChanged.connect(self._on_enhancement_changed)
        self.slider_brightness.valueChanged.connect(self._on_enhancement_changed)
        self.slider_contrast.valueChanged.connect(self._on_enhancement_changed)
        self.slider_sharpness.valueChanged.connect(self._on_enhancement_changed)
        self.btn_enhance_reset.clicked.connect(self.reset_enhancement_settings)

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background-color: #1e1f22;
                color: #e6e6e6;
                font-size: 12px;
            }
            QPushButton {
                background-color: #2d3138;
                border: 1px solid #3c424d;
                border-radius: 6px;
                padding: 7px 10px;
            }
            QPushButton:hover {
                background-color: #383d47;
            }
            QPushButton:checked {
                background-color: #0f5a7a;
                border-color: #58c2ff;
            }
            QGraphicsView {
                border: 1px solid #3c424d;
                background-color: #121212;
            }
            QTableWidget {
                gridline-color: #3f4652;
                background-color: #1a1c20;
                alternate-background-color: #21252c;
            }
            QHeaderView::section {
                background-color: #2b2f37;
                color: #f0f0f0;
                padding: 6px;
                border: none;
            }
            QStatusBar {
                background: #1a1b1f;
            }
            """
        )

    def _bgr_to_pixmap(self, image_bgr) -> QPixmap:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def _collect_enhancement_from_sliders(self) -> dict:
        return slider_values_to_enhancement(
            int(self.slider_saturation.value()),
            int(self.slider_brightness.value()),
            int(self.slider_contrast.value()),
            int(self.slider_sharpness.value()),
        )

    def _update_enhancement_labels(self) -> None:
        params = self._collect_enhancement_from_sliders()
        self.lbl_saturation.setText(f"{params['saturation']:.2f}")
        self.lbl_brightness.setText(f"{params['brightness']:.0f}")
        self.lbl_contrast.setText(f"{params['contrast']:.2f}")
        self.lbl_sharpness.setText(f"{params['sharpness']:.2f}")

    def _refresh_image_preview_with_enhancement(self) -> None:
        if not self.image_path:
            return
        if not os.path.exists(self.image_path):
            return

        display_image, scale = backend.load_display_image(self.image_path)
        enhanced_display = backend.apply_image_enhancement(display_image, self.enhancement_params)
        self.display_scale = scale
        self.scene.set_image_pixmap(self._bgr_to_pixmap(enhanced_display))
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self._redraw_roi_items()
        self._redraw_nuclei_items()

    def _on_enhancement_changed(self) -> None:
        self.enhancement_params = self._collect_enhancement_from_sliders()
        self._update_enhancement_labels()
        self._refresh_image_preview_with_enhancement()

    def reset_enhancement_settings(self) -> None:
        defaults = backend.get_default_enhancement_params()
        sat, bri, con, sha = enhancement_to_slider_values(defaults)
        self.slider_saturation.setValue(sat)
        self.slider_brightness.setValue(bri)
        self.slider_contrast.setValue(con)
        self.slider_sharpness.setValue(sha)

    def _set_enhancement_params(self, params: dict, refresh: bool = True) -> None:
        self.enhancement_params = backend.normalize_enhancement_params(params)
        sat, bri, con, sha = enhancement_to_slider_values(self.enhancement_params)
        self.slider_saturation.blockSignals(True)
        self.slider_brightness.blockSignals(True)
        self.slider_contrast.blockSignals(True)
        self.slider_sharpness.blockSignals(True)
        self.slider_saturation.setValue(sat)
        self.slider_brightness.setValue(bri)
        self.slider_contrast.setValue(con)
        self.slider_sharpness.setValue(sha)
        self.slider_saturation.blockSignals(False)
        self.slider_brightness.blockSignals(False)
        self.slider_contrast.blockSignals(False)
        self.slider_sharpness.blockSignals(False)
        self._update_enhancement_labels()
        if refresh:
            self._refresh_image_preview_with_enhancement()

    def _push_undo_state(self) -> None:
        if self._restoring_state:
            return
        snapshot = {
            "image_path": self.image_path,
            "display_scale": self.display_scale,
            "pixels_per_mm": self.pixels_per_mm,
            "enhancement_params": copy.deepcopy(self.enhancement_params),
            "rois": copy.deepcopy(self.rois),
            "nuclei": copy.deepcopy(self.nuclei),
            "next_roi_id": self.next_roi_id,
        }
        self.undo_stack.append(snapshot)
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)

    def undo(self) -> None:
        if not self.undo_stack:
            return

        snapshot = self.undo_stack.pop()
        self._restoring_state = True
        try:
            self.image_path = snapshot["image_path"]
            self.display_scale = snapshot["display_scale"]
            self.pixels_per_mm = snapshot["pixels_per_mm"]
            self.enhancement_params = copy.deepcopy(
                snapshot.get("enhancement_params", backend.get_default_enhancement_params())
            )
            self.rois = copy.deepcopy(snapshot["rois"])
            self.nuclei = copy.deepcopy(snapshot["nuclei"])
            self.next_roi_id = snapshot["next_roi_id"]

            sat, bri, con, sha = enhancement_to_slider_values(self.enhancement_params)
            self.slider_saturation.blockSignals(True)
            self.slider_brightness.blockSignals(True)
            self.slider_contrast.blockSignals(True)
            self.slider_sharpness.blockSignals(True)
            self.slider_saturation.setValue(sat)
            self.slider_brightness.setValue(bri)
            self.slider_contrast.setValue(con)
            self.slider_sharpness.setValue(sha)
            self.slider_saturation.blockSignals(False)
            self.slider_brightness.blockSignals(False)
            self.slider_contrast.blockSignals(False)
            self.slider_sharpness.blockSignals(False)
            self._update_enhancement_labels()

            if self.image_path and os.path.exists(self.image_path):
                img, self.display_scale = backend.load_display_image(self.image_path)
                img_enhanced = backend.apply_image_enhancement(img, self.enhancement_params)
                self.scene.set_image_pixmap(self._bgr_to_pixmap(img_enhanced))
                self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            else:
                self.scene.set_image_pixmap(None)
                self.image_path = None

            self._redraw_roi_items()
            self._redraw_nuclei_items()
            self._refresh_table()
            self._update_scale_status()
            self.proc_label.setText("Статус: отменено последнее действие")
        finally:
            self._restoring_state = False

    def set_tool(self, mode: ToolMode) -> None:
        self.scene.set_tool_mode(mode)
        self.view.setCursor(Qt.CrossCursor)
        self.view.viewport().setCursor(Qt.CrossCursor)

    def cancel_drawing(self) -> None:
        self.scene.cancel_current_drawing()
        self.tool_group.setExclusive(False)
        self.btn_rect.setChecked(False)
        self.btn_poly.setChecked(False)
        self.btn_line.setChecked(False)
        self.tool_group.setExclusive(True)
        self.set_tool(ToolMode.NONE)

    def _ensure_scale_ready(self) -> bool:
        if self.pixels_per_mm is not None and self.pixels_per_mm > 0:
            return True
        self.activate_calibration()
        return self.pixels_per_mm is not None and self.pixels_per_mm > 0

    def _enforce_initial_scale(self) -> None:
        if self._initial_scale_completed:
            return
        ok = self._ensure_scale_ready()
        if not ok:
            QMessageBox.warning(
                self,
                "Требуется масштаб",
                "Без калибровки масштаба работа приложения невозможна.",
            )
            self.close()
            return
        self._initial_scale_completed = True

    def activate_calibration(self) -> None:
        candidate = "Фотография линейки.jpg"
        initial_path = None
        if os.path.exists(candidate):
            initial_path = candidate
        elif self.image_path and os.path.exists(self.image_path):
            initial_path = self.image_path

        dialog = ScaleCalibrationDialog(self, initial_path=initial_path)
        if dialog.exec_() != QDialog.Accepted:
            return
        if dialog.pixels_per_mm is None:
            return

        self._push_undo_state()
        self.pixels_per_mm = dialog.pixels_per_mm
        self._initial_scale_completed = True
        self._update_scale_status()
        self._refresh_table()
        self.proc_label.setText("Статус: масштаб задан")

    def run_cell_tuning_dialog(self) -> bool:
        if not self.image_path:
            QMessageBox.warning(self, "Нет изображения", "Сначала откройте изображение")
            return False
        return self._run_cell_tuning_for_image(
            self.image_path,
            enhancement_params=self.enhancement_params,
        )

    def _run_cell_tuning_for_image(
        self,
        image_path: str,
        enhancement_params: dict | None = None,
    ) -> bool:
        dialog = CellSelectionDialog(
            image_path,
            enhancement_params=enhancement_params if enhancement_params is not None else self.enhancement_params,
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted:
            return False

        if (
            dialog.selected_diameter_px is None
            or dialog.selected_radius_px is None
            or dialog.selected_center_xy is None
        ):
            QMessageBox.warning(
                self,
                "Нет данных",
                "Не удалось определить выделенную клетку",
            )
            return False

        try:
            source_image = backend.load_image(image_path)
            effective_enhancement = backend.normalize_enhancement_params(
                enhancement_params if enhancement_params is not None else self.enhancement_params
            )
            analysis_image = backend.apply_image_enhancement(source_image, effective_enhancement)
            params = backend.recommend_detection_params_from_selection(
                analysis_image,
                center=dialog.selected_center_xy,
                radius_px=float(dialog.selected_radius_px),
                preset=str(dialog.selected_preset),
            )
            backend.set_detection_params(params)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка подбора параметров", str(exc))
            return False

        self.cell_diameter_px = float(dialog.selected_diameter_px)
        self.cell_tune_preset = str(dialog.selected_preset)
        self._update_detector_status()
        self.proc_label.setText(
            f"Статус: параметры подобраны по клетке ({self.cell_tune_preset}, "
            f"диаметр {self.cell_diameter_px:.1f} px, цвет ядра учтён)"
        )
        return True

    def open_detection_params(self) -> None:
        dialog = DetectionParamsDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return
        self._update_detector_status()
        self.proc_label.setText("Статус: параметры детекции обновлены")

    def open_image(self) -> None:
        if not self._ensure_scale_ready():
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Открыть изображение",
            "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg)",
        )
        if not path:
            return

        try:
            display_image, scale = backend.load_display_image(path)
            enhanced_display = backend.apply_image_enhancement(display_image, self.enhancement_params)
            pixmap = self._bgr_to_pixmap(enhanced_display)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))
            return

        self._push_undo_state()

        self.image_path = path
        self.display_scale = scale
        self.cell_diameter_px = None
        self.cell_tune_preset = "точный"
        self.rois.clear()
        self.nuclei.clear()
        self.metrics_rows.clear()
        self.next_roi_id = 1

        self.scene.set_image_pixmap(pixmap)
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.table.setRowCount(0)

        self._update_scale_status()
        self.proc_label.setText("Статус: изображение загружено")

    def _list_images_in_folder(self, folder_path: str) -> list[Path]:
        folder = Path(folder_path)
        return [
            p for p in sorted(folder.iterdir())
            if p.is_file() and p.suffix.lower() in backend.SUPPORTED_IMAGE_FORMATS
        ]

    def _build_full_frame_roi(self, image_path: str) -> list[dict]:
        image = backend.load_image(image_path)
        h, w = image.shape[:2]
        points = [
            (0.0, 0.0),
            (float(w - 1), 0.0),
            (float(w - 1), float(h - 1)),
            (0.0, float(h - 1)),
        ]
        return [
            {
                "id": 1,
                "name": "Весь кадр",
                "type": "full_frame",
                "points": points,
            }
        ]

    def _collect_rois_for_batch(
        self,
        image_paths: list[Path],
        mode_key: str,
        enhancement_params: dict,
    ) -> dict[str, list[dict]] | None:
        rois_by_file: dict[str, list[dict]] = {}
        if mode_key == BATCH_MODE_FULL:
            for image_path in image_paths:
                rois_by_file[str(image_path)] = self._build_full_frame_roi(str(image_path))
            return rois_by_file

        for index, image_path in enumerate(image_paths, start=1):
            self.proc_label.setText(
                f"Статус: разметка ROI {index}/{len(image_paths)}"
            )
            dialog = RoiAnnotationDialog(
                image_path=str(image_path),
                mode_key=mode_key,
                enhancement_params=enhancement_params,
                name_suggestions=self.roi_name_history,
                parent=self,
            )
            if dialog.exec_() != QDialog.Accepted:
                return None

            rois_by_file[str(image_path)] = copy.deepcopy(dialog.rois)
            for roi_name in dialog.new_names:
                if roi_name not in self.roi_name_history:
                    self.roi_name_history.append(roi_name)
        return rois_by_file

    def run_batch_processing(self) -> None:
        if self._det_thread is not None:
            QMessageBox.information(
                self,
                "Пакетная обработка",
                "Сначала дождитесь завершения текущей детекции",
            )
            return
        if not self._ensure_scale_ready():
            return

        setup_dialog = BatchSetupDialog(self)
        if setup_dialog.exec_() != QDialog.Accepted:
            return
        if not setup_dialog.folder_path:
            return

        image_paths = self._list_images_in_folder(setup_dialog.folder_path)
        if not image_paths:
            QMessageBox.warning(
                self,
                "Нет изображений",
                "В указанной папке нет подходящих изображений",
            )
            return

        color_dialog = ColorTuningDialog(
            image_path=str(image_paths[0]),
            initial_params=self.enhancement_params,
            parent=self,
        )
        if color_dialog.exec_() != QDialog.Accepted or color_dialog.result_params is None:
            return
        batch_enhancement = backend.normalize_enhancement_params(color_dialog.result_params)
        self._set_enhancement_params(batch_enhancement, refresh=True)

        if backend.get_loaded_model_info() is None:
            default_info = backend.get_default_detector_info()
            if default_info.get("type") == "unavailable":
                QMessageBox.warning(
                    self,
                    "Нейросеть недоступна",
                    "Предобученная модель недоступна. "
                    "Установите необходимые зависимости или загрузите .pt/.onnx модель.",
                )
                return
            if self.cell_diameter_px is None:
                ok = self._run_cell_tuning_for_image(
                    str(image_paths[0]),
                    enhancement_params=batch_enhancement,
                )
                if not ok:
                    self.proc_label.setText("Статус: пакетная обработка отменена")
                    return

        rois_by_file = self._collect_rois_for_batch(
            image_paths=image_paths,
            mode_key=setup_dialog.mode_key,
            enhancement_params=batch_enhancement,
        )
        if rois_by_file is None:
            self.proc_label.setText("Статус: пакетная обработка отменена")
            return
        if self._batch_thread is not None:
            QMessageBox.information(
                self,
                "Пакетная обработка",
                "Пакетная обработка уже выполняется",
            )
            return

        self._pending_batch_folder_name = Path(setup_dialog.folder_path).name
        self._pending_batch_total_files = len(image_paths)

        self._batch_progress = QProgressDialog(
            "Выполняется пакетная детекция...",
            "Отмена",
            0,
            len(image_paths),
            self,
        )
        self._batch_progress.setWindowTitle("Пакетная обработка")
        self._batch_progress.setMinimumDuration(0)
        self._batch_progress.setValue(0)
        self._batch_progress.setAutoClose(False)
        self._batch_progress.setAutoReset(False)

        self._batch_thread = QThread(self)
        detection_params = backend.get_detection_params()
        model_info = backend.get_loaded_model_info()
        custom_model_path = str(model_info.get("path")) if model_info else None
        self._batch_worker = BatchDetectionWorker(
            image_paths=[str(p) for p in image_paths],
            rois_by_file=rois_by_file,
            pixels_per_mm=float(self.pixels_per_mm),
            enhancement_params=batch_enhancement,
            detection_params=detection_params,
            custom_model_path=custom_model_path,
        )
        self._batch_worker.moveToThread(self._batch_thread)

        self._batch_thread.started.connect(self._batch_worker.run)
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.finished.connect(self._on_batch_finished)
        self._batch_worker.failed.connect(self._on_batch_failed)
        self._batch_worker.canceled.connect(self._on_batch_canceled)
        self._batch_progress.canceled.connect(self._batch_worker.request_cancel)

        self._batch_worker.finished.connect(self._batch_thread.quit)
        self._batch_worker.failed.connect(self._batch_thread.quit)
        self._batch_worker.canceled.connect(self._batch_thread.quit)
        self._batch_thread.finished.connect(self._cleanup_batch_thread)

        self.btn_batch.setEnabled(False)
        self.btn_detect.setEnabled(False)
        self.proc_label.setText("Статус: запущена пакетная детекция")
        self._batch_thread.start()

    def _on_batch_progress(self, current: int, total: int, file_name: str) -> None:
        if self._batch_progress is not None:
            self._batch_progress.setMaximum(total)
            self._batch_progress.setValue(current)
            self._batch_progress.setLabelText(
                f"Выполняется пакетная детекция...\n{current}/{total}: {file_name}"
            )
        if total > 0:
            self.proc_label.setText(f"Статус: детекция {current}/{total} ({file_name})")

    def _on_batch_finished(self, batch_rows: list[dict]) -> None:
        if not batch_rows:
            QMessageBox.warning(
                self,
                "Нет результатов",
                "После обработки не получено данных для экспорта",
            )
            self.proc_label.setText("Статус: пакетная обработка завершена без результатов")
            return

        default_name = f"пакет_{self._pending_batch_folder_name}.csv"
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить пакетный отчёт",
            default_name,
            "CSV (*.csv);;Excel (*.xlsx)",
        )
        if not output_path:
            self.proc_label.setText("Статус: экспорт пакетного отчёта отменён")
            return

        try:
            backend.export_batch_results(batch_rows, output_path)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка экспорта", str(exc))
            self.proc_label.setText("Статус: ошибка экспорта пакетного отчёта")
            return

        self.proc_label.setText(
            f"Статус: пакетная обработка завершена ({self._pending_batch_total_files} файлов)"
        )

    def _on_batch_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Ошибка пакетной детекции", message)
        self.proc_label.setText("Статус: ошибка пакетной детекции")

    def _on_batch_canceled(self) -> None:
        self.proc_label.setText("Статус: пакетная обработка отменена")

    def _cleanup_batch_thread(self) -> None:
        if self._batch_progress is not None:
            self._batch_progress.close()
            self._batch_progress.deleteLater()
            self._batch_progress = None

        if self._batch_worker is not None:
            self._batch_worker.deleteLater()
            self._batch_worker = None

        if self._batch_thread is not None:
            self._batch_thread.deleteLater()
            self._batch_thread = None

        self.btn_batch.setEnabled(True)
        self.btn_detect.setEnabled(True)

    def _on_cursor_moved(self, x: float, y: float) -> None:
        ox = x * self.display_scale
        oy = y * self.display_scale
        self.coord_label.setText(f"Курсор: x {ox:.1f}, y {oy:.1f}")

    def _on_roi_created(self, payload: dict) -> None:
        if not self.image_path:
            return

        points_display = payload.get("points", [])
        roi_type = payload.get("type", "polygon")
        if len(points_display) < 3:
            return

        self._push_undo_state()

        points_orig = [
            (float(x) * self.display_scale, float(y) * self.display_scale)
            for x, y in points_display
        ]
        roi = {
            "id": self.next_roi_id,
            "type": roi_type,
            "points": points_orig,
        }

        self.rois.append(roi)
        self.scene.add_roi_item(self.next_roi_id, roi_type, points_display)
        self.next_roi_id += 1

        self._refresh_table()
        self.proc_label.setText("Статус: область добавлена")

    def _on_line_created(self, line: tuple) -> None:
        if not self.image_path:
            return

        real_mm, ok = QInputDialog.getDouble(
            self,
            "Калибровка",
            "Реальная длина линии (мм):",
            1.0,
            0.0001,
            1_000_000.0,
            4,
        )
        if not ok:
            return

        (x1, y1), (x2, y2) = line
        line_orig = (
            (x1 * self.display_scale, y1 * self.display_scale),
            (x2 * self.display_scale, y2 * self.display_scale),
        )

        try:
            image = backend.load_image(self.image_path)
            ppm = backend.calibrate_scale(image, line_orig, real_mm)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка калибровки", str(exc))
            return

        self._push_undo_state()
        self.pixels_per_mm = ppm
        self._update_scale_status()
        self._refresh_table()
        self.proc_label.setText("Статус: масштаб задан")

    def detect_nuclei(self) -> None:
        if not self.image_path:
            QMessageBox.warning(self, "Нет изображения", "Сначала откройте изображение")
            return
        if not self._ensure_scale_ready():
            return
        if self._batch_thread is not None:
            QMessageBox.information(
                self,
                "Детекция",
                "Пакетная обработка уже запущена. Дождитесь её завершения.",
            )
            return

        if backend.get_loaded_model_info() is None:
            default_info = backend.get_default_detector_info()
            if default_info.get("type") == "unavailable":
                detector_name = str(default_info.get("name", "встроенная модель"))
                QMessageBox.warning(
                    self,
                    "Нейросеть недоступна",
                    f"{detector_name} недоступна. "
                    "Установите зависимости для выбранной встроенной модели "
                    "или загрузите .pt/.onnx модель.",
                )
                return
            if self.cell_diameter_px is None:
                ok = self.run_cell_tuning_dialog()
                if not ok:
                    self.proc_label.setText("Статус: детекция отменена")
                    return

        if self._det_thread is not None:
            return

        self.btn_detect.setEnabled(False)
        self.proc_label.setText("Статус: выполняется детекция ядер...")

        self._det_thread = QThread(self)
        detection_params = backend.get_detection_params()
        model_info = backend.get_loaded_model_info()
        custom_model_path = str(model_info.get("path")) if model_info else None
        self._det_worker = DetectionWorker(
            self.image_path,
            self.enhancement_params,
            detection_params=detection_params,
            custom_model_path=custom_model_path,
        )
        self._det_worker.moveToThread(self._det_thread)

        self._det_thread.started.connect(self._det_worker.run)
        self._det_worker.finished.connect(self._on_detect_finished)
        self._det_worker.failed.connect(self._on_detect_failed)

        self._det_worker.finished.connect(self._det_thread.quit)
        self._det_worker.failed.connect(self._det_thread.quit)
        self._det_thread.finished.connect(self._cleanup_detection_thread)

        self._det_thread.start()

    def _cleanup_detection_thread(self) -> None:
        if self._det_worker is not None:
            self._det_worker.deleteLater()
            self._det_worker = None
        if self._det_thread is not None:
            self._det_thread.deleteLater()
            self._det_thread = None
        self.btn_detect.setEnabled(True)

    def _on_detect_finished(self, nuclei: list[dict]) -> None:
        self._push_undo_state()
        self.nuclei = nuclei
        self._redraw_nuclei_items()
        self._refresh_table()
        self.proc_label.setText(f"Статус: найдено ядер: {len(nuclei)}")

    def _on_detect_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Ошибка детекции", message)
        self.proc_label.setText("Статус: ошибка детекции")

    def _redraw_nuclei_items(self) -> None:
        if not self.nuclei:
            self.scene.clear_nuclei_items()
            return

        nuclei_display: list[dict] = []
        for nuc in self.nuclei:
            center = nuc.get("center")
            contour = nuc.get("contour", [])
            if center is None:
                continue

            nuclei_display.append(
                {
                    "center": (
                        float(center[0]) / self.display_scale,
                        float(center[1]) / self.display_scale,
                    ),
                    "contour": [
                        (float(x) / self.display_scale, float(y) / self.display_scale)
                        for x, y in contour
                    ],
                }
            )

        self.scene.set_nuclei_items(nuclei_display)

    def _redraw_roi_items(self) -> None:
        self.scene.clear_roi_items()
        for roi in self.rois:
            points_disp = [
                (float(x) / self.display_scale, float(y) / self.display_scale)
                for x, y in roi.get("points", [])
            ]
            self.scene.add_roi_item(int(roi["id"]), str(roi.get("type", "polygon")), points_disp)

    def _update_scale_status(self) -> None:
        if self.pixels_per_mm is None:
            self.scale_label.setText("Масштаб: не задан")
        else:
            self.scale_label.setText(f"Масштаб: {self.pixels_per_mm:.4f} px/mm")

    def _refresh_table(self) -> None:
        self.metrics_rows = []
        for roi in self.rois:
            row = backend.build_roi_metrics(roi, self.nuclei, self.pixels_per_mm)
            self.metrics_rows.append(row)

        self.table.setRowCount(len(self.metrics_rows))

        for r, row in enumerate(self.metrics_rows):
            values = [
                row.get("ROI ID"),
                row.get("Тип"),
                row.get("Площадь (мм²)"),
                row.get("Количество ядер"),
                row.get("Плотность (ядра/мм²)"),
            ]

            for c, value in enumerate(values):
                if value is None:
                    text = "-"
                elif isinstance(value, float):
                    text = f"{value:.4f}"
                else:
                    text = str(value)
                self.table.setItem(r, c, QTableWidgetItem(text))

        if self.rois and self.pixels_per_mm is None:
            self.statusBar().showMessage(
                "Калибровка не задана: площадь и плотность в мм² не рассчитаны", 4000
            )

    def export_results(self) -> None:
        if not self.metrics_rows:
            QMessageBox.warning(self, "Нет данных", "Нет ROI для экспорта")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт результатов",
            "results.csv",
            "CSV (*.csv);;Excel (*.xlsx)",
        )
        if not path:
            return

        try:
            backend.export_results(self.metrics_rows, path)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка экспорта", str(exc))
            return

        self.proc_label.setText("Статус: экспорт завершён")

    def delete_selected_rois(self) -> None:
        selected_ids = self.scene.selected_roi_ids()
        if not selected_ids:
            return

        self._push_undo_state()
        selected_set = set(selected_ids)
        self.rois = [roi for roi in self.rois if int(roi["id"]) not in selected_set]
        for roi_id in selected_set:
            self.scene.remove_roi_item(roi_id)

        self._refresh_table()
        self.proc_label.setText("Статус: область удалена")

    def load_custom_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Загрузить модель",
            "",
            "Models (*.pt *.onnx)",
        )
        if not path:
            return

        try:
            backend.load_custom_model(path)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка модели", str(exc))
            return

        self._update_detector_status()
        self.proc_label.setText("Статус: пользовательская модель загружена")

    def reset_custom_model(self) -> None:
        backend.reset_custom_model()
        self._update_detector_status()
        self.proc_label.setText("Статус: используется предобученная модель")

    def _update_detector_status(self) -> None:
        model_info = backend.get_loaded_model_info()
        if model_info is not None:
            file_name = Path(model_info["path"]).name
            self.detector_label.setText(
                f"Детектор: {file_name} ({model_info['runtime']})"
            )
            return

        default_info = backend.get_default_detector_info()
        params = backend.get_detection_params()
        backend_name = str(params.get("detector_backend", "stardist")).strip().lower()
        if backend_name == "cellpose_nuclei":
            self.detector_label.setText(
                f"Детектор: {default_info['name']} | flow {float(params.get('cellpose_flow_threshold', 0.4)):.2f} "
                f"| mask {float(params.get('cellpose_cellprob_threshold', -0.5)):.2f} "
                f"| диам. {float(params.get('cellpose_diameter_px', 14.0)):.1f}px"
            )
            return

        mode_ru = preprocess_mode_to_russian(str(params.get("preprocess_mode", "")))
        self.detector_label.setText(
            f"Детектор: {default_info['name']} | увер. {params['prob_thresh']:.2f} "
            f"| раздел. {params['nms_thresh']:.2f} | режим: {mode_ru}"
        )


def create_app() -> tuple[QApplication, MainWindow]:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    window = MainWindow()
    return app, window
