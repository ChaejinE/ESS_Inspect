import cv2
import time
import os
from typing import Tuple, Any, List
import numpy as np


class RoiSelector:
    def __init__(self, image: Any, resize: Tuple[int, int] = (704, 256),
                 is_save: bool = False, is_mouse: bool = True) -> None:
        self._image: np.array = cv2.resize(image, tuple(resize[::-1])) if resize else image
        self._is_dragging: bool = False
        self._is_save = is_save
        self._using_mouse = is_mouse
        self._x, self._y, self._w, self._h = -1, -1, -1, -1
        self._roi_coordinates: list = []

    @property
    def get_origin_image(self):
        return self._image

    @property
    def get_roi_width(self) -> int or None:
        if self._w >= 0:
            return self._w
        return None

    @property
    def get_roi_height(self) -> int or None:
        if self._h >= 0:
            return self._h
        return None

    @property
    def get_roi_coordinates(self) -> Tuple[int] or None:
        return self._roi_coordinates

    def get_roi_image(self, y_x_h_w: Tuple[int, int, int, int] = None) -> np.array:
        y, x, h, w = y_x_h_w if y_x_h_w else (self._y, self._x, self._h, self._w)
        _roi_image = self._image[y:y+h, x:x+w]

        if self._is_save:
            save_dir = "nut_roi"
            os.makedirs(save_dir, exist_ok=True) if not os.path.exists(save_dir) else None
            cv2.imwrite(os.path.join(save_dir, f"{time.time() * 1000}_roi_image.jpg"), _roi_image)

        return _roi_image

    def get_roi_images(self):
        roi_images = []
        for roi_coord in self._roi_coordinates:
            y0, x0, y1, x1 = roi_coord
            roi_images.append(self.get_roi_image((y0, x0, y1-y0, x1-x0)))

        return roi_images

    def draw_fixed_roi(self, coordinates: List[Tuple[int, int, int, int]],
                       color: Tuple[int, int, int] = (255, 255, 0), is_show: bool = False) -> None:
        self._roi_coordinates = list(coordinates)
        drawn_image = self._image.copy()
        for idx, coord in enumerate(coordinates):
            y0, x0, y1, x1 = coord
            cv2.rectangle(drawn_image, (x0, y0), (x1, y1), color, 1)
            roi_image = self.get_roi_image((y0, x0, y1-y0, x1-x0))

            if is_show:
                cv2.imshow('img', self._image)
                cv2.moveWindow("img", 150, 150)
                cv2.imshow(f"cropped_{int(time.time())}", roi_image)
                cv2.imshow('img', drawn_image)
                cv2.waitKey(0)

    def draw_roi_with_mouse(self, event, x: int, y: int, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._is_dragging = True
            self._x, self._y = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._is_dragging:
                drawn_image = self._image.copy()
                cv2.rectangle(drawn_image, (self._x, self._y), (x, y), (0, 0, 255), 1)
                cv2.imshow('img', drawn_image)

        elif event == cv2.EVENT_LBUTTONUP:
            if self._is_dragging:
                self._is_dragging = False
                self._w = x - self._x
                self._h = y - self._y
                if self._w > 0 and self._h > 0:
                    self._roi_coordinates.append((self._y, self._x, self._y+self._h, self._x+self._w))
                    drawn_image = self._image.copy()
                    cv2.rectangle(drawn_image, (self._x, self._y), (x, y), (0, 0, 255), 1)
                    cv2.imshow('img', drawn_image)
                    img_roi = self.get_roi_image()
                    cv2.imshow('cropped', img_roi)
                    cv2.moveWindow('cropped', 0, 0)
                else:
                    print("좌측 상단에서 우측 하단으로 영역을 드래그 하세요.")

    def operate(self, coordinates: List[Tuple[int, int, int, int]] = None) -> None:
        if self._using_mouse:
            cv2.imshow('img', self._image)
            cv2.moveWindow("img", 150, 150)
            cv2.setMouseCallback('img', self.draw_roi_with_mouse)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            if not coordinates:
                raise ValueError("coordinates 인자 값을 설정해주세요. ex) [(y0_1, x0_1, y1_1, x1_1), (y0_2, x0_2, y1_2, x1_2)]")
            self.draw_fixed_roi(coordinates=coordinates)
