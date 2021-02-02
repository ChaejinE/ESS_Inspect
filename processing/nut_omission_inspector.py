from src.main.python.inspector.inspector import Inspector
from typing import Tuple, Any
import cv2
import time


class NutOmissionInspector(Inspector):
    def __init__(self, bin_thr: int = 120, area_thr: int = 750, resize_h_w: Tuple[int, int] = (70, 70),
                 kernel_size: Tuple[int, int] = (5, 5), is_show: bool = False):
        super(NutOmissionInspector, self).__init__()
        self._resize_h_w = resize_h_w[::-1]
        self._binary_threshold = bin_thr
        self._area_threshold = area_thr
        self._kernel_size = kernel_size
        self._is_show = is_show
        self._area = 0
        self._preprocess_response_time = 0
        self._inspect_response_time = 0

    def preprocess(self, image) -> Tuple[bool, Any]:
        start = time.time()
        resized_image = cv2.resize(image, self._resize_h_w)
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        ret, thr = cv2.threshold(gray, thresh=self._binary_threshold, maxval=255, type=cv2.THRESH_BINARY)
        morph = cv2.morphologyEx(thr, cv2.MORPH_CLOSE,
                                 kernel=cv2.getStructuringElement(cv2.MORPH_RECT, self._kernel_size)) if ret else None

        cv2.imshow("gray", gray)
        cv2.imshow("threshold", thr)
        cv2.imshow("morphology", morph)
        self._preprocess_response_time = time.time() - start

        return ret, morph

    def inspect(self, frame) -> Tuple[bool, bool, None, None]:
        start = time.time()

        ret, result_image = self.preprocess(image=frame)
        if ret or result_image is not None:
            contours, _ = cv2.findContours(result_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            temp = cv2.resize(frame, self._resize_h_w)
            cv2.drawContours(temp, contours, -1, (0, 255, 0), 3)
            cv2.imshow('frame', temp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            areas = list(map(cv2.contourArea, contours))
            print(areas)
            areas_len = len(areas)
            circle_area = [areas.pop(areas.index(max(areas))) for _ in range(2)] if areas_len >= 2 else []
            self._area = circle_area[0] - circle_area[1] if len(circle_area) == 2 else 0

            is_normal = self._area >= self._area_threshold
        else:
            is_normal = None

        self._inspect_response_time = time.time() - start

        return ret, is_normal, None, None
