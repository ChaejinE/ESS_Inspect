import os
import cv2
import numpy as np
from typing import List
from src.main.python.inspector.inspector import Inspector
from typing import Tuple


class UnmoldedInspector(Inspector):
    def __init__(self, avg_hist_thr: float, resize_h_w: Tuple[int, int] = (256, 256),
                 normal_path: str = None, image_ext: str = "bmp"):
        super(UnmoldedInspector, self).__init__()
        self._resize_h_w = resize_h_w[::-1]
        self._normal_path = normal_path
        self._image_ext = image_ext
        self._normal_image_file_names = filter(lambda x: x.endswith(self._image_ext), os.listdir(self._normal_path))
        self._normal_images = self.load_normal_images()
        self._normal_images = self.preprocess_normal_images(self._normal_images)

        self._normal_hists = list(map(self.calc_hist, self._normal_images))
        self._abnormal_hist = None
        self._avg_hist = avg_hist_thr

    def preprocess_normal_images(self, normal_images):
        preprocessed_normal_images = [cv2.GaussianBlur(image, (5, 5), 0) for image in normal_images]

        return preprocessed_normal_images

    @staticmethod
    def calc_hist(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        return cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

    def load_normal_images(self) -> List[np.array]:
        normal_images = []
        for normal_file_name in self._normal_image_file_names:
            normal_image = cv2.imread(os.path.join(self._normal_path, normal_file_name))
            normal_image = cv2.resize(normal_image, self._resize_h_w)
            normal_images.append(normal_image)

        return normal_images

    def preprocess(self, image) -> None:
        resized_target_image = cv2.resize(image, self._resize_h_w)
        resized_target_image = cv2.GaussianBlur(resized_target_image, (5, 5), 0)
        self._abnormal_hist = self.calc_hist(resized_target_image)

    @staticmethod
    def analyze(image):
        import matplotlib.pyplot as plt
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        fig = plt.figure()
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        ax1 = fig.add_subplot(121)
        ax1.imshow(cv2.resize(image, (100, 100)))
        ax1.set_title('Origin Image')
        ax1.axis('off')

        # fig2 = plt.figure()
        ax2 = fig.add_subplot(122)
        ax2.set_ylabel('Hue')
        ax2.set_xlabel('Saturation')
        p = ax2.imshow(hist)
        ax2.set_title('Hue and Saturation')
        plt.colorbar(p)

        plt.show()

    def inspect(self, frame) -> Tuple[bool, bool]:
        self.preprocess(image=frame)
        avg_hist = 0
        is_normal = None
        ret = True
        try:
            print("====================")
            for idx, hist in enumerate(self._normal_hists):
                ret = cv2.compareHist(self._abnormal_hist, hist, cv2.HISTCMP_CORREL)
                avg_hist += ret
                print(f"ret: {ret}")

            avg_hist /= len(self._normal_hists)
            is_normal = True if avg_hist > self._avg_hist else False
            print(f"avg : {avg_hist}")
            print("====================")

        except Exception as e:
            print(e)
            ret = False

        return ret, is_normal
