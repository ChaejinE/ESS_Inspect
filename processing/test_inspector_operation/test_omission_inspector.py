import os
import cv2
import unittest
from src.main.python.inspector.pr.nut_omission_inspector import NutOmissionInspector
from src.main.python.utils.roi_selector import RoiSelector


class TestOmissionOperation(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_op(self):
        # Warning :  ESS 전체 이미지의 path, crop image path 아님
        target_dir = "/Users/jeongchaejin/Desktop/JCJ/Project/2021/ESS/ESS_20210108/양품"

        normal, defect = 0, 0
        for file_name in os.listdir(target_dir):
            if not file_name.endswith("jpg") and not file_name.endswith("bmp"):
                break
            img = cv2.imread(os.path.join(target_dir, file_name))
            selector = RoiSelector(img, is_save=False, is_mouse=False)
            test_image = selector.get_origin_image.copy()
            for roi_coord in [(132, 170, 250, 235), (533, 180, 671, 237)]:
                y0, x0, y1, x1 = roi_coord
                roi_image = selector.get_roi_image((y0, x0, y1-y0, x1-x0))
                inspector = NutOmissionInspector()
                ret, is_defect, _, _ = inspector.inspect(roi_image)
                color = (0, 255, 0) if is_defect else (0, 0, 255)
                if is_defect:
                    defect += 1
                else:
                    normal += 1
                cv2.rectangle(test_image, (x0, y0), (x1, y1), color, 2)
                cv2.putText(test_image, f"Omission:{str(is_defect).upper()}", (x0-25, y0-5), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.3, color=color, thickness=1, lineType=cv2.LINE_AA)

            cv2.imshow("Omission", test_image)
            cv2.moveWindow("Omission", 150, 150)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print(f"defect : {defect}, normal : {normal}")