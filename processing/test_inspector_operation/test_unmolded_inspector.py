import cv2
import os
import unittest
from src.main.python.inspector.pr.unmolded_inspector import UnmoldedInspector
from src.main.python.utils.roi_selector import RoiSelector


class TestUnmoldedOperation(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_op(self):
        # Warning : ESS 정상 부분 crop image path
        normal_path = "/Users/jeongchaejin/Projects/ess_unmolded/src/main/python/utils/unmolded_roi"
        # Warning : ESS 전체 이미지의 path, crop image path 아님
        target_dir = "/Users/jeongchaejin/Desktop/JCJ/Project/2021/ESS/ESS_20210108/미성형"

        normal, defect = 0, 0
        inspector = UnmoldedInspector(avg_hist_thr=0.7, normal_path=normal_path)
        for file_name in os.listdir(target_dir):
            if not file_name.endswith("bmp") and not file_name.endswith("jpg"):ㅁㅁ
                break
            img = cv2.imread(os.path.join(target_dir, file_name))
            selector = RoiSelector(img, (704, 256), is_save=False, is_mouse=False)
            test_image = selector.get_origin_image.copy()
            for roi_coord in [(34, 105, 92, 126)]:
                y0, x0, y1, x1 = roi_coord
                roi_image = selector.get_roi_image((y0, x0, y1-y0, x1-x0))
                ret, is_normal = inspector.inspect(roi_image)

                color =  (0, 255, 0) if is_normal else (0, 0, 255)
                if is_normal:
                    normal += 1
                else:
                    defect += 1
                cv2.rectangle(test_image, (x0, y0), (x1, y1), color, 2)
                cv2.putText(test_image, f"Unmolded:{str(is_normal).upper()}", (x0-25, y0-5), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.3, color=color, thickness=1, lineType=cv2.LINE_AA)

            cv2.imshow("Unmolded", test_image)
            cv2.moveWindow("Unmolded", 150, 150)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
