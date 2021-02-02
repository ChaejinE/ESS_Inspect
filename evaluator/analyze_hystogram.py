import os
import cv2
from src.main.python.inspector.pr.unmolded_inspector import UnmoldedInspector
from src.main.python.utils.roi_selector import RoiSelector

normal_path = "/Users/jeongchaejin/Projects/ess_unmolded/src/main/python/utils/unmolded_roi"
abnormal_path = "/Users/jeongchaejin/Desktop/JCJ/Project/2021/ESS/ESS_20210108/미성형"

normal_images = [cv2.imread(os.path.join(normal_path, file_name)) for file_name
                 in os.listdir(normal_path) if file_name.endswith("bmp")]

abnormal_images = [cv2.imread(os.path.join(abnormal_path, file_name)) for file_name
                   in os.listdir(abnormal_path) if file_name.endswith("jpg")]
temp = []
for image in abnormal_images:
    selector = RoiSelector(image, is_mouse=False)
    selector.operate([(34, 105, 92, 126)])
    temp.append(selector.get_roi_images()[0])

normal_images, abnormal_images = list(map(lambda x: cv2.resize(x, (256, 256)), normal_images)),\
                                 list(map(lambda x: cv2.resize(x, (256, 256)), temp))

inspector = UnmoldedInspector(0.7, normal_path=normal_path)

for image in (normal_images + abnormal_images):
    inspector.analyze(image)
# inspector.analyze(normal_images[-1])
