import os
import cv2
import numpy as np
from src.main.python.inspector.pr.unmolded_inspector import UnmoldedInspector
from evaluator.eval_nut_inspector import NutInspectVerification, calc_eval_metrics


class UnmoldedInspectVerification(NutInspectVerification):
    def __init__(self, inspector, dir_path, label):
        super(UnmoldedInspectVerification, self).__init__(inspector, dir_path, label)

    def run(self):
        tp, fp, tn, fn = 0, 0, 0, 0
        img_paths = [os.path.join(self.dir_path, path) for path in os.listdir(self.dir_path) if path.endswith('.jpg')]
        for path in img_paths:
            img = cv2.imread(path)
            ret, is_normal = self.inspector.inspect(img)

            self.num_of_item += 1
            if ret:
                tp, fp, tn, fn = self.count(is_normal, tp, fp, tn, fn)

                if (is_normal is True) and (self.label is False):
                    self.fp_img_paths.append(path)
                elif (is_normal is False) and (self.label is True):
                    self.fn_img_paths.append(path)

            else:
                self.disable_img_paths.append(path)

        return tp, fp, tn, fn


if __name__ == '__main__':
    import copy
    # normal
    normal_path = "/Users/rangkim/projects/datasets/ess_case_insert_mold/ESS_20210107/normal_mold"
    # abnormal
    unmolded = "/Users/rangkim/projects/datasets/ess_case_insert_mold/ESS_20210107/unmold"

    dir_paths = [normal_path, unmolded]
    labels = [True, False]
    thr1 = 0.4
    inspector = UnmoldedInspector(thr1, normal_path=normal_path)
    inspectors = [inspector, inspector]
    total_metrics = 0, 0, 0, 0
    fp_paths, fn_paths, disable_img_paths = [], [], []

    print('\nclass        tot tp fp tn fn   mean  std  median')
    print('=================================================')
    for inspector, dir_path, label in zip(inspectors, dir_paths, labels):
        verification = UnmoldedInspectVerification(inspector, dir_path, label)
        metrics = verification.run()

        fp_paths += verification.fp_img_paths
        fn_paths += verification.fn_img_paths
        disable_img_paths += verification.disable_img_paths
        total_metrics = tuple(map(sum, zip(total_metrics, metrics)))
        tot = sum(metrics)
        class_name = dir_path.split('/')[-2]
        print(f"{class_name:10} {tot:4} {metrics}")

    pre, rec, acc = calc_eval_metrics(*total_metrics)
    print(f'\n{pre:.2f}, {rec:.2f}, {acc:.2f}\n')
    print(f'{fp_paths}')
    print(f'{fn_paths}')
    print(f'{disable_img_paths}')
