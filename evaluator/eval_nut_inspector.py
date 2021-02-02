import os
import cv2
import numpy as np
from src.main.python.inspector.pr.nut_protrusion_inspector import NutProtrusionInspector
from src.main.python.inspector.pr.nut_inverse_inspector import NutInverseInspector
from src.main.python.inspector.pr.nut_omission_inspector import NutOmissionInspector


class NutInspectVerification:
    def __init__(self, inspector, dir_path, label):
        self.inspector = inspector
        self.dir_path = dir_path
        self.label = label
        self.fp_img_paths = []
        self.fn_img_paths = []
        self.disable_img_paths = []
        self.dists = []
        self.num_of_item = 0

    def run(self):
        tp, fp, tn, fn = 0, 0, 0, 0
        img_paths = [os.path.join(self.dir_path, path) for path in os.listdir(self.dir_path) if path.endswith('.jpg')]
        for path in img_paths:
            img = cv2.imread(path)
            ret, is_normal, p1, p2 = self.inspector.inspect(img)
            self.num_of_item += 1
            if ret:
                dist = np.round(np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2), 2) if p1 is not None else None
                self.dists.append(dist) if p1 is not None else None
                tp, fp, tn, fn = self.count(is_normal, tp, fp, tn, fn)

                if (is_normal is True) and (self.label is False):
                    self.fp_img_paths.append(path)
                elif (is_normal is False) and (self.label is True):
                    self.fn_img_paths.append(path)

            else:
                self.disable_img_paths.append(path)

        return tp, fp, tn, fn

    def calc_statistics(self):
        if self.dists:
            dists = np.array(self.dists)
            mean = dists.mean()
            std = dists.std()
            dists.sort()
            median = dists[int(len(dists)/2)]
        else:
            mean, std, median = 0, 0, 0

        return mean, std, median

    def count(self, is_normal, tp, fp, tn, fn):
        if (is_normal is True) and (self.label is True):
            tp += 1
        elif (is_normal is True) and (self.label is False):
            fp += 1
        elif (is_normal is False) and (self.label is False):
            tn += 1
        elif (is_normal is False) and (self.label is True):
            fn += 1

        return tp, fp, tn, fn


def calc_eval_metrics(tp, fp, tn, fn):
    try:
        pre = tp / (tp + fp) * 100
        rec = tp / (tp + fn) * 100
        acc = (tp+tn) / (tp+fp+tn+fn) * 100
    except Exception() as e:
        pre, rec, acc = 0, 0, 0
        print(e)

    return pre, rec, acc


def display(img, is_normal, point1, point2):
    cv2.circle(img, (int(point1[0]), int(point1[1])), 1, (255, 0, 255), 1, cv2.LINE_AA)
    cv2.circle(img, (int(point2[0]), int(point2[1])), 1, (255, 255, 0), 1, cv2.LINE_AA)

    dist = np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
    print(f"{is_normal}, {dist:.2f}")

    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyWindow('img')


if __name__ == '__main__':
    # normal
    normal_path = "/Users/rangkim/projects/datasets/ess_case_insert_mold/ESS_20210107/normal_mold"
    # abnormal
    omission = "/Users/rangkim/projects/datasets/ess_case_insert_mold/ESS_20210107/miss_insert"
    protrusion1 = "/Users/rangkim/projects/datasets/ess_case_insert_mold/ESS_20210107/protrusion"
    protrusion2 = "/Users/rangkim/projects/datasets/ess_case_insert_mold/ESS_20210107/protrusion"
    inverse = "/Users/rangkim/projects/datasets/ess_case_insert_mold/ESS_20210107/reverse_insert"

    dir_paths = [normal_path, omission]
    labels = [True, False]
    thr1 = 31.5
    thr2 = 35.5
    thr3 = 30.
    thr4 = 120
    thr5 = 750
    inspector1 = NutProtrusionInspector(thr1)
    inspector2 = NutInverseInspector(thr2, thr3)
    inspector3 = NutOmissionInspector(thr4, thr5)
    inspectors = [inspector3, inspector3]
    total_metrics = 0, 0, 0, 0
    fp_paths, fn_paths, disable_img_paths = [], [], []

    print('\nclass        tot tp fp tn fn   mean  std  median')
    print('=================================================')
    for inspector, dir_path, label in zip(inspectors, dir_paths, labels):
        verification = NutInspectVerification(inspector, dir_path, label)
        metrics = verification.run()
        mean, std, median = verification.calc_statistics()

        fp_paths += verification.fp_img_paths
        fn_paths += verification.fn_img_paths
        disable_img_paths += verification.disable_img_paths
        total_metrics = tuple(map(sum, zip(total_metrics, metrics)))
        tot = sum(metrics)
        class_name = dir_path.split('/')[-2]
        print(f"{class_name:10} {tot:4} {metrics} {mean:5.2f} {std:5.2f} {median:5.2f}")

    pre, rec, acc = calc_eval_metrics(*total_metrics)
    print(f'\n{pre:.2f}, {rec:.2f}, {acc:.2f}\n')
    print(f'{fp_paths}')
    print(f'{fn_paths}')
    print(f'{disable_img_paths}')
