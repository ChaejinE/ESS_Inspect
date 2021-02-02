import os
import cv2
import numpy as np
from absl import logging
from src.main.python.inspector.dl.tf2_model import TF2SavedModel
from evaluator.eval_metric import CategoricalResult


def main(_):
    tf_model = TF2SavedModel(pb_dir=FLAGS.model_dir)
    # categories = [c for c in os.listdir(FLAGS.data_dir) if c[0] != "."]
    categories = {"0.normal_insert": 0, "1.normal_mold": 1, "2.miss_insert": 2,
                  "3.reverse_insert": 3, "4.protous_insert": 4, "5.unmold": 5}
    logging.info(categories)
    eval_result = CategoricalResult(categories)

    for c in categories:
        c_dir = os.path.join(FLAGS.data_dir, c)
        files = [f for f in os.listdir(c_dir) if f.endswith(FLAGS.ext)]
        for file in files:
            file_path = os.path.join(c_dir, file)
            image = np.array(cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB))
            output = tf_model.inference(image, is_decoded=False, argmax=True)
            predict_c = int(output)
            eval_result.append(c, predict_c)

    logging.info(eval_result.result())


if __name__ == "__main__":
    from absl import flags
    from absl import app

    flags.DEFINE_string("model_dir", default=None, help="the Directory which has tf2 frozen model(*.pb)", short_name="m")
    flags.DEFINE_string("data_dir", default=None, help="the Directory which has images for testing", short_name="d")
    flags.DEFINE_string("ext", default="jpg", help="", short_name="e")
    flags.DEFINE_multi_string("categories", default=None, help="Names of category to testing ( None == all )")

    flags.mark_flag_as_required("model_dir")
    flags.mark_flag_as_required("data_dir")

    FLAGS = flags.FLAGS

    app.run(main, )

