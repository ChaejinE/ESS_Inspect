from typing import Union, Dict


class CategoricalResult:
    def __init__(self, category_dict: Dict):
        self._result_dict = {c: {True: 0, False: 0} for c in category_dict}
        self._category_dict = {category_dict[c]: c for c in category_dict}

    def append(self,
               category: Union[str, int],
               predict: int):
        category_name = category
        if isinstance(category, int):
            category_name = self._category_dict[category]

        is_true = category_name == self._category_dict[predict]

        self._result_dict[category_name][is_true] += 1

    def total_count(self):
        count = 0
        for c in self._result_dict:
            true_count = self._result_dict[c][True]
            false_count = self._result_dict[c][False]

            count += (true_count + false_count)

        return count

    def result(self, name: str = "acc"):
        """

        :param name: only use acc
        :return:
        """

        result_dict = {"total_acc": 0}
        true_count = 0
        false_count = 0
        for c in self._result_dict:
            true_count += self._result_dict[c][True]
            false_count += self._result_dict[c][False]

            result_dict["{}_total".format(c)] = (self._result_dict[c][True] + self._result_dict[c][False])
            c_acc = self._result_dict[c][True] / result_dict["{}_total".format(c)]
            result_dict["{}_acc".format(c)] = c_acc

        result_dict["total_acc"] = (true_count + false_count) / true_count

        return result_dict
