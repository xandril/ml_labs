import argparse
import pickle
import sys

from tqdm import tqdm

from cnn import Params, load_mnist, classify
from drawer import *


def make_classify(test_data: np.ndarray, test_target: np.ndarray, params: Params) -> (np.ndarray, list[float]):
    conf_matrix = np.zeros(shape=(CLASS_NUMBER, CLASS_NUMBER))
    progressbar: tqdm = tqdm(range(len(test_data)), file=sys.stdout)
    correct: float = 0.0
    confidence_list: list[tuple[int, np.ndarray]] = []
    for iteration in progressbar:
        data: np.ndarray = test_data[iteration]
        target: int = test_target[iteration]
        output, confidence = classify(data, params)
        confidence_list.append((target, confidence))
        conf_matrix[output][target] += 1
        if output == target:
            correct += 1
        progressbar.set_description(f'Accuracy: {float(correct / (iteration + 1)) * 100:0.2f}%')
    return conf_matrix, confidence_list


def calc_metrics(conf_matrix: np.ndarray) -> (list[float], list[float], list[float], list[int], list[int]):
    cls_counter = [0] * CLASS_NUMBER
    not_cls_counter = [0] * CLASS_NUMBER
    precision_list = [0.0] * CLASS_NUMBER
    recall_list = [0.0] * CLASS_NUMBER
    f1_list = [0.0] * CLASS_NUMBER
    for cls in range(CLASS_NUMBER):
        tp: int = conf_matrix[cls, cls]
        fp: int = conf_matrix[cls, :].sum() - tp
        fn: int = conf_matrix[:, cls].sum() - tp
        tn: int = conf_matrix.sum() - tp - fp - fn
        cls_counter[cls] = tp + fn
        not_cls_counter[cls] = tn + fp
        precision: float = tp / (tp + fp)
        recall: float = tp / (tp + fn)
        f1_score: float = 2 * precision * recall / (precision + recall)
        precision_list[cls] = precision
        recall_list[cls] = recall
        f1_list[cls] = f1_score

    return precision_list, recall_list, f1_list, cls_counter, not_cls_counter


def calc_roc_curve(cls: int, positive: int, negative: int, confidence: list[tuple[int, np.ndarray]]) -> (
        list[float], list[float], list[float]):
    target_len = len(confidence)
    fpr: list[float] = [0.0] * (target_len + 1)
    tpr: list[float] = [0.0] * (target_len + 1)
    auc: float = 0
    confidence: list[tuple[int, np.ndarray]] = sorted(confidence, key=lambda it: -it[1][cls])
    for i in range(target_len):
        if cls != confidence[i][0]:
            fpr[i + 1] = fpr[i] + 1 / negative
            tpr[i + 1] = tpr[i]
            auc += tpr[i + 1] / negative
        else:
            fpr[i + 1] = fpr[i]
            tpr[i + 1] = tpr[i] + 1 / positive
    return fpr, tpr, auc


def calc_roc_curves(cls_counter: list[int], not_cls_counter: list[int], cls_count: int,
                    confidence: list[tuple[int, np.ndarray]]) -> list[tuple[list[float], list[float], float]]:
    roc_list = []
    for i in range(cls_count):
        fpr, tpr, auc = calc_roc_curve(i, cls_counter[i], not_cls_counter[i], confidence)
        roc_list.append((fpr, tpr, auc))
    return roc_list


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('parameters', metavar='parameters', type=str)
    args = parser.parse_args()
    parameters_file_name: str = args.parameters
    with open(parameters_file_name, 'rb') as parameters_file:
        params, cost = pickle.load(parameters_file)
    _, (test_data, test_target) = load_mnist()
    confusion_matrix, confidence_list = make_classify(test_data, test_target, params)
    precision_list, recall_list, f1_list, cls_counter, not_cls_counter = calc_metrics(confusion_matrix)
    roc_list = calc_roc_curves(cls_counter, not_cls_counter, CLASS_NUMBER, confidence_list)
    class_names = [str(i) for i in range(CLASS_NUMBER)]
    cost_fig = draw_cost(cost)
    conf_mat_fig = draw_conf_matrix(confusion_matrix)
    roc_fig = draw_roc_curves(roc_list, class_names)
    precision_fig = draw_metric_bar(class_names, precision_list, "precision")
    recall_fig = draw_metric_bar(class_names, recall_list, "recall")
    f1_fig = draw_metric_bar(class_names, f1_list, "f1_score")
    save_param_to_html(cost_fig, parameters_file_name, "cost.html")
    save_param_to_html(conf_mat_fig, parameters_file_name, "conf_mat.html")
    save_param_to_html(roc_fig, parameters_file_name, "roc_curve.html")
    save_param_to_html(precision_fig, parameters_file_name, "precision.html")
    save_param_to_html(recall_fig, parameters_file_name, "recall.html")
    save_param_to_html(f1_fig, parameters_file_name, "f1.html")


if __name__ == '__main__':
    main()
