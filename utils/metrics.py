import numpy as np
from sklearn.metrics import confusion_matrix


def calculate_binary_classification_metric(preds, gts):
    """
    计算 二分类 任务的评价指标
    :param preds:
    :param gts:
    :return:
    """
    preds = np.array(preds).reshape(-1)
    gts = np.array(gts).reshape(-1)
    confusion = confusion_matrix(gts, preds)
    TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]
    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    precision = float(TP) / float(TP + FP) if float(TP + FP) != 0 else 0
    f1 = float(2 * precision * sensitivity) / float(precision + sensitivity) if float(precision + sensitivity) != 0 else 0

    metric_dict = {'accuracy': accuracy,
                   'sensitivity': sensitivity,
                   'specificity': specificity,
                   'precision': precision,
                   'f1_score': f1,
                   'confusion_matrix': confusion}
    return accuracy, sensitivity, specificity, precision, f1