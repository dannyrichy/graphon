import numpy as np
from sklearn.metrics import confusion_matrix


def make_cost_m(conf_matrix):
    s = np.max(conf_matrix)
    return - conf_matrix + s


def error(gt_real, labels):
    cm = confusion_matrix(gt_real, labels)
    indexes = linear_assignment(make_cost_m(cm))  # Hungarian algorithm
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]
    err = 1 - np.trace(cm2) / np.sum(cm2)
    return err