from typing import Dict

import numpy as np


def get_scores(
    y_target: np.ndarray, y_pred: np.ndarray, y_pathology: np.ndarray, postfix: str = ""
) -> Dict[str, float]:
    ddr = []  # ddx precision
    ddp = []  # ddx recall
    gtpa = []  # ground truth pathology accuracy
    gtpa_at_1 = []

    for y_target_i, y_pred_i, y_pathology_i in zip(y_target, y_pred, y_pathology):
        y_pred_i_ix = set(np.where(y_pred_i > 0.5)[0])
        y_target_i_ix = set(np.where(y_target_i > 0.5)[0])

        # precision and recall
        intersection = y_pred_i_ix.intersection(y_target_i_ix)

        ddr.append(len(intersection) / len(y_target_i_ix))
        if len(y_pred_i_ix) > 0:
            ddp.append(len(intersection) / len(y_pred_i_ix))
        else:
            ddp.append(0)

        # gtpa
        if y_pathology_i in y_pred_i_ix:
            gtpa.append(1)
        else:
            gtpa.append(0)

        # gtpa @ 1
        first_pathology_predicted = y_pred_i.argmax()
        if y_pathology_i == first_pathology_predicted:
            gtpa_at_1.append(1)
        else:
            gtpa_at_1.append(0)

    recall = np.mean(ddr)
    precision = np.mean(ddp)
    if recall + precision <= 1e-6:
        f1 = 0
    else:
        f1 = 2 * recall * precision / (recall + precision)

    gtpa = np.mean(gtpa)
    gtpa_at_1 = np.mean(gtpa_at_1)

    return {
        f"recall{postfix}": recall,
        f"precision{postfix}": precision,
        f"f1{postfix}": f1,
        f"gtpa{postfix}": gtpa,
        f"gtpa_at_1{postfix}": gtpa_at_1,
    }
