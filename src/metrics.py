from sklearn.metrics import confusion_matrix, f1_score
import numpy as np

# Compute confusion matrix for ordinal classes
def compute_ordinal_conf_matrix(y_true, y_pred, num_classes=5):
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

# Print F1-score for each class
def print_f1_per_class(y_true, y_pred, num_classes=5):
    scores = f1_score(y_true, y_pred, average=None, labels=range(num_classes))
    for i, score in enumerate(scores):
        print(f"Class {i} F1: {score:.4f}")
    return scores

# xView2 scoring: combines localization and damage detection performance
def calculate_xview2_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    loc_true = (y_true > 0).astype(int)
    loc_pred = (y_pred > 0).astype(int)
    loc_f1 = f1_score(loc_true, loc_pred, zero_division=0)
    building_mask = y_true > 0
    dmg_f1 = f1_score(y_true[building_mask], y_pred[building_mask], average='macro', zero_division=0) if np.any(building_mask) else 0.0
    return 0.3 * loc_f1 + 0.7 * dmg_f1