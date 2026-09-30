import numpy as np
from sklearn.metrics import auc, confusion_matrix, precision_recall_fscore_support, roc_curve


class GallstoneEvaluator:
    def evaluate_all(self, model, test_loader, conf=0.25, iou=0.45):
        return {
            **self._compute_coco_metrics(model, test_loader, conf, iou),
            **self._compute_clinical_metrics(model, test_loader, conf),
            **self._compute_per_size_metrics(model, test_loader, conf),
        }

    def _compute_coco_metrics(self, model, loader, conf, iou):
        return {
            "mAP50": np.nan,
            "mAP50_95": np.nan,
            "mAP_small": np.nan,
            "mAP_medium": np.nan,
            "mAP_large": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
        }

    def _compute_clinical_metrics(self, model, loader, conf):
        y_true, y_score = [], []
        for batch in loader or []:
            y_true.extend(batch.get("has_stone", []))
            y_score.extend(batch.get("score", []))
        if not y_true:
            return {
                "sensitivity": np.nan,
                "specificity": np.nan,
                "auc_roc": np.nan,
                "fnr": np.nan,
                "fpr": np.nan,
                "ppv": np.nan,
                "npv": np.nan,
            }
        y_pred = [int(s >= conf) for s in y_score]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fpr_curve, tpr_curve, _ = roc_curve(y_true, y_score)
        return {
            "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "auc_roc": auc(fpr_curve, tpr_curve),
            "fnr": fn / max(tp + fn, 1),
            "fpr": fp / max(tn + fp, 1),
            "ppv": tp / max(tp + fp, 1),
            "npv": tn / max(tn + fn, 1),
        }

    def _compute_per_size_metrics(self, model, loader, conf):
        return {"recall_tiny": np.nan, "recall_small": np.nan, "recall_medium": np.nan, "recall_large": np.nan}


def precision_recall_f1(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"precision": precision, "recall": recall, "f1": f1}
