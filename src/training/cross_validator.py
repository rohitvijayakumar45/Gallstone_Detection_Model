import numpy as np
from sklearn.model_selection import StratifiedKFold


class CrossValidator:
    def __init__(self, n_folds=5, model_type="rf_detr"):
        self.n_folds = n_folds
        self.model_type = model_type
        self.fold_metrics = []

    def run(self, dataset, config):
        patient_ids = sorted({d["patient_id"] for d in dataset})
        patient_labels = []
        for pid in patient_ids:
            patient_labels.append(int(any(d.get("has_gallstone", False) for d in dataset if d["patient_id"] == pid)))

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(patient_ids, patient_labels)):
            train_patients = {patient_ids[i] for i in train_idx}
            val_patients = {patient_ids[i] for i in val_idx}
            train_data = [d for d in dataset if d["patient_id"] in train_patients]
            val_data = [d for d in dataset if d["patient_id"] in val_patients]
            model = self._train_fold(train_data, val_data, config, fold_idx)
            metrics = self._evaluate_fold(model, val_data)
            self.fold_metrics.append(metrics)
        self._print_cv_summary()
        return self.fold_metrics

    def _train_fold(self, train_data, val_data, config, fold_idx):
        raise NotImplementedError

    def _evaluate_fold(self, model, val_data):
        raise NotImplementedError

    def _print_cv_summary(self):
        maps = [m["map50"] for m in self.fold_metrics]
        recalls = [m["recall"] for m in self.fold_metrics]
        print(f"mAP50: {np.mean(maps):.4f} +/- {np.std(maps):.4f}")
        print(f"Recall: {np.mean(recalls):.4f} +/- {np.std(recalls):.4f}")
