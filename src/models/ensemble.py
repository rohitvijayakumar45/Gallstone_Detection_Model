from __future__ import annotations

import cv2
import numpy as np


def _predict_model(model, image, conf_floor: float = 0.05):
    """Wrap heterogeneous detector .predict() APIs, passing an explicit low
    confidence floor so downstream WBF sees candidates the model would
    otherwise drop at its default threshold. Returns (boxes_norm, scores, labels).

    `image` is assumed BGR (cv2 convention, matching every caller in this
    codebase). Ultralytics YOLO expects BGR and converts internally. RF-DETR
    (rfdetr package, built on `supervision`) expects RGB — feeding it BGR
    silently produces near-empty/garbage predictions with no error, which
    previously caused the ensemble to lose all RF-DETR detections while a
    direct RGB-converted call to the same model found real boxes.
    """
    cls_name = type(model).__name__.lower()

    # Ultralytics YOLO — accepts `conf=` kwarg, wants BGR (cv2 default).
    if cls_name.startswith("yolo"):
        try:
            results = model.predict(image, conf=conf_floor, verbose=False)
            if isinstance(results, list) and len(results) > 0 and hasattr(results[0], "boxes"):
                r = results[0]
                if r.boxes is None or len(r.boxes) == 0:
                    return [], [], []
                return (r.boxes.xyxyn.cpu().numpy(),
                        r.boxes.conf.cpu().numpy(),
                        r.boxes.cls.cpu().numpy())
        except Exception:
            pass

    # RF-DETR — accepts `threshold=` kwarg, REQUIRES RGB input.
    if "detr" in cls_name or "rfdetr" in cls_name:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        try:
            pred = model.predict(rgb, threshold=conf_floor)
        except TypeError:
            pred = model.predict(rgb)
        if hasattr(pred, "xyxy") and hasattr(pred, "confidence"):
            return (pred.xyxy, pred.confidence,
                    pred.class_id if hasattr(pred, "class_id") else [0] * len(pred.xyxy))
        if isinstance(pred, tuple) and len(pred) == 3:
            return pred
        if isinstance(pred, dict):
            return pred.get("boxes", []), pred.get("scores", []), pred.get("labels", [])
        return [], [], []

    # Generic fallback — try predict with keyword arg, else without.
    try:
        pred = model.predict(image, conf=conf_floor)
    except Exception:
        try:
            pred = model.predict(image)
        except Exception:
            return [], [], []
    if hasattr(pred, "xyxy") and hasattr(pred, "confidence"):
        return (pred.xyxy, pred.confidence,
                pred.class_id if hasattr(pred, "class_id") else [0] * len(pred.xyxy))
    if isinstance(pred, tuple) and len(pred) == 3:
        return pred
    if isinstance(pred, dict):
        return pred.get("boxes", []), pred.get("scores", []), pred.get("labels", [])
    return [], [], []


class GallstoneEnsemble:
    """RF-DETR + YOLOv8 + YOLO11 ensemble with Weighted Box Fusion."""

    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [0.5, 0.3, 0.2][: len(models)]

    def predict(self, image, conf_threshold=0.25, iou_threshold=0.5):
        all_boxes, all_scores, all_labels = [], [], []
        h, w = image.shape[:2]
        # Feed the models a low floor so WBF gets low-confidence candidates
        # too; downstream skip_box_thr still filters the fused output.
        model_floor = min(conf_threshold, 0.05)

        for model in self.models:
            boxes, scores, labels = _predict_model(model, image, conf_floor=model_floor)

            boxes = np.asarray(boxes)
            if boxes.size > 0 and np.max(boxes) > 1.05:
                boxes = boxes / [w, h, w, h]

            all_boxes.append(boxes.tolist())
            all_scores.append(np.asarray(scores).tolist())
            all_labels.append(np.asarray(labels).astype(int).tolist())

        try:
            from ensemble_boxes import weighted_boxes_fusion
        except Exception:
            return self._naive_avg_fusion(all_boxes, all_scores, all_labels,
                                          iou_threshold, conf_threshold)

        boxes_out, scores_out, labels_out = weighted_boxes_fusion(
            all_boxes,
            all_scores,
            all_labels,
            weights=self.weights,
            iou_thr=iou_threshold,
            skip_box_thr=conf_threshold,
        )

        # Single-model preservation: if WBF returned nothing but a detector
        # produced candidates above `conf_threshold`, keep them verbatim. This
        # avoids losing DETR-only detections on OOD images where YOLO is
        # silent and WBF's mean-score can drop below skip_box_thr.
        if len(scores_out) == 0:
            surv_b, surv_s, surv_l = [], [], []
            for bs, ss, ls in zip(all_boxes, all_scores, all_labels):
                for b, s, l in zip(bs, ss, ls):
                    if s >= conf_threshold:
                        surv_b.append(b); surv_s.append(s); surv_l.append(l)
            if surv_b:
                return (np.asarray(surv_b, dtype=np.float32),
                        np.asarray(surv_s, dtype=np.float32),
                        np.asarray(surv_l, dtype=np.int64))

        return boxes_out, scores_out, labels_out

    @staticmethod
    def _iou(a, b):
        xa = max(a[0], b[0]); ya = max(a[1], b[1])
        xb = min(a[2], b[2]); yb = min(a[3], b[3])
        inter = max(0.0, xb - xa) * max(0.0, yb - ya)
        aa = (a[2] - a[0]) * (a[3] - a[1])
        ab = (b[2] - b[0]) * (b[3] - b[1])
        return float(inter / (aa + ab - inter + 1e-9))

    def _naive_avg_fusion(self, all_boxes, all_scores, all_labels,
                          iou_thr, conf_thr):
        """Simple IoU-matched averager, used when `ensemble_boxes` is absent."""
        flat = []
        for mi, (bs, ss, ls) in enumerate(zip(all_boxes, all_scores, all_labels)):
            w = self.weights[mi] if mi < len(self.weights) else 1.0
            for b, s, l in zip(bs, ss, ls):
                if s >= conf_thr:
                    flat.append((np.array(b, dtype=np.float32), float(s) * float(w), int(l)))
        flat.sort(key=lambda x: -x[1])
        fused_b, fused_s, fused_l = [], [], []
        used = [False] * len(flat)
        for i, (bi, si, li) in enumerate(flat):
            if used[i]:
                continue
            group_b, group_s = [bi], [si]
            for j in range(i + 1, len(flat)):
                if used[j] or flat[j][2] != li:
                    continue
                if self._iou(bi, flat[j][0]) >= iou_thr:
                    group_b.append(flat[j][0])
                    group_s.append(flat[j][1])
                    used[j] = True
            fb = np.average(np.stack(group_b), axis=0,
                            weights=np.asarray(group_s))
            fused_b.append(fb.tolist())
            fused_s.append(float(np.mean(group_s)))
            fused_l.append(li)
        return np.asarray(fused_b), np.asarray(fused_s), np.asarray(fused_l)

    def predict_with_tta(self, image, tta_transforms):
        all_boxes, all_scores, all_labels, weights = [], [], [], []
        for transform in tta_transforms:
            augmented = transform(image=image)["image"]
            for model, weight in zip(self.models, self.weights):
                boxes, scores, labels = _predict_model(model, augmented)
                all_boxes.append(np.asarray(boxes).tolist())
                all_scores.append(np.asarray(scores).tolist())
                all_labels.append(np.asarray(labels).astype(int).tolist())
                weights.append(weight)

        from ensemble_boxes import weighted_boxes_fusion

        return weighted_boxes_fusion(
            all_boxes,
            all_scores,
            all_labels,
            weights=weights,
            iou_thr=0.5,
            skip_box_thr=0.1,
        )
    def predict_multi_scale(self, image, scales=[640, 800, 1024], conf_threshold=0.25, iou_threshold=0.5):
        import cv2
        all_boxes, all_scores, all_labels, weights = [], [], [], []
        orig_h, orig_w = image.shape[:2]
        
        for scale in scales:
            # Resize image to target scale
            resized = cv2.resize(image, (scale, scale), interpolation=cv2.INTER_LINEAR)
            
            for model, weight in zip(self.models, self.weights):
                boxes, scores, labels = _predict_model(model, resized)
                
                # boxes from _predict_model are normalized [0, 1] relative to resized image
                all_boxes.append(np.asarray(boxes).tolist())
                all_scores.append(np.asarray(scores).tolist())
                all_labels.append(np.asarray(labels).astype(int).tolist())
                weights.append(weight)

        from ensemble_boxes import weighted_boxes_fusion

        return weighted_boxes_fusion(
            all_boxes,
            all_scores,
            all_labels,
            weights=weights,
            iou_thr=iou_threshold,
            skip_box_thr=conf_threshold,
        )
