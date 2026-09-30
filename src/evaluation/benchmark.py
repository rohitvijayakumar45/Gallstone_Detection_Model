def compare_to_targets(metrics):
    return {
        "map50_target_met": metrics.get("mAP50", 0) >= 0.97,
        "recall_target_met": metrics.get("recall", 0) >= 0.95,
        "fnr_target_met": metrics.get("fnr", 1) < 0.05,
    }
