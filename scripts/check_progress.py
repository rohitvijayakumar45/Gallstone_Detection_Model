"""
Check HPO and Training Progress
"""
import sqlite3
from pathlib import Path
import json
from datetime import datetime

def check_hpo_progress():
    """Check HPO progress from database"""
    hpo_db = Path("yolo_hpo_results.db")

    if not hpo_db.exists():
        print("No HPO database found")
        return

    try:
        conn = sqlite3.connect(hpo_db)
        cursor = conn.cursor()

        # Get total trials
        cursor.execute("SELECT COUNT(*) FROM trials")
        total_trials = cursor.fetchone()[0]

        # Get best trial
        cursor.execute("SELECT trial_number, value FROM trials ORDER BY value DESC LIMIT 1")
        best_trial = cursor.fetchone()

        # Get recent trials
        cursor.execute("SELECT trial_number, value FROM trials ORDER BY trial_number DESC LIMIT 5")
        recent_trials = cursor.fetchall()

        conn.close()

        print("=" * 60)
        print("HPO PROGRESS")
        print("=" * 60)
        print(f"Total trials completed: {total_trials}")

        if best_trial:
            trial_num, best_value = best_trial
            print(f"Best trial: #{trial_num}")
            print(f"Best value: {best_value:.4f}")

        print("\nRecent trials:")
        for trial_num, value in recent_trials:
            print(f"  Trial #{trial_num}: {value:.4f}")

    except Exception as e:
        print(f"Error reading HPO database: {e}")

def check_training_progress():
    """Check training progress from results files"""
    results_files = [
        "runs/detect/gallstone_detection/yolo26m_gallstone/results.csv",
        "runs/detect/gallstone_detection/yolo26l_gallstone/results.csv"
    ]

    print("\n" + "=" * 60)
    print("TRAINING PROGRESS")
    print("=" * 60)

    for results_file in results_files:
        results_path = Path(results_file)
        if not results_path.exists():
            continue

        model_name = results_path.parent.name
        print(f"\n{model_name}:")

        try:
            with open(results_path, 'r') as f:
                lines = f.readlines()

            if len(lines) < 2:
                print("  No training data")
                continue

            # Get header
            header = lines[0].strip().split(',')

            # Get last line
            last_line = lines[-1].strip().split(',')

            # Create dict
            data = dict(zip(header, last_line))

            # Extract key metrics
            epoch = data.get('epoch', 'N/A')
            mAP50 = data.get('metrics/mAP50(B)', 'N/A')
            precision = data.get('metrics/precision(B)', 'N/A')
            recall = data.get('metrics/recall(B)', 'N/A')
            mAP50_95 = data.get('metrics/mAP50-95(B)', 'N/A')

            print(f"  Epochs: {epoch}")
            print(f"  mAP50: {mAP50}")
            print(f"  Precision: {precision}")
            print(f"  Recall: {recall}")
            print(f"  mAP50-95: {mAP50_95}")

        except Exception as e:
            print(f"  Error reading results: {e}")

def check_dataset_status():
    """Check dataset status"""
    print("\n" + "=" * 60)
    print("DATASET STATUS")
    print("=" * 60)

    datasets = ["dataset_final_resplit", "data/processed"]

    for dataset in datasets:
        dataset_path = Path(dataset)
        if not dataset_path.exists():
            continue

        print(f"\n{dataset}:")

        # Count images in each split
        for split in ['train', 'val', 'test']:
            split_dir = dataset_path / split / 'images'
            if split_dir.exists():
                count = len(list(split_dir.glob("*.jpg"))) + len(list(split_dir.glob("*.png")))
                print(f"  {split}: {count} images")

        # Check split info if available
        split_info = dataset_path / "split_info.json"
        if split_info.exists():
            with open(split_info, 'r') as f:
                info = json.load(f)
            print(f"  Total images: {info.get('total_images', 'N/A')}")
            print(f"  Last updated: {info.get('last_updated', 'N/A')}")

def check_model_weights():
    """Check available model weights"""
    print("\n" + "=" * 60)
    print("MODEL WEIGHTS")
    print("=" * 60)

    # Check YOLO weights
    yolo_weights_dir = Path("runs/detect/gallstone_detection")
    if yolo_weights_dir.exists():
        for model_dir in yolo_weights_dir.iterdir():
            if model_dir.is_dir():
                weights_dir = model_dir / "weights"
                if weights_dir.exists():
                    best_weight = weights_dir / "best.pt"
                    last_weight = weights_dir / "last.pt"

                    print(f"\n{model_dir.name}:")
                    if best_weight.exists():
                        size_mb = best_weight.stat().st_size / (1024 * 1024)
                        print(f"  best.pt: {size_mb:.1f} MB")
                    if last_weight.exists():
                        size_mb = last_weight.stat().st_size / (1024 * 1024)
                        print(f"  last.pt: {size_mb:.1f} MB")

    # Check RF-DETR weights
    rfdetr_weights_dir = Path("weights/rf_detr")
    if rfdetr_weights_dir.exists():
        print(f"\nRF-DETR weights:")
        for weight_file in rfdetr_weights_dir.glob("*.pth"):
            size_mb = weight_file.stat().st_size / (1024 * 1024)
            print(f"  {weight_file.name}: {size_mb:.1f} MB")

def main():
    print("=" * 60)
    print("GALLSTONE DETECTION - PROGRESS CHECK")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    check_hpo_progress()
    check_training_progress()
    check_dataset_status()
    check_model_weights()

    print("\n" + "=" * 60)
    print("PROGRESS SUMMARY")
    print("=" * 60)

    # Overall assessment
    print("\n[COMPLETED]")
    print("  - Dataset resplitting (1,233 images)")
    print("  - Validation image review (28 bad images removed)")
    print("  - Baseline YOLO training (150 epochs)")

    print("\n[IN PROGRESS]")
    print("  - YOLO hyperparameter optimization")

    print("\n[NEXT STEPS]")
    print("  - Complete YOLO HPO")
    print("  - Train RF-DETR model")
    print("  - Run RF-DETR HPO")
    print("  - Full optimization pipeline")

if __name__ == "__main__":
    main()