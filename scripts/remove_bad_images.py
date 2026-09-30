"""
Remove Bad Images from Dataset
Removes images marked as bad during the review process
"""
import shutil
from pathlib import Path
import json
from datetime import datetime

def remove_bad_images(data_dir="dataset_final_resplit", backup=True, split="val"):
    """
    Remove images marked as bad from the dataset

    Args:
        data_dir: Dataset directory
        backup: Whether to create a backup before deletion
        split: Dataset split to remove images from (train, val, test)
    """
    data_path = Path(data_dir)
    review_file = data_path / f"review_state_{split}.json"

    if not review_file.exists():
        print(f"No review state found at {review_file}")
        print("Run the review tool first to mark bad images.")
        return False

    # Load review state
    with open(review_file, 'r') as f:
        state = json.load(f)

    bad_images = set(state.get('bad', []))

    if not bad_images:
        print("No bad images marked for removal.")
        return False

    print(f"Found {len(bad_images)} bad images marked for removal")

    # Create backup if requested
    if backup:
        backup_dir = data_path.parent / f"{data_path.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Creating backup at {backup_dir}")

        # Copy entire dataset to backup
        shutil.copytree(data_path, backup_dir)
        print(f"Backup created successfully")

    # Remove bad images
    split_dir = data_path / split
    img_dir = split_dir / "images"
    label_dir = split_dir / "labels"

    removed_count = 0
    failed_removals = []

    for img_name in bad_images:
        img_path = img_dir / img_name
        label_path = label_dir / f"{img_name.rsplit('.', 1)[0]}.txt"

        try:
            # Remove image
            if img_path.exists():
                img_path.unlink()
                print(f"Removed image: {img_name}")
                removed_count += 1
            else:
                print(f"Warning: Image not found: {img_name}")
                failed_removals.append(img_name)

            # Remove label
            if label_path.exists():
                label_path.unlink()
                print(f"Removed label: {label_path.name}")
            else:
                print(f"Warning: Label not found: {label_path.name}")

        except Exception as e:
            print(f"Error removing {img_name}: {e}")
            failed_removals.append(img_name)

    # Update review state
    state['bad'] = []
    state['removed'] = list(bad_images)
    state['removal_timestamp'] = datetime.now().isoformat()

    with open(review_file, 'w') as f:
        json.dump(state, f, indent=2)

    print(f"\n" + "=" * 60)
    print("REMOVAL SUMMARY")
    print("=" * 60)
    print(f"Total bad images marked: {len(bad_images)}")
    print(f"Successfully removed: {removed_count}")
    print(f"Failed to remove: {len(failed_removals)}")

    if failed_removals:
        print(f"\nFailed removals:")
        for img_name in failed_removals:
            print(f"  - {img_name}")

    # Count remaining images
    remaining_images = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
    print(f"\nRemaining validation images: {remaining_images}")

    # Update split info
    split_info_file = data_path / "split_info.json"
    if split_info_file.exists():
        with open(split_info_file, 'r') as f:
            split_info = json.load(f)

        if split not in split_info:
            split_info[split] = {}
            
        split_info[split]['count'] = remaining_images
        split_info[split]['removed'] = split_info[split].get('removed', 0) + removed_count
        split_info['last_updated'] = datetime.now().isoformat()

        with open(split_info_file, 'w') as f:
            json.dump(split_info, f, indent=2)

        print(f"Updated split info: {split_info_file}")

    print("\n" + "=" * 60)
    print("BAD IMAGES REMOVED SUCCESSFULLY")
    print("=" * 60)

    if backup:
        print(f"\nBackup available at: {backup_dir}")
        print("You can restore from backup if needed.")

    return True

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Remove bad images from dataset")
    parser.add_argument("--data_dir", type=str, default="dataset_final_resplit",
                       help="Dataset directory")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                       help="Dataset split to clean (train, val, test)")
    parser.add_argument("--no-backup", action="store_true",
                       help="Skip creating backup before deletion")

    args = parser.parse_args()

    print("=" * 60)
    print(f"REMOVE BAD IMAGES FROM DATASET ({args.split.upper()})")
    print("=" * 60)

    success = remove_bad_images(
        data_dir=args.data_dir,
        backup=not args.no_backup,
        split=args.split
    )

    if success:
        print("\nNext steps:")
        print("1. Re-run preprocessing with cleaned dataset")
        print("2. Train model with clean validation set")
        print("3. Run evaluation to verify improvement")
    else:
        print("\nRemoval failed. Please check the error messages above.")

if __name__ == "__main__":
    main()