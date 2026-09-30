"""
Master Optimization Pipeline
Runs all optimizations in sequence to achieve >97% mAP50
"""
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print()

    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        print(f"\n[SUCCESS] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] {description} failed with error: {e}")
        return False

def main():
    print("=" * 60)
    print("GALLSTONE DETECTION - MASTER OPTIMIZATION PIPELINE")
    print("Target: >97% mAP50")
    print("=" * 60)

    # Step 1: Extend HPO training duration (already done)
    print("\n[HPO] Training duration extended to 50 epochs")

    # Step 2: Run hard negative mining (already done)
    print("\n[HPO] Hard negative mining completed (0 hard negatives found)")

    # Step 3: Optimize ensemble weights
    print("\n" + "=" * 60)
    print("STEP 3: OPTIMIZE ENSEMBLE WEIGHTS")
    print("=" * 60)
    success = run_command(
        "python -c \"import sys; sys.path.insert(0, '.'); exec(open('scripts/optimize_ensemble_weights.py').read())\"",
        "Ensemble weight optimization"
    )

    if not success:
        print("[WARNING] Ensemble weight optimization failed, continuing...")

    # Step 4: Tune shadow analyzer threshold
    print("\n" + "=" * 60)
    print("STEP 4: TUNE SHADOW ANALYZER THRESHOLD")
    print("=" * 60)
    success = run_command(
        "python -c \"import sys; sys.path.insert(0, '.'); exec(open('scripts/tune_shadow_threshold.py').read())\"",
        "Shadow threshold tuning"
    )

    if not success:
        print("[WARNING] Shadow threshold tuning failed, continuing...")

    # Step 5: Optimize multi-scale TTA
    print("\n" + "=" * 60)
    print("STEP 5: OPTIMIZE MULTI-SCALE TTA")
    print("=" * 60)
    success = run_command(
        "python scripts/optimize_tta_scales.py",
        "Multi-scale TTA optimization"
    )

    if not success:
        print("[WARNING] Multi-scale TTA optimization failed, continuing...")

    # Step 6: Run comprehensive evaluation
    print("\n" + "=" * 60)
    print("STEP 6: COMPREHENSIVE EVALUATION")
    print("=" * 60)
    success = run_command(
        "python -c \"import sys; sys.path.insert(0, '.'); exec(open('scripts/comprehensive_evaluation.py').read())\"",
        "Comprehensive evaluation"
    )

    if not success:
        print("[ERROR] Comprehensive evaluation failed")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("OPTIMIZATION PIPELINE COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review results in configs/comprehensive_evaluation.json")
    print("2. If mAP50 < 97%, consider:")
    print("   - Running RF-DETR hyperparameter optimization")
    print("   - Collecting more training data")
    print("   - Trying larger model variants")
    print("3. If mAP50 >= 97%, proceed to clinical deployment")

if __name__ == "__main__":
    main()