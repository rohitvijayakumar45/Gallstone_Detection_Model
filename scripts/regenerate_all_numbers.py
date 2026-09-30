"""Cross-platform regen driver. Equivalent to scripts/regenerate_all_numbers.sh
but runs on Windows cmd/PowerShell without needing WSL/bash.

Usage:
  python scripts/regenerate_all_numbers.py
  python scripts/regenerate_all_numbers.py --skip tests --skip manifest
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    (
        "tests",
        [sys.executable, "-m", "pytest", "tests/", "-q",
         "--ignore=tests/test_smoke.py"],
        "unit tests (fast; skips model-loading smoke)",
    ),
    (
        "manifest",
        [sys.executable, "scripts/generate_manifest.py"],
        "SHA256 of every image + weight -> MANIFEST.md/json",
    ),
    (
        "baseline",
        [sys.executable, "scripts/comprehensive_evaluation.py",
         "--split", "test",
         "--out", "runs/final_eval/comprehensive_evaluation.json"],
        "ablation ladder on test=201",
    ),
    (
        "shadow",
        [sys.executable, "scripts/tune_shadow_threshold.py",
         "--split", "val",
         "--out", "runs/final_eval/shadow_threshold_tuning.json"],
        "shadow-threshold sweep on val=368",
    ),
    (
        "calibration",
        [sys.executable, "scripts/fit_calibration.py",
         "--split", "val",
         "--out-dir", "production_models",
         "--report", "runs/final_eval/calibration.json"],
        "temperature scaling + isotonic fusion on val=368",
    ),
    (
        "conformal",
        [sys.executable, "scripts/fit_conformal.py",
         "--calibration-split", "val",
         "--test-split", "test",
         "--alpha", "0.05",
         "--out-dir", "production_models",
         "--report", "runs/final_eval/conformal.json"],
        "split-conformal box expansion + CRC recall bound",
    ),
]


def _run(name: str, cmd: list[str], desc: str) -> tuple[str, float, int]:
    print(f"\n{'=' * 78}\n== {name}: {desc}\n{'-' * 78}")
    print("  $ " + " ".join(cmd))
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.perf_counter() - t0
    status = "OK" if proc.returncode == 0 else f"FAIL ({proc.returncode})"
    print(f"  -> {status} in {elapsed:.1f}s")
    return name, elapsed, proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate all published numbers")
    parser.add_argument("--skip", action="append", default=[],
                        choices=[s[0] for s in STEPS],
                        help="skip a step (repeat to skip several)")
    parser.add_argument("--only", action="append", default=[],
                        choices=[s[0] for s in STEPS],
                        help="run only these steps (repeat to add several)")
    parser.add_argument("--stop-on-fail", action="store_true",
                        help="abort at first non-zero exit code")
    args = parser.parse_args()

    steps = STEPS
    if args.only:
        steps = [s for s in steps if s[0] in args.only]
    if args.skip:
        steps = [s for s in steps if s[0] not in args.skip]

    results: list[tuple[str, float, int]] = []
    for name, cmd, desc in steps:
        results.append(_run(name, cmd, desc))
        if args.stop_on_fail and results[-1][2] != 0:
            print(f"\n[abort] {name} failed; --stop-on-fail set")
            break

    print(f"\n{'=' * 78}\n== SUMMARY\n{'-' * 78}")
    for name, elapsed, rc in results:
        print(f"  {'OK ' if rc == 0 else 'FAIL':<4}  {name:<15} {elapsed:>7.1f}s")
    print()
    failed = [r for r in results if r[2] != 0]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
