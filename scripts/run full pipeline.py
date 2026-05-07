#!/usr/bin/env python3
"""
run_full_pipeline.py

End-to-end pipeline that executes all steps:
1. Data loading & preprocessing
2. Model training (all models)
3. Evaluation on validation set
4. Robustness simulation
5. Figure generation

Usage:
    python scripts/run_full_pipeline.py --config config.yaml

This is the primary entry point to reproduce the entire paper's computational workflow.
"""

import argparse
import sys
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="DeepTriage-CN Full Reproducibility Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training (assumes models already exist)"
    )
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Skip robustness simulation"
    )
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    project_root = scripts_dir.parent

    steps = []

    # Step 1: Data exploration (optional, but recommended)
    # This is a notebook step; we skip here and go directly to training

    # Step 2: Train all models
    if not args.skip_training:
        print("=" * 60)
        print("STEP 1/5: Training all models ...")
        print("=" * 60)
        subprocess.run(
            [sys.executable, str(scripts_dir / "train_all_models.py"),
             "--config", args.config],
            check=True, cwd=str(project_root)
        )
    else:
        print("[SKIP] Model training skipped.")

    # Step 3: Evaluate all models
    print("\n" + "=" * 60)
    print("STEP 2/5: Evaluating all models ...")
    print("=" * 60)
    subprocess.run(
        [sys.executable, str(scripts_dir / "evaluate_all_models.py"),
         "--config", args.config],
        check=True, cwd=str(project_root)
    )

    # Step 4: Robustness simulation
    if not args.skip_robustness:
        print("\n" + "=" * 60)
        print("STEP 3/5: Robustness simulation ...")
        print("=" * 60)
        subprocess.run(
            [sys.executable, str(scripts_dir / "robustness_simulation.py"),
             "--config", args.config],
            check=True, cwd=str(project_root)
        )
    else:
        print("[SKIP] Robustness simulation skipped.")

    # Step 5: Generate all figures
    print("\n" + "=" * 60)
    print("STEP 4/5: Generating all figures ...")
    print("=" * 60)
    subprocess.run(
        [sys.executable, str(scripts_dir / "generate_all_figures.py"),
         "--config", args.config],
        check=True, cwd=str(project_root)
    )

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print(f"Outputs saved to:")
    print(f"  Models:   {project_root / 'outputs/models/'}")
    print(f"  Results:  {project_root / 'outputs/results/'}")
    print(f"  Figures:  {project_root / 'outputs/figures/'}")
    print("=" * 60)

if __name__ == "__main__":
    main()