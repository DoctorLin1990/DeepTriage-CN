#!/usr/bin/env python3
"""
run_full_pipeline.py

One-command entry point to reproduce all DeepTriage-CN results from scratch:

    Step 1  generate_synthetic_data    – create synthetic validation dataset
    Step 2  train_all_models           – train all models
    Step 3  evaluate_all_models        – compute Table 2 metrics + CIs
    Step 4  robustness_simulation      – data-degradation robustness table
    Step 5  generate_all_figures       – produce all paper figures (TIFF)

Bug fix applied (B6): All original script file names contained spaces
(e.g. "train all models.py"), which are illegal as Python module names and
break subprocess calls. All references now use underscores.

Usage:
    # Quick smoke-test on synthetic data (no GPU required)
    python scripts/run_full_pipeline.py --config config.yaml

    # Skip data generation if synthetic CSV already exists
    python scripts/run_full_pipeline.py --config config.yaml --skip-datagen

    # Skip figure generation (faster CI run)
    python scripts/run_full_pipeline.py --config config.yaml --skip-figures
"""

import argparse
import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent


def run(script: str, config: str, description: str) -> None:
    """Run a pipeline step as a subprocess and abort on failure."""
    cmd = [sys.executable, str(SCRIPTS_DIR / script), "--config", config]
    logger.info("=" * 60)
    logger.info("STEP: %s", description)
    logger.info("CMD : %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error(
            "Step '%s' failed (exit code %d). Aborting pipeline.",
            description, result.returncode,
        )
        sys.exit(result.returncode)
    logger.info("DONE: %s", description)


def main():
    parser = argparse.ArgumentParser(
        description="Run the full DeepTriage-CN reproducibility pipeline."
    )
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config YAML (default: config.yaml)")
    parser.add_argument("--skip-datagen", action="store_true",
                        help="Skip synthetic data generation (CSV must already exist).")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Skip figure generation.")
    args = parser.parse_args()

    logger.info("DeepTriage-CN — Full Reproducibility Pipeline")
    logger.info("Config: %s", args.config)

    # Step 1 – Synthetic data generation (optional)
    if not args.skip_datagen:
        # generate_synthetic_data.py does not use --config; it writes directly
        datagen_script = SCRIPTS_DIR.parent / "data" / "generate_synthetic_data.py"
        cmd = [sys.executable, str(datagen_script)]
        logger.info("=" * 60)
        logger.info("STEP: Generate synthetic dataset")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logger.error("Synthetic data generation failed.")
            sys.exit(result.returncode)
    else:
        logger.info("Skipping synthetic data generation.")

    # Step 2 – Train all models
    run("train_all_models.py", args.config,
        "Train all models (DeepTriage-CN, TabNet, baselines)")

    # Step 3 – Evaluate all models
    run("evaluate_all_models.py", args.config,
        "Evaluate all models (Table 2, CIs, DeLong, NRI, calibration)")

    # Step 4 – Robustness simulation
    run("robustness_simulation.py", args.config,
        "Robustness simulation (10%/20%/30% MNAR + noise)")

    # Step 5 – Generate figures
    if not args.skip_figures:
        run("generate_all_figures.py", args.config,
            "Generate all paper figures (Figures 2–6, Supp. Fig. 1)")
    else:
        logger.info("Skipping figure generation.")

    logger.info("=" * 60)
    logger.info("Pipeline complete.  All outputs written to the paths")
    logger.info("specified in '%s'.", args.config)


if __name__ == "__main__":
    main()
