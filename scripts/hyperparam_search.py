"""
Hyperparameter grid search runner for this repo.

- Logs results to JSONL (search_outputs/results.jsonl).
- Saves model weights per run and keeps best weights as best_generator_search.pth.
- Supports resume and basic parallelism (default n_jobs=1 to avoid GPU contention).
Adjust import path for train_and_evaluate if needed.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def train_and_evaluate_wrapper(config: Dict[str, Any], data_root: str, device: str):
    """Import and call the package-level train_and_evaluate implementation.

    This prefers `mrisyngan.train.train_and_evaluate` which is the canonical
    implementation (added to make the function importable from scripts).
    """
    try:
        from mrisyngan.train import train_and_evaluate as real_fn
    except Exception as e:
        raise ImportError(
            "Could not import mrisyngan.train.train_and_evaluate. Ensure the package is installed or the PYTHONPATH is correct."
        ) from e
    return real_fn(config, data_root, device)


def run_one(config: Dict[str, Any], data_root: str, device: str, out_dir: Path):
    started = time.time()
    try:
        score, model_state = train_and_evaluate_wrapper(config, data_root, device)
        if not isinstance(score, (int, float)):
            raise ValueError("train_and_evaluate must return a numeric score as first element")
        ts = int(time.time())
        weights_path = out_dir / f"gen_{ts}_{abs(hash(str(config)))%100000}.pth"
        torch.save(model_state, str(weights_path))
        return {
            "config": config,
            "score": float(score),
            "duration_s": time.time() - started,
            "weights": str(weights_path),
        }, None
    except Exception as e:
        return None, str(e)


def load_tested_configs(results_file: Path):
    seen = set()
    if not results_file.exists():
        return seen
    with results_file.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
                seen.add(json.dumps(j.get("config", {}), sort_keys=True))
            except Exception:
                continue
    return seen


def main():
    parser = argparse.ArgumentParser(description="Grid search wrapper for train_and_evaluate")
    parser.add_argument("--data_root", type=str, default="E:/PATTERN/PATIENTS0")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--out_dir", type=Path, default=Path("search_outputs"))
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel jobs (use 1 for GPU stability)")
    parser.add_argument("--resume", action="store_true", help="Skip configs already recorded in results.jsonl")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_file = args.out_dir / "results.jsonl"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Validate data_root path early to fail fast with clear message
    if not args.data_root or not os.path.isdir(args.data_root):
        logging.error("ERROR: --data_root must be set to an existing directory. Provided: %s", args.data_root)
        raise SystemExit(1)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    param_grid = {
        "lr": [0.0001, 0.00005, 0.0002],
        "latent_dim": [64, 128],
        "n_critic": [5, 10, 20],
    }
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    tested = load_tested_configs(results_file) if args.resume else set()
    combos_to_run = [c for c in combinations if json.dumps(c, sort_keys=True) not in tested]

    logging.info("Starting search: %d total, %d to run", len(combinations), len(combos_to_run))

    best_score = float("inf")
    best_config = None
    best_path = None

    with results_file.open("a", encoding="utf-8") as rf, ThreadPoolExecutor(max_workers=max(1, args.n_jobs)) as ex:
        futures = {ex.submit(run_one, cfg, args.data_root, args.device, args.out_dir): cfg for cfg in combos_to_run}
        for fut in as_completed(futures):
            cfg = futures[fut]
            res, err = fut.result()
            if err:
                logging.error("Error for %s: %s", cfg, err)
                continue
            rf.write(json.dumps(res) + "\n")
            rf.flush()
            logging.info("Done: score=%.6f cfg=%s", res["score"], res["config"])
            if res["score"] < best_score:
                best_score = res["score"]
                best_config = res["config"]
                best_path = Path(res["weights"])
                # Save a copy as best_generator_search.pth
                torch.save(torch.load(str(best_path)), str(args.out_dir / "best_generator_search.pth"))

    logging.info("Search completed. Best score=%.6f config=%s weights=%s", best_score, best_config, best_path)


if __name__ == "__main__":
    main()
