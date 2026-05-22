from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from binary_yolo.train_fruits import collect_search_results, search_configs


def parse_args():
    ap = argparse.ArgumentParser(description="Run binary YOLO grid-search configs in parallel, one process per GPU.")
    ap.add_argument("--run-dir", default="runs/binary_yolo_grid32_seed7")
    ap.add_argument("--input-size", type=int, default=256)
    ap.add_argument("--dataset", default="fruits_detection")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-overfit", action="store_true", default=True)
    ap.add_argument("--gpus", default="")
    ap.add_argument("--search-set", default="default", choices=["default", "long_lr", "long_exp", "long_plateau", "long_model"])
    return ap.parse_args()


def gpu_ids(arg: str) -> list[str]:
    if arg:
        return [item.strip() for item in arg.split(",") if item.strip()]
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        text=True,
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def completed(root: Path) -> set[str]:
    return {path.parent.name for path in root.glob("*/summary.json")}


def launch_config(name: str, gpu_id: str, args) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    cmd = [
        "python3",
        "-m",
        "binary_yolo.train_fruits",
        "--search",
        "--search-set",
        args.search_set,
        "--config-name",
        name,
        "--run-dir",
        args.run_dir,
        "--dataset",
        args.dataset,
        "--input-size",
        str(args.input_size),
        "--val-fraction",
        str(args.val_fraction),
        "--seed",
        str(args.seed),
        "--skip-overfit",
    ]
    log_path = Path(args.run_dir) / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = log_path.open("w")
    return subprocess.Popen(cmd, cwd=Path(__file__).resolve().parents[1], env=env, stdout=log_fp, stderr=subprocess.STDOUT)


def main():
    args = parse_args()
    root = Path(args.run_dir)
    root.mkdir(parents=True, exist_ok=True)
    todo = [name for name, _, _ in search_configs(args.input_size, args.search_set) if name not in completed(root)]
    free_gpus = gpu_ids(args.gpus)
    running: dict[str, tuple[str, subprocess.Popen]] = {}

    while todo or running:
        while todo and free_gpus:
            gpu_id = free_gpus.pop(0)
            name = todo.pop(0)
            proc = launch_config(name, gpu_id, args)
            running[gpu_id] = (name, proc)
            print(f"launched {name} on gpu {gpu_id}")
        time.sleep(10)
        finished = []
        for gpu_id, (name, proc) in running.items():
            code = proc.poll()
            if code is None:
                continue
            if code != 0:
                raise SystemExit(f"{name} failed on gpu {gpu_id} with exit code {code}")
            finished.append(gpu_id)
            print(f"finished {name} on gpu {gpu_id}")
        for gpu_id in finished:
            del running[gpu_id]
            free_gpus.append(gpu_id)

    rows = collect_search_results(root)
    print(json.dumps(rows[0], indent=2, sort_keys=True) if rows else "no results")


if __name__ == "__main__":
    main()
