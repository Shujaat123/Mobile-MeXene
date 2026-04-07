import logging
import os
from datetime import datetime
import json
import random
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
import csv
from collections import defaultdict


################## logger #########################

def create_log_directory(base_dir: str) -> str:
    os.makedirs(base_dir, exist_ok=True)

    existing_dirs = os.listdir(base_dir)
    log_numbers = []

    for d in existing_dirs:
        if d.startswith("log_"):
            try:
                num_str = d.split("_")[1]
                log_numbers.append(int(num_str))
            except (ValueError, IndexError):
                pass

    next_num = 1 if not log_numbers else max(log_numbers) + 1
    log_num = f"{next_num:03d}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    new_log_dir = os.path.join(base_dir, f"log_{log_num}_{timestamp}")
    os.makedirs(new_log_dir, exist_ok=True)
    return new_log_dir


def setup_logger(log_dir: str, filename: str = "training_log.txt", logger_name: str = "train_logger"):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(os.path.join(log_dir, filename))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def log_args(logger, args):
    logger.info("############### Args ##########################")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info("################################################")

#################### config ##########################

def save_config(path: str, cfg: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True, ensure_ascii=False)


def load_config(log_dir: str) -> dict:
    cfg_path = os.path.join(log_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.json not found in {log_dir}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

#################### seed ##########################

def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

#################### time_utils ##########################

def format_seconds(seconds: float) -> str:
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

#################### seed ##########################

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    mse = mean_squared_error(y_true, y_pred)
    max_pixel_value = np.max(y_true)
    psnr = float(20 * np.log10(max_pixel_value / np.sqrt(mse))) if mse > 0 else float("inf")

    mae = float(np.mean(np.abs(y_true - y_pred)))

    return {
        "rmse": rmse,
        "r2": r2,
        "psnr": psnr,
        "mae": mae,
    }

####################  ##########################
import csv
from collections import defaultdict


def append_csv_row(csv_path: str, row: dict, write_header: bool = False):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def compute_mean_std(values):
    values = np.array(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=0))


def aggregate_experiment_rows(rows, group_key="train_percent"):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[group_key]].append(row)

    aggregate_rows = []
    for key, group in grouped.items():
        agg_row = {
            group_key: key,
            "num_experiments": len(group),
        }

        metric_names = [
            "best_val_loss",
            "test_loss",
            "test_rmse",
            "test_r2",
            "test_psnr",
            "test_mae",
            "training_time_sec",
        ]

        for metric in metric_names:
            vals = [float(r[metric]) for r in group]
            mean_val, std_val = compute_mean_std(vals)
            agg_row[f"{metric}_mean"] = mean_val
            agg_row[f"{metric}_std"] = std_val

        aggregate_rows.append(agg_row)

    return aggregate_rows

####################  ##########################

def get_valid_savgol_grid(windows, polyorders):
    grid = []
    for w in windows:
        for d in polyorders:
            if w % 2 == 1 and d < w:
                grid.append((w, d))
    return grid    


def aggregate_experiment_rows_multi(rows, group_keys):
    grouped = defaultdict(list)

    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped[key].append(row)

    aggregate_rows = []
    for key, group in grouped.items():
        agg_row = {k: v for k, v in zip(group_keys, key)}
        agg_row["num_experiments"] = len(group)

        metric_names = [
            "best_val_loss",
            "test_loss",
            "test_rmse",
            "test_r2",
            "test_psnr",
            "test_mae",
            "training_time_sec",
        ]

        for metric in metric_names:
            vals = [float(r[metric]) for r in group]
            vals = np.array(vals, dtype=np.float64)
            agg_row[f"{metric}_mean"] = float(vals.mean())
            agg_row[f"{metric}_std"] = float(vals.std(ddof=0))

        aggregate_rows.append(agg_row)

    return aggregate_rows

#################### seed ##########################


#################### seed ##########################


