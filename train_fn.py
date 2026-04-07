import copy
import csv
import os
import time
import numpy as np
import torch
import torch.nn as nn
import json
from scipy.signal import savgol_filter
from utils import regression_metrics, format_seconds


def smooth_spectrum_array(predictions: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing along the spectrum dimension.
    predictions shape: (num_samples, spectrum_dim)
    """
    if window_length % 2 == 0:
        raise ValueError(f"window_length must be odd, got {window_length}")
    if polyorder >= window_length:
        raise ValueError(
            f"polyorder must be smaller than window_length, got polyorder={polyorder}, window_length={window_length}"
        )

    return savgol_filter(
        predictions,
        window_length=window_length,
        polyorder=polyorder,
        axis=1,
    )


@torch.no_grad()
def predict_model(model, loader, device):
    model.eval()

    predictions = []
    ground_truth = []

    for images, spectra in loader:
        images = images.to(device)
        spectra = spectra.to(device)

        outputs = model(images)

        predictions.append(outputs.detach().cpu().numpy())
        ground_truth.append(spectra.detach().cpu().numpy())

    predictions = np.vstack(predictions)
    ground_truth = np.vstack(ground_truth)
    return predictions, ground_truth


def run_one_epoch(model, loader, criterion, optimizer, device, train: bool = True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    preds = []
    targets = []

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for batch_idx, (images, spectra) in enumerate(loader):
            images = images.to(device)
            spectra = spectra.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, spectra)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds.append(outputs.detach().cpu().numpy())
            targets.append(spectra.detach().cpu().numpy())

    avg_loss = total_loss / max(1, len(loader))
    preds = np.vstack(preds)
    targets = np.vstack(targets)

    metrics = regression_metrics(targets, preds)
    metrics["loss"] = float(avg_loss)
    return metrics


def save_epoch_log_csv(csv_path, row_dict, write_header=False):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    logger,
    run_dir,
    num_epochs=100,
    patience=20,
    lr=1e-3,
    weight_decay=0.0,
):
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    early_stop_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_rmse": [],
        "val_rmse": [],
        "train_r2": [],
        "val_r2": [],
        "epoch_time_sec": [],
    }

    epoch_csv = os.path.join(run_dir, "epoch_metrics.csv")
    training_start = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=True,
        )

        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=False,
        )

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_rmse"].append(train_metrics["rmse"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["train_r2"].append(train_metrics["r2"])
        history["val_r2"].append(val_metrics["r2"])
        history["epoch_time_sec"].append(epoch_time)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_rmse": train_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "train_psnr": train_metrics["psnr"],
            "train_mae": train_metrics["mae"],
            "val_loss": val_metrics["loss"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "val_psnr": val_metrics["psnr"],
            "val_mae": val_metrics["mae"],
            "epoch_time_sec": epoch_time,
            "epoch_time_hms": format_seconds(epoch_time),
        }
        save_epoch_log_csv(epoch_csv, row, write_header=(epoch == 0))

        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {train_metrics['loss']:.6f} | Train RMSE: {train_metrics['rmse']:.6f} | "
            f"Val Loss: {val_metrics['loss']:.6f} | Val RMSE: {val_metrics['rmse']:.6f} | "
            f"Val R2: {val_metrics['r2']:.6f} | "
            f"Epoch Time: {format_seconds(epoch_time)}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            early_stop_counter = 0

            torch.save(best_state, os.path.join(checkpoint_dir, "best_model.pth"))
            logger.info(f"Best model updated at epoch {best_epoch}")
        else:
            early_stop_counter += 1

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            },
            os.path.join(checkpoint_dir, "last_checkpoint.pth"),
        )

        if early_stop_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    total_train_time = time.time() - training_start
    logger.info(f"Training finished in {format_seconds(total_train_time)}")

    if best_state is not None:
        model.load_state_dict(best_state)

    history_path = os.path.join(run_dir, "history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "training_time_sec": total_train_time,
        "training_time_hms": format_seconds(total_train_time),
    }


@torch.no_grad()
def evaluate_model(
    model,
    test_loader,
    device,
    apply_savgol: bool = False,
    savgol_window: int = 11,
    savgol_polyorder: int = 2,
):
    predictions, ground_truth = predict_model(model, test_loader, device)

    if apply_savgol:
        predictions = smooth_spectrum_array(
            predictions,
            window_length=savgol_window,
            polyorder=savgol_polyorder,
        )

    metrics = regression_metrics(ground_truth, predictions)
    metrics["loss"] = float(np.mean((ground_truth - predictions) ** 2))
    metrics["apply_savgol"] = bool(apply_savgol)
    metrics["savgol_window"] = int(savgol_window) if apply_savgol else None
    metrics["savgol_polyorder"] = int(savgol_polyorder) if apply_savgol else None

    return metrics

