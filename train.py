import argparse
import csv
import os
from datetime import datetime
import json
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from build_data import ImageSpectrumDataset, build_subsets_for_experiment, make_run_splits
from train_fn import evaluate_model, train_model
from build_model import build_model
from utils import save_config, create_log_directory, log_args, setup_logger, seed_worker, set_seed, append_csv_row, aggregate_experiment_rows, get_valid_savgol_grid, aggregate_experiment_rows_multi

def parse_args():
    parser = argparse.ArgumentParser(description="Train MXene spectral prediction models")

    # data
    parser.add_argument("--image_dir", type=str, default="/data/muhammad_jabbar/mxene_forward/data_forward/MXene-Absorbers-DNN-main/images")
    parser.add_argument("--spectrum_file", type=str, default="/data/muhammad_jabbar/mxene_forward/data_forward/MXene-Absorbers-DNN-main/Absorption spectra.xlsx")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    # split
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument(
        "--train_percents",
        type=float,
        nargs="+",
        default=[1.0],
        help="Percentages of train pool to use, e.g. 1.0 0.8 0.6",
    )

    # experiment repetition
    parser.add_argument("--runs", type=int, default=10) # number of repeated runs with different data splits and seeds
    parser.add_argument("--seed", type=int, default=4341)
    parser.add_argument("--seed_increment", type=int, default=2)

    # model
    parser.add_argument("--model_name", type=str, default="convstack",
        choices=["conv1d0", "convstack", "pred_mxene_cnn", "pred_mxene_deformable_cnn"])

    # ONLY for conv1d0 and convstack models
    # {{{
    parser.add_argument("--backbone_name", type=str, default="mobilenet_v2",
        choices=["mobilenet_v2",
        "vgg11_bn", "vgg16_bn", "resnet18", "convnext_tiny", "vit_b_16",
        "tiny_cnn", "mlp_baseline"]) 
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--output_dim", type=int, default=102)

    # Train
    parser.add_argument("--use_mcsr", type=bool, default=True)
    # }}}

    # evaluation-time smoothing
    parser.add_argument("--eval_raw", type=bool, default=True)
    parser.add_argument("--eval_with_savgol", type=bool, default=True)
    parser.add_argument("--eval_savgol_windows", type=int, nargs="+", default=[11])
    parser.add_argument("--eval_savgol_polyorders", type=int, nargs="+", default=[2])

    # optimization
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # system
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--base_log_dir", type=str, default="logs")

    parser.add_argument("--exp_string", type=str, default="ablations_pred_mxene")
    parser.add_argument("--exp_description", type=str, default="check args")

    return parser.parse_args()



def main():
    args = parse_args()

    base_dir = os.path.join(args.base_log_dir, args.model_name, args.exp_string, args.backbone_name)
    log_dir = create_log_directory(base_dir)
    args.log_dir = log_dir
    # args.checkpoint_dir = os.path.join(log_dir, "checkpoints")
    # os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.device = str(device)

    logger = setup_logger(log_dir=log_dir)
    log_args(logger, args)

    cfg = dict(vars(args))
    cfg.update({
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })
    save_config(os.path.join(log_dir, "config.json"), cfg)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    dataset = ImageSpectrumDataset(
        image_dir=args.image_dir,
        spectrum_file=args.spectrum_file,
        transform=transform,
    )

    logger.info(f"Dataset size: {len(dataset)}")

    summary_csv = os.path.join(log_dir, "all_runs_summary.csv")
    first_summary_row = True
    all_summary_rows = []

    smoothing_summary_csv = os.path.join(log_dir, "all_runs_smoothing_summary.csv")
    first_smoothing_row = True
    all_smoothing_rows = []

    for run_idx in range(args.runs):
        run_seed = args.seed + (args.seed_increment * run_idx)
        set_seed(run_seed)

        logger.info("=" * 80)
        logger.info(f"Starting run {run_idx + 1}/{args.runs} with run_seed={run_seed}")

        splits = make_run_splits(
            dataset_size=len(dataset),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=run_seed,
        )

        logger.info(
            f"Base split for run {run_idx + 1}: "
            f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
        )

        run_dir = os.path.join(log_dir, f"run_{run_idx+1:02d}")
        os.makedirs(run_dir, exist_ok=True)
        run_split_dir = os.path.join(run_dir, "base_split")
        os.makedirs(run_split_dir, exist_ok=True)

        np.save(os.path.join(run_split_dir, "train_indices.npy"), splits["train"])
        np.save(os.path.join(run_split_dir, "val_indices.npy"), splits["val"])
        np.save(os.path.join(run_split_dir, "test_indices.npy"), splits["test"])

        with open(os.path.join(run_split_dir, "split_info.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_idx": run_idx + 1,
                    "run_seed": run_seed,
                    "train_size": int(len(splits["train"])),
                    "val_size": int(len(splits["val"])),
                    "test_size": int(len(splits["test"])),
                },
                f,
                indent=2,
            )

        for percent_idx, train_percent in enumerate(args.train_percents):
            subset_seed = run_seed

            exp_name = f"run_{run_idx+1:02d}_trainpct_{int(train_percent*100):03d}"
            exp_dir = os.path.join(run_dir, f"trainpct_{int(train_percent*100):03d}")
            os.makedirs(exp_dir, exist_ok=True)

            generator = torch.Generator()
            generator.manual_seed(run_seed)

            train_dataset, val_dataset, test_dataset, train_subset_idx = build_subsets_for_experiment(
                dataset=dataset,
                splits=splits,
                train_percent=train_percent,
                subset_seed=subset_seed,
            )

            np.save(os.path.join(exp_dir, "train_subset_indices.npy"), train_subset_idx)
            logger.info("-" * 80)
            logger.info(
                f"Experiment: {exp_name} | run_seed={run_seed} | subset_seed={subset_seed} | train_percent={train_percent}"
            )
            logger.info(
                f"Subset sizes -> train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}"
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=generator,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )

            model = build_model(args).to(device)

            num_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model: {model.__class__.__name__} | Num params: {num_params}")

            exp_cfg = dict(cfg)
            exp_cfg.update({
                "run_idx": run_idx + 1,
                "run_seed": run_seed,
                "subset_seed": subset_seed,
                "train_percent": train_percent,
                "experiment_dir": exp_dir,
                "model_class_name": model.__class__.__name__,
                "num_params": num_params,
            })
            save_config(os.path.join(exp_dir, "config.json"), exp_cfg)

            train_out = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                logger=logger,
                run_dir=exp_dir,
                num_epochs=args.epochs,
                patience=args.patience,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

            trained_model = train_out["model"]

            # ---------------------------------------------------
            # Raw evaluation
            # ---------------------------------------------------
            raw_test_metrics = evaluate_model(
                model=trained_model,
                test_loader=test_loader,
                device=device,
                apply_savgol=False,
            )

            serializable_raw_test_metrics = {
                k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in raw_test_metrics.items()
            }
            with open(os.path.join(exp_dir, "test_metrics_raw.json"), "w", encoding="utf-8") as f:
                json.dump(serializable_raw_test_metrics, f, indent=2)

            logger.info(f"Raw test metrics for {exp_name}: {raw_test_metrics}")

            summary_row = {
                "run_idx": run_idx + 1,
                "run_seed": run_seed,
                "subset_seed": subset_seed,
                "train_percent": train_percent,
                "model_name": args.model_name,
                "backbone_name": getattr(args, "backbone_name", None),
                "num_params": num_params,
                "best_epoch": train_out["best_epoch"],
                "best_val_loss": train_out["best_val_loss"],
                "test_loss": raw_test_metrics["loss"],
                "test_rmse": raw_test_metrics["rmse"],
                "test_r2": raw_test_metrics["r2"],
                "test_psnr": raw_test_metrics["psnr"],
                "test_mae": raw_test_metrics["mae"],
                "training_time_sec": train_out["training_time_sec"],
                "training_time_hms": train_out["training_time_hms"],
                "experiment_dir": exp_dir,
                "eval_mode": "raw",
            }
            append_csv_row(summary_csv, summary_row, write_header=first_summary_row)
            first_summary_row = False
            all_summary_rows.append(summary_row)

            # ---------------------------------------------------
            # Savitzky-Golay smoothing evaluation grid
            # ---------------------------------------------------
            per_exp_savgol_csv = os.path.join(exp_dir, "test_metrics_savgol_grid.csv")
            first_exp_savgol_row = True

            if args.eval_with_savgol:
                savgol_grid = get_valid_savgol_grid(
                    args.eval_savgol_windows,
                    args.eval_savgol_polyorders,
                )

                for w, d in savgol_grid:
                    smoothed_test_metrics = evaluate_model(
                        model=trained_model,
                        test_loader=test_loader,
                        device=device,
                        apply_savgol=True,
                        savgol_window=w,
                        savgol_polyorder=d,
                    )

                    logger.info(
                        f"Savgol test metrics for {exp_name} | w={w}, d={d}: {smoothed_test_metrics}"
                    )

                    smoothing_row = {
                        "run_idx": run_idx + 1,
                        "run_seed": run_seed,
                        "subset_seed": subset_seed,
                        "train_percent": train_percent,
                        "model_name": args.model_name,
                        "backbone_name": getattr(args, "backbone_name", None),
                        "num_params": num_params,
                        "best_epoch": train_out["best_epoch"],
                        "best_val_loss": train_out["best_val_loss"],
                        "test_loss": smoothed_test_metrics["loss"],
                        "test_rmse": smoothed_test_metrics["rmse"],
                        "test_r2": smoothed_test_metrics["r2"],
                        "test_psnr": smoothed_test_metrics["psnr"],
                        "test_mae": smoothed_test_metrics["mae"],
                        "training_time_sec": train_out["training_time_sec"],
                        "training_time_hms": train_out["training_time_hms"],
                        "savgol_window": w,
                        "savgol_polyorder": d,
                        "experiment_dir": exp_dir,
                        "eval_mode": "savgol",
                    }

                    append_csv_row(
                        per_exp_savgol_csv,
                        smoothing_row,
                        write_header=first_exp_savgol_row,
                    )
                    first_exp_savgol_row = False

                    append_csv_row(
                        smoothing_summary_csv,
                        smoothing_row,
                        write_header=first_smoothing_row,
                    )
                    first_smoothing_row = False

                    all_smoothing_rows.append(smoothing_row)

    logger.info("All runs completed.")
    logger.info("-"*40)

    aggregate_rows = aggregate_experiment_rows(all_summary_rows, group_key="train_percent")
    aggregate_csv = os.path.join(log_dir, "aggregate_metrics.csv")

    for idx, row in enumerate(aggregate_rows):
        append_csv_row(aggregate_csv, row, write_header=(idx == 0))

    # logger.info(f"Raw test metrics for {exp_name}: {raw_test_metrics}")
    logger.info("Aggregated raw metrics saved.")


    if len(all_smoothing_rows) > 0:
        smoothing_aggregate_rows = aggregate_experiment_rows_multi(
            all_smoothing_rows,
            group_keys=["train_percent", "savgol_window", "savgol_polyorder"],
        )
        smoothing_aggregate_csv = os.path.join(log_dir, "aggregate_savgol_metrics.csv")

        for idx, row in enumerate(smoothing_aggregate_rows):
            append_csv_row(smoothing_aggregate_csv, row, write_header=(idx == 0))

        logger.info("Aggregated Savitzky-Golay metrics saved.")

if __name__ == "__main__":
    main()