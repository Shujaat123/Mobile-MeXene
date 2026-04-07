import argparse
import os

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from build_data import ImageSpectrumDataset, build_subsets_for_experiment, make_run_splits
from train_fn import evaluate_model
from build_model import build_model
from utils import load_config, setup_logger, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MXene spectral prediction model")
    parser.add_argument("--log_dir", type=str, required=True, help="Experiment log directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def evaluate_from_config(log_dir: str, device=None):
    cfg = load_config(log_dir)

    class Args:
        pass

    args = Args()
    for k, v in cfg.items():
        setattr(args, k, v)

    logger = setup_logger(log_dir, filename="evaluation_log.txt", logger_name="eval_logger")

    if device is None:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    dataset = ImageSpectrumDataset(
        image_dir=args.image_dir,
        spectrum_file=args.spectrum_file,
        transform=transform,
    )

    splits = make_run_splits(
        dataset_size=len(dataset),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.run_seed,
    )

    _, _, test_dataset, _ = build_subsets_for_experiment(
        dataset=dataset,
        splits=splits,
        train_percent=args.train_percent,
        subset_seed=args.subset_seed,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args).to(device)

    checkpoint_path = os.path.join(log_dir, "checkpoints", "best_model.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    test_metrics = evaluate_model(model, test_loader, device)
    logger.info(f"Test metrics: {test_metrics}")

    return test_metrics


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    _ = evaluate_from_config(args.log_dir, device=device)


if __name__ == "__main__":
    main()