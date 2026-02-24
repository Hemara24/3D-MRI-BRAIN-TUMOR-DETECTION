"""
Training pipeline for the BrainTumorGCN model.

Usage
-----
    python -m src.train \\
        --manifest  data/train.csv \\
        --val_manifest data/val.csv \\
        --epochs 50 \\
        --batch_size 16 \\
        --hidden_channels 64 \\
        --out_channels 128 \\
        --lr 1e-3 \\
        --output_dir checkpoints/
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from .dataset import MRITumorDataset
from .evaluate import compute_metrics, print_metrics
from .gcn_model import BrainTumorGCN
from .graph_construction import NUM_NODE_FEATURES


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train BrainTumorGCN on 3T MRI data.")
    p.add_argument("--manifest", required=True, help="Path to training manifest CSV.")
    p.add_argument("--val_manifest", required=True, help="Path to validation manifest CSV.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--hidden_channels", type=int, default=64)
    p.add_argument("--out_channels", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--n_segments", type=int, default=500)
    p.add_argument("--k_neighbors", type=int, default=10)
    p.add_argument("--patience", type=int, default=10, help="Early-stopping patience.")
    p.add_argument(
        "--output_dir",
        default="checkpoints",
        help="Directory to save model checkpoints.",
    )
    return p


def train_one_epoch(
    model: BrainTumorGCN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch and return mean loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def evaluate(
    model: BrainTumorGCN,
    loader: DataLoader,
    device: torch.device,
):
    """Return (y_true, y_pred, y_prob) lists for the whole loader."""
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        y_true.extend(batch.y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        y_prob.extend(probs[:, 1].cpu().tolist())
    return y_true, y_pred, y_prob


def train(args: argparse.Namespace) -> BrainTumorGCN:
    """Full training loop with validation and early stopping.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments (see :func:`_build_arg_parser`).

    Returns
    -------
    BrainTumorGCN
        The best model (lowest validation loss) found during training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = MRITumorDataset(
        args.manifest,
        n_segments=args.n_segments,
        k_neighbors=args.k_neighbors,
    )
    val_dataset = MRITumorDataset(
        args.val_manifest,
        n_segments=args.n_segments,
        k_neighbors=args.k_neighbors,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = BrainTumorGCN(
        in_channels=NUM_NODE_FEATURES,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = output_dir / "best_model.pt"

    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        y_true, y_pred, y_prob = evaluate(model, val_loader, device)

        val_loss = nn.CrossEntropyLoss()(
            torch.tensor([[1 - p, p] for p in y_prob]),
            torch.tensor(y_true),
        ).item()

        metrics = compute_metrics(y_true, y_pred, y_prob)
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} acc={metrics['accuracy']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), best_checkpoint)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    print("\nBest validation metrics:")
    model.load_state_dict(torch.load(best_checkpoint, map_location=device))
    y_true, y_pred, y_prob = evaluate(model, val_loader, device)
    print_metrics(compute_metrics(y_true, y_pred, y_prob))

    return model


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
