"""
CLI diagnostic inference tool for brain tumor detection using a trained
BrainTumorGCN model on low-quality 3T MRI images.

Usage
-----
    python -m src.predict \\
        --model  checkpoints/best_model.pt \\
        --input  patient_scan.nii.gz \\
        [--hidden_channels 64] \\
        [--out_channels 128] \\
        [--n_segments 500] \\
        [--k_neighbors 10] \\
        [--threshold 0.5]
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from .gcn_model import BrainTumorGCN
from .graph_construction import NUM_NODE_FEATURES, build_graph_from_volume
from .preprocessing import preprocess_3t_mri


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Diagnose brain tumor presence from a 3T MRI scan."
    )
    p.add_argument("--model", required=True, help="Path to a saved model checkpoint (.pt).")
    p.add_argument("--input", required=True, help="Path to the NIfTI MRI file to analyse.")
    p.add_argument("--hidden_channels", type=int, default=64)
    p.add_argument("--out_channels", type=int, default=128)
    p.add_argument("--n_segments", type=int, default=500)
    p.add_argument("--k_neighbors", type=int, default=10)
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold above which a tumor is reported (default 0.5).",
    )
    return p


def load_model(
    checkpoint_path: str,
    hidden_channels: int,
    out_channels: int,
    device: torch.device,
) -> BrainTumorGCN:
    """Load a BrainTumorGCN from a saved checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the .pt file saved by the training script.
    hidden_channels : int
        Must match the value used during training.
    out_channels : int
        Must match the value used during training.
    device : torch.device
        Target device.

    Returns
    -------
    BrainTumorGCN
        Model loaded in evaluation mode.

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    model = BrainTumorGCN(
        in_channels=NUM_NODE_FEATURES,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
    ).to(device)
    model.load_state_dict(torch.load(str(path), map_location=device))
    model.eval()
    return model


@torch.no_grad()
def predict(
    model: BrainTumorGCN,
    mri_path: str,
    n_segments: int = 500,
    k_neighbors: int = 10,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Run inference on a single MRI scan and return a diagnostic report.

    Parameters
    ----------
    model : BrainTumorGCN
        Loaded model in evaluation mode.
    mri_path : str
        Path to the NIfTI MRI file.
    n_segments : int
        SLIC supervoxel count (default 500).
    k_neighbors : int
        k-NN neighbours for graph edges (default 10).
    device : torch.device
        Target device for inference.

    Returns
    -------
    dict
        ``{tumor_probability: float, predicted_class: int,
           diagnosis: str}``
    """
    volume, _ = preprocess_3t_mri(mri_path)
    graph = build_graph_from_volume(volume, n_segments=n_segments, k_neighbors=k_neighbors)
    graph = graph.to(device)

    # Add a trivial batch vector (single graph)
    batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
    logits = model(graph.x, graph.edge_index, batch)
    probs = F.softmax(logits, dim=1)

    tumor_prob = float(probs[0, 1].item())
    predicted_class = int(probs[0].argmax().item())
    diagnosis = "TUMOR DETECTED" if predicted_class == 1 else "NO TUMOR DETECTED"

    return {
        "tumor_probability": tumor_prob,
        "predicted_class": predicted_class,
        "diagnosis": diagnosis,
    }


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = load_model(
            args.model,
            hidden_channels=args.hidden_channels,
            out_channels=args.out_channels,
            device=device,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        result = predict(
            model,
            mri_path=args.input,
            n_segments=args.n_segments,
            k_neighbors=args.k_neighbors,
            device=device,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("=" * 45)
    print("  3T MRI BRAIN TUMOR DIAGNOSTIC REPORT")
    print("=" * 45)
    print(f"  Input scan      : {args.input}")
    print(f"  Tumor prob.    : {result['tumor_probability']:.4f}")
    print(f"  Threshold       : {args.threshold:.2f}")
    print(f"  Diagnosis       : {result['diagnosis']}")
    print("=" * 45)

    # Exit with code 1 when above threshold so the tool can be piped / scripted
    if result["tumor_probability"] >= args.threshold:
        sys.exit(1)


if __name__ == "__main__":
    main()
