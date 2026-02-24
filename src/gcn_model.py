"""
GCN model architecture for brain tumor detection in 3T MRI images.

The BrainTumorGCN uses three graph convolutional layers with batch
normalisation, followed by global pooling and a multi-layer classifier
head to produce per-scan tumor/non-tumor predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class BrainTumorGCN(nn.Module):
    """Graph Convolutional Network for brain tumor detection.

    Parameters
    ----------
    in_channels : int
        Number of input node features.
    hidden_channels : int
        Number of hidden channels in the first GCN layer.
    out_channels : int
        Number of output channels from the final GCN layer.
    num_classes : int
        Number of output classes (default 2: no-tumor / tumor).
    dropout : float
        Dropout probability used during training (default 0.5).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_classes: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels * 2)
        self.conv3 = GCNConv(hidden_channels * 2, out_channels)

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 2)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.classifier = nn.Sequential(
            nn.Linear(out_channels * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor, shape (N, in_channels)
            Node feature matrix.
        edge_index : Tensor, shape (2, E)
            Graph connectivity in COO format.
        batch : Tensor, shape (N,)
            Batch vector mapping each node to its graph.

        Returns
        -------
        Tensor, shape (B, num_classes)
            Class logits for each graph in the batch.
        """
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.relu(self.bn3(self.conv3(x, edge_index)))

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        return self.classifier(x)
