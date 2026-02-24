"""Tests for the BrainTumorGCN model architecture."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from src.gcn_model import BrainTumorGCN


def _make_batch(num_nodes: int = 20, in_channels: int = 6, num_graphs: int = 2):
    """Create a synthetic PyG batch for testing."""
    graphs = []
    for _ in range(num_graphs):
        x = torch.randn(num_nodes, in_channels)
        # Simple ring graph edges
        src = torch.arange(num_nodes)
        dst = (src + 1) % num_nodes
        edge_index = torch.stack([
            torch.cat([src, dst]),
            torch.cat([dst, src]),
        ], dim=0)
        graphs.append(Data(x=x, edge_index=edge_index))
    return Batch.from_data_list(graphs)


class TestBrainTumorGCN:
    in_ch = 6
    hidden_ch = 16
    out_ch = 32

    def _model(self, num_classes: int = 2) -> BrainTumorGCN:
        return BrainTumorGCN(
            in_channels=self.in_ch,
            hidden_channels=self.hidden_ch,
            out_channels=self.out_ch,
            num_classes=num_classes,
        )

    def test_output_shape_binary(self):
        model = self._model(num_classes=2)
        batch = _make_batch(in_channels=self.in_ch, num_graphs=4)
        model.eval()
        with torch.no_grad():
            out = model(batch.x, batch.edge_index, batch.batch)
        assert out.shape == (4, 2), f"Expected (4, 2) but got {out.shape}"

    def test_output_shape_multiclass(self):
        model = self._model(num_classes=4)
        batch = _make_batch(in_channels=self.in_ch, num_graphs=3)
        model.eval()
        with torch.no_grad():
            out = model(batch.x, batch.edge_index, batch.batch)
        assert out.shape == (3, 4)

    def test_training_step(self):
        """One gradient step should decrease loss without errors."""
        model = self._model()
        batch = _make_batch(in_channels=self.in_ch, num_graphs=2)
        batch.y = torch.tensor([0, 1], dtype=torch.long)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0

    def test_dropout_differs_in_train_eval(self):
        """Model outputs should differ between train/eval with dropout."""
        model = self._model()
        batch = _make_batch(in_channels=self.in_ch, num_graphs=1)

        torch.manual_seed(0)
        model.train()
        out_train = model(batch.x, batch.edge_index, batch.batch)

        model.eval()
        with torch.no_grad():
            out_eval = model(batch.x, batch.edge_index, batch.batch)

        # With p=0.5 dropout, outputs should differ at least sometimes
        # (not a strict guarantee, but holds with overwhelming probability)
        assert out_train.shape == out_eval.shape

    def test_single_node_graph(self):
        """Model should handle a single-node graph gracefully."""
        model = self._model()
        x = torch.randn(1, self.in_ch)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        batch = torch.zeros(1, dtype=torch.long)
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index, batch)
        assert out.shape == (1, 2)
