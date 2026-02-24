"""Tests for supervoxel-based graph construction."""

import numpy as np
import pytest
import torch

from src.graph_construction import NUM_NODE_FEATURES, build_graph_from_volume


def _random_volume(shape=(32, 32, 32), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, shape).astype(np.float32)


class TestBuildGraphFromVolume:
    def test_returns_data_object(self):
        from torch_geometric.data import Data

        vol = _random_volume()
        graph = build_graph_from_volume(vol, n_segments=50, k_neighbors=5)
        assert isinstance(graph, Data)

    def test_node_feature_dimension(self):
        vol = _random_volume()
        graph = build_graph_from_volume(vol, n_segments=50, k_neighbors=5)
        assert graph.x.shape[1] == NUM_NODE_FEATURES

    def test_edge_index_shape(self):
        vol = _random_volume()
        graph = build_graph_from_volume(vol, n_segments=50, k_neighbors=5)
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_index.dtype == torch.long

    def test_edge_index_values_in_range(self):
        vol = _random_volume()
        graph = build_graph_from_volume(vol, n_segments=50, k_neighbors=5)
        n = graph.num_nodes
        assert int(graph.edge_index.max()) < n
        assert int(graph.edge_index.min()) >= 0

    def test_label_attached_when_provided(self):
        vol = _random_volume()
        graph = build_graph_from_volume(vol, n_segments=50, k_neighbors=5, label=1)
        assert hasattr(graph, "y")
        assert int(graph.y.item()) == 1

    def test_no_label_when_none(self):
        vol = _random_volume()
        graph = build_graph_from_volume(vol, n_segments=50, k_neighbors=5, label=None)
        assert not hasattr(graph, "y") or graph.y is None

    def test_num_nodes_positive(self):
        vol = _random_volume()
        graph = build_graph_from_volume(vol, n_segments=50, k_neighbors=5)
        assert graph.num_nodes > 0

    def test_k_neighbors_clipped_for_small_volumes(self):
        """When volume produces fewer nodes than k_neighbors, no error raised."""
        vol = _random_volume(shape=(8, 8, 8))
        graph = build_graph_from_volume(vol, n_segments=10, k_neighbors=20)
        assert graph.num_nodes > 0

    def test_num_node_features_constant(self):
        assert NUM_NODE_FEATURES == 6
