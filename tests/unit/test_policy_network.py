"""Unit tests for PolicyNetwork and CellNN classes."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from captain.agents.policy_network import CellNN, PolicyNetwork


class TestCellNNInit:
    """Tests for CellNN initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        model = CellNN(input_dim=10, hidden_dim=16)

        assert model.input_dim == 10
        assert model.hidden_dim == 16

    def test_init_relu_activation(self):
        """Test ReLU activation."""
        model = CellNN(input_dim=10, hidden_dim=16, activation="relu")

        # Should not raise
        x = torch.randn(10, 100)
        output = model(x)
        assert output.shape == (100,)

    def test_init_tanh_activation(self):
        """Test tanh activation."""
        model = CellNN(input_dim=10, hidden_dim=16, activation="tanh")

        x = torch.randn(10, 100)
        output = model(x)
        assert output.shape == (100,)

    def test_init_gelu_activation(self):
        """Test GELU activation."""
        model = CellNN(input_dim=10, hidden_dim=16, activation="gelu")

        x = torch.randn(10, 100)
        output = model(x)
        assert output.shape == (100,)

    def test_init_invalid_activation(self):
        """Test invalid activation raises error."""
        with pytest.raises(ValueError, match="Unknown activation"):
            CellNN(input_dim=10, hidden_dim=16, activation="invalid")


class TestCellNNForward:
    """Tests for CellNN forward pass."""

    def test_forward_shape(self):
        """Test forward pass output shape."""
        model = CellNN(input_dim=13, hidden_dim=32)
        x = torch.randn(13, 1000)

        output = model(x)

        assert output.shape == (1000,)

    def test_forward_deterministic(self):
        """Test forward pass is deterministic."""
        model = CellNN(input_dim=10, hidden_dim=16)
        model.eval()

        x = torch.randn(10, 100)
        output1 = model(x)
        output2 = model(x)

        assert torch.allclose(output1, output2)

    def test_forward_different_batch_sizes(self):
        """Test forward works with different batch sizes."""
        model = CellNN(input_dim=10, hidden_dim=16)

        for n_cells in [1, 10, 100, 1000]:
            x = torch.randn(10, n_cells)
            output = model(x)
            assert output.shape == (n_cells,)


class TestPolicyNetworkInit:
    """Tests for PolicyNetwork initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        model = CellNN(input_dim=10, hidden_dim=16)
        policy = PolicyNetwork(model)

        assert policy.model is model
        assert policy.device == torch.device("cpu")

    def test_init_with_device(self):
        """Test initialization with device."""
        model = CellNN(input_dim=10, hidden_dim=16)
        policy = PolicyNetwork(model, device="cpu")

        assert policy.device == torch.device("cpu")


class TestPolicyNetworkWeights:
    """Tests for weight management methods."""

    def test_get_flat_weights(self):
        """Test get_flat_weights returns 1D array."""
        model = CellNN(input_dim=10, hidden_dim=16)
        policy = PolicyNetwork(model)

        weights = policy.get_flat_weights()

        assert isinstance(weights, np.ndarray)
        assert weights.ndim == 1
        assert len(weights) == policy.get_n_params()

    def test_set_flat_weights_numpy(self):
        """Test set_flat_weights with numpy array."""
        model = CellNN(input_dim=10, hidden_dim=16)
        policy = PolicyNetwork(model)

        original_weights = policy.get_flat_weights()
        new_weights = np.zeros_like(original_weights)

        policy.set_flat_weights(new_weights)

        assert np.allclose(policy.get_flat_weights(), new_weights)

    def test_set_flat_weights_torch(self):
        """Test set_flat_weights with torch tensor."""
        model = CellNN(input_dim=10, hidden_dim=16)
        policy = PolicyNetwork(model)

        n_params = policy.get_n_params()
        new_weights = torch.ones(n_params)

        policy.set_flat_weights(new_weights)

        assert np.allclose(policy.get_flat_weights(), new_weights.numpy())

    def test_get_n_params(self):
        """Test get_n_params returns correct count."""
        model = CellNN(input_dim=10, hidden_dim=16)
        policy = PolicyNetwork(model)

        n_params = policy.get_n_params()

        # Linear(10, 16) = 10*16 + 16 = 176
        # Linear(16, 1) = 16*1 + 1 = 17
        # Total = 193
        assert n_params == 193


class TestPolicyNetworkScoring:
    """Tests for scoring methods."""

    def test_get_scores_numpy(self):
        """Test get_scores with numpy input."""
        model = CellNN(input_dim=10, hidden_dim=16)
        policy = PolicyNetwork(model)

        obs = np.random.randn(10, 100).astype(np.float32)
        scores = policy.get_scores(obs)

        assert isinstance(scores, torch.Tensor)
        assert scores.shape == (100,)

    def test_get_scores_torch(self):
        """Test get_scores with torch input."""
        model = CellNN(input_dim=10, hidden_dim=16)
        policy = PolicyNetwork(model)

        obs = torch.randn(10, 100)
        scores = policy.get_scores(obs)

        assert isinstance(scores, torch.Tensor)
        assert scores.shape == (100,)

    def test_select_k_cells(self):
        """Test select_k_cells returns correct count."""
        model = CellNN(input_dim=10, hidden_dim=16)
        policy = PolicyNetwork(model)

        scores = torch.randn(100)
        selected = policy.select_k_cells(scores, n_cells=10)

        assert len(selected) == 10
        assert torch.all(selected < 100)  # Valid indices

    def test_select_k_cells_unique(self):
        """Test select_k_cells returns unique indices."""
        model = CellNN(input_dim=10, hidden_dim=16)
        policy = PolicyNetwork(model)

        scores = torch.randn(100)
        selected = policy.select_k_cells(scores, n_cells=10)

        assert len(torch.unique(selected)) == 10


class TestPolicyNetworkActions:
    """Tests for get_actions method."""

    def test_get_actions_basic(self):
        """Test get_actions returns correct count."""
        model = CellNN(input_dim=10, hidden_dim=16)
        policy = PolicyNetwork(model)

        obs = torch.randn(10, 100)
        actions = policy.get_actions(obs, n_cells=5)

        assert len(actions) == 5

    def test_get_actions_with_constraint_mask(self):
        """Test get_actions respects constraint mask."""
        model = CellNN(input_dim=10, hidden_dim=16)
        policy = PolicyNetwork(model)

        obs = torch.randn(10, 100)
        # Mask first 50 cells
        mask = torch.zeros(100, dtype=torch.bool)
        mask[:50] = True

        actions = policy.get_actions(obs, n_cells=10, constraint_mask=mask)

        # All selected cells should be in unmasked region
        assert torch.all(actions >= 50)

    def test_get_actions_numpy_mask(self):
        """Test get_actions with numpy constraint mask."""
        model = CellNN(input_dim=10, hidden_dim=16)
        policy = PolicyNetwork(model)

        obs = torch.randn(10, 100)
        mask = np.zeros(100, dtype=bool)
        mask[:50] = True

        actions = policy.get_actions(obs, n_cells=10, constraint_mask=mask)

        assert torch.all(actions >= 50)


class TestPolicyNetworkDevice:
    """Tests for device handling."""

    def test_to_cpu(self):
        """Test moving to CPU."""
        model = CellNN(input_dim=10, hidden_dim=16)
        policy = PolicyNetwork(model)

        policy.to("cpu")

        assert policy.device == torch.device("cpu")

    def test_to_returns_self(self):
        """Test that to() returns self for chaining."""
        model = CellNN(input_dim=10, hidden_dim=16)
        policy = PolicyNetwork(model)

        result = policy.to("cpu")

        assert result is policy


class TestPolicyNetworkSaveLoad:
    """Tests for save/load functionality."""

    def test_save_load_roundtrip(self):
        """Test save and load preserves weights."""
        model = CellNN(input_dim=10, hidden_dim=16)
        policy = PolicyNetwork(model)

        original_weights = policy.get_flat_weights()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            policy.save(path)

            # Create new policy and load
            model2 = CellNN(input_dim=10, hidden_dim=16)
            policy2 = PolicyNetwork(model2)
            policy2.load(path)

            loaded_weights = policy2.get_flat_weights()

            assert np.allclose(original_weights, loaded_weights)
        finally:
            Path(path).unlink(missing_ok=True)
