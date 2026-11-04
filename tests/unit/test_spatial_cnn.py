"""Unit tests for SpatialCNN module."""

from __future__ import annotations

import pytest
import torch

from captain.agents.spatial_cnn import SpatialCNN


class TestSpatialCNNInit:
    """Tests for SpatialCNN initialization."""

    def test_init_defaults(self):
        """Test default initialization."""
        cnn = SpatialCNN()

        assert cnn.n_input_channels == 12
        assert cnn.n_branch_filters == 8
        assert cnn.n_output_features == 8

    def test_init_custom(self):
        """Test custom initialization."""
        cnn = SpatialCNN(n_input_channels=6, n_branch_filters=4, n_output_features=4)

        assert cnn.n_input_channels == 6
        assert cnn.n_branch_filters == 4
        assert cnn.n_output_features == 4

    def test_param_count_default(self):
        """Test parameter count matches plan (~1,880)."""
        cnn = SpatialCNN(n_input_channels=12, n_branch_filters=8, n_output_features=8)
        n_params = sum(p.numel() for p in cnn.parameters())

        # local:    12*8*3*3 + 8 = 872
        # regional: 12*8*3*3 + 8 = 872
        # mix:      16*8*1*1 + 8 = 136
        # total: 1880
        assert n_params == 1880


class TestSpatialCNNForward:
    """Tests for SpatialCNN forward pass."""

    def test_output_shape(self):
        """Test output shape matches input spatial dims."""
        cnn = SpatialCNN()
        x = torch.randn(1, 12, 50, 40)
        out = cnn(x)

        assert out.shape == (1, 8, 50, 40)

    def test_output_shape_batched(self):
        """Test output shape with batch dimension > 1."""
        cnn = SpatialCNN()
        x = torch.randn(4, 12, 30, 20)
        out = cnn(x)

        assert out.shape == (4, 8, 30, 20)

    def test_output_shape_custom(self):
        """Test output shape with custom filter counts."""
        cnn = SpatialCNN(n_input_channels=6, n_branch_filters=4, n_output_features=4)
        x = torch.randn(1, 6, 20, 15)
        out = cnn(x)

        assert out.shape == (1, 4, 20, 15)

    def test_deterministic(self):
        """Test forward pass is deterministic in eval mode."""
        cnn = SpatialCNN()
        cnn.eval()
        x = torch.randn(1, 12, 20, 15)

        out1 = cnn(x)
        out2 = cnn(x)

        assert torch.allclose(out1, out2)

    def test_output_finite(self):
        """Test output has no NaN or Inf."""
        cnn = SpatialCNN()
        x = torch.randn(1, 12, 30, 25)
        out = cnn(x)

        assert torch.isfinite(out).all()

    def test_small_grid(self):
        """Test with very small grids (edge case)."""
        cnn = SpatialCNN()
        x = torch.randn(1, 12, 3, 3)
        out = cnn(x)

        assert out.shape == (1, 8, 3, 3)
        assert torch.isfinite(out).all()

    def test_spatial_dims_preserved(self):
        """Test that H and W are preserved exactly."""
        cnn = SpatialCNN()
        for h, w in [(10, 10), (7, 13), (100, 80), (1, 1)]:
            x = torch.randn(1, 12, h, w)
            out = cnn(x)
            assert out.shape[2:] == (h, w)


class TestSpatialCNNDevice:
    """Tests for device handling."""

    def test_cpu(self):
        """Test on CPU."""
        cnn = SpatialCNN()
        x = torch.randn(1, 12, 10, 10)
        out = cnn(x)

        assert out.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self):
        """Test on CUDA."""
        cnn = SpatialCNN().cuda()
        x = torch.randn(1, 12, 10, 10, device="cuda")
        out = cnn(x)

        assert out.device.type == "cuda"
