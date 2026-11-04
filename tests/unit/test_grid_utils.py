"""Unit tests for grid_utils module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from captain.utils.grid_utils import (
    calculate_delta,
    compute_convolution_matrix,
    dispersal_distances_threshold_coords,
    flatten_grid,
    load_dispersal_distances,
    reconstruct_grid,
    save_dispersal_distances,
)


class TestDispersalDistances:
    """Tests for dispersal distance computation."""

    def test_dispersal_returns_csr_matrix(self):
        """Test dispersal_distances_threshold_coords returns sparse matrix."""
        coords = (np.array([0, 1, 2]), np.array([0, 1, 2]))
        result = dispersal_distances_threshold_coords(lambda_0=1.0, coords=coords)

        assert isinstance(result, csr_matrix)

    def test_dispersal_shape(self):
        """Test output shape matches number of cells."""
        n_cells = 5
        coords = (np.arange(n_cells), np.arange(n_cells))
        result = dispersal_distances_threshold_coords(lambda_0=1.0, coords=coords)

        assert result.shape == (n_cells, n_cells)

    def test_dispersal_diagonal_is_one(self):
        """Test that dispersal from cell to itself is 1."""
        coords = (np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3, 4]))
        result = dispersal_distances_threshold_coords(lambda_0=1.0, coords=coords)

        # Diagonal should be 1 (distance 0 -> exp(0) = 1)
        diagonal = result.diagonal()
        assert np.allclose(diagonal, 1.0)

    def test_dispersal_decays_with_distance(self):
        """Test that dispersal probability decays with distance."""
        # Create a simple linear arrangement
        coords = (np.array([0, 0, 0]), np.array([0, 1, 2]))
        result = dispersal_distances_threshold_coords(lambda_0=1.0, coords=coords)
        dense = result.toarray()

        # Dispersal from cell 0 to cell 1 should be > dispersal from cell 0 to cell 2
        assert dense[0, 1] > dense[0, 2]

    def test_dispersal_threshold_excludes_distant_cells(self):
        """Test that threshold excludes cells beyond range."""
        # Two cells far apart
        coords = (np.array([0, 10]), np.array([0, 10]))
        result = dispersal_distances_threshold_coords(
            lambda_0=1.0, coords=coords, threshold=3
        )
        dense = result.toarray()

        # Off-diagonal should be 0 (beyond threshold)
        assert dense[0, 1] == 0.0
        assert dense[1, 0] == 0.0

    def test_dispersal_symmetric(self):
        """Test dispersal matrix is symmetric."""
        coords = (np.array([0, 1, 2, 0, 1]), np.array([0, 0, 0, 1, 1]))
        result = dispersal_distances_threshold_coords(lambda_0=1.0, coords=coords)

        # Should be symmetric
        assert np.allclose(result.toarray(), result.T.toarray())


class TestSaveLoadDispersalDistances:
    """Tests for saving and loading dispersal distances."""

    def test_save_returns_matrix_when_no_filename(self):
        """Test save_dispersal_distances returns matrix if no filename."""
        coords = (np.array([0, 1, 2]), np.array([0, 1, 2]))
        result = save_dispersal_distances(lambda_0=1.0, coords=coords)

        assert isinstance(result, csr_matrix)

    def test_save_and_load_roundtrip(self):
        """Test save and load preserves matrix."""
        coords = (np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]))
        original = dispersal_distances_threshold_coords(lambda_0=2.0, coords=coords)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "dispersal.npz"
            save_dispersal_distances(
                lambda_0=2.0, coords=coords, filename=str(filepath)
            )

            loaded = load_dispersal_distances(str(filepath))

        assert np.allclose(original.toarray(), loaded.toarray())


class TestFlattenGrid:
    """Tests for flatten_grid function."""

    def test_flatten_basic(self):
        """Test basic grid flattening."""
        # 2 channels, 3x3 grid
        grid = np.ones((2, 3, 3))
        data_2d, coords, shape = flatten_grid(grid)

        assert data_2d.shape == (2, 9)
        assert len(coords[0]) == 9
        assert shape == (3, 3)

    def test_flatten_with_nan(self):
        """Test flattening with NaN values (masked)."""
        grid = np.ones((2, 3, 3))
        grid[:, 0, 0] = np.nan  # Mask one cell

        data_2d, coords, shape = flatten_grid(grid)

        assert data_2d.shape == (2, 8)  # One less cell
        assert len(coords[0]) == 8

    def test_flatten_with_explicit_mask(self):
        """Test flattening with explicit mask."""
        grid = np.ones((2, 3, 3))
        mask = np.ones((3, 3))
        mask[0, 0] = np.nan
        mask[1, 1] = np.nan

        data_2d, coords, shape = flatten_grid(grid, mask=mask)

        assert data_2d.shape == (2, 7)  # Two cells masked

    def test_flatten_preserves_values(self):
        """Test that flattening preserves values."""
        grid = np.arange(18).reshape(2, 3, 3).astype(float)

        data_2d, coords, shape = flatten_grid(grid)

        # Should preserve all values
        assert data_2d.sum() == grid.sum()

    def test_flatten_coords_correct(self):
        """Test that coordinates are correct."""
        grid = np.ones((1, 2, 3))
        grid[:, 1, 2] = np.nan  # Mask one cell

        data_2d, coords, shape = flatten_grid(grid)

        # Coordinates should be all valid cells
        assert len(coords[0]) == 5


class TestReconstructGrid:
    """Tests for reconstruct_grid function."""

    def test_reconstruct_basic(self):
        """Test basic reconstruction."""
        data_2d = np.array([[1, 2, 3], [4, 5, 6]])  # 2 channels, 3 cells
        coords = (np.array([0, 0, 1]), np.array([0, 1, 0]))  # Cell positions
        shape = (2, 2)

        reconstructed = reconstruct_grid(data_2d, coords, shape)

        assert reconstructed.shape == (2, 2, 2)
        assert reconstructed[0, 0, 0] == 1
        assert reconstructed[0, 0, 1] == 2
        assert reconstructed[0, 1, 0] == 3

    def test_reconstruct_fills_nan(self):
        """Test reconstruction fills missing cells with NaN."""
        data_2d = np.array([[1, 2]])  # 1 channel, 2 cells
        coords = (np.array([0, 1]), np.array([0, 0]))
        shape = (2, 2)

        reconstructed = reconstruct_grid(data_2d, coords, shape)

        # Cell (0, 1) and (1, 1) should be NaN
        assert np.isnan(reconstructed[0, 0, 1])
        assert np.isnan(reconstructed[0, 1, 1])

    def test_flatten_reconstruct_roundtrip(self):
        """Test that flatten and reconstruct are inverses."""
        original = np.random.randn(3, 5, 4)

        data_2d, coords, shape = flatten_grid(original)
        reconstructed = reconstruct_grid(data_2d, coords, shape)

        assert np.allclose(original, reconstructed)

    def test_flatten_reconstruct_with_mask(self):
        """Test roundtrip with masked data."""
        original = np.random.randn(2, 4, 4)
        original[:, 0, 0] = np.nan
        original[:, 2, 3] = np.nan

        data_2d, coords, shape = flatten_grid(original)
        reconstructed = reconstruct_grid(data_2d, coords, shape)

        # Valid cells should match
        valid_mask = ~np.isnan(original[0])
        assert np.allclose(
            original[:, valid_mask], reconstructed[:, valid_mask]
        )


class TestComputeConvolutionMatrix:
    """Tests for compute_convolution_matrix function."""

    def test_convolution_matrix_shape(self):
        """Test convolution matrix has correct shape."""
        n_cells = 10
        coords = (np.arange(n_cells), np.arange(n_cells))

        conv = compute_convolution_matrix(coords, radius=2)

        assert conv.shape == (n_cells, n_cells)

    def test_convolution_matrix_sparse(self):
        """Test convolution matrix is sparse."""
        coords = (np.arange(20), np.arange(20))

        conv = compute_convolution_matrix(coords)

        assert hasattr(conv, "toarray")  # Sparse matrix check

    def test_convolution_matrix_row_normalized(self):
        """Test rows sum to 1 (before transpose)."""
        # Create a 5x5 grid
        x, y = np.meshgrid(np.arange(5), np.arange(5))
        coords = (x.flatten(), y.flatten())

        conv = compute_convolution_matrix(coords, radius=1)

        # Since it's transposed for right multiplication, columns should sum to 1
        col_sums = np.array(conv.sum(axis=0)).flatten()
        assert np.allclose(col_sums, 1.0)

    def test_convolution_matrix_includes_neighbors(self):
        """Test convolution includes nearby cells."""
        # 3x3 grid
        x, y = np.meshgrid(np.arange(3), np.arange(3))
        coords = (x.flatten(), y.flatten())

        conv = compute_convolution_matrix(coords, radius=1)
        dense = conv.toarray()

        # Center cell (index 4) should connect to all 9 cells
        # After transpose, column 4 should have 9 non-zero entries
        assert np.count_nonzero(dense[:, 4]) == 9

    def test_convolution_radius_affects_neighbors(self):
        """Test that larger radius includes more neighbors."""
        x, y = np.meshgrid(np.arange(5), np.arange(5))
        coords = (x.flatten(), y.flatten())

        conv1 = compute_convolution_matrix(coords, radius=1)
        conv2 = compute_convolution_matrix(coords, radius=2)

        # Larger radius should have more non-zeros
        assert conv2.nnz > conv1.nnz

    def test_convolution_decay(self):
        """With decay, nearer neighbours get larger weights; rows still sum to 1."""
        x, y = np.meshgrid(np.arange(5), np.arange(5))
        coords = (x.flatten(), y.flatten())

        dense = compute_convolution_matrix(
            coords, radius=2, include_self=False, decay=1.0
        ).toarray()
        center = dense[:, 12]  # column = averaging weights of cell (2, 2)

        assert np.allclose(dense.sum(axis=0), 1.0)
        assert center[12] == 0.0
        assert center[13] > center[14] > 0  # distance 1 > distance 2
        assert center[13] / center[14] == pytest.approx(np.exp(1.0))

    def test_convolution_exclude_self(self):
        """Test include_self=False drops the diagonal and keeps rows normalized."""
        x, y = np.meshgrid(np.arange(3), np.arange(3))
        coords = (x.flatten(), y.flatten())

        dense = compute_convolution_matrix(coords, radius=1, include_self=False).toarray()

        assert np.allclose(np.diag(dense), 0.0)
        assert np.allclose(dense.sum(axis=0), 1.0)
        # Center cell (index 4) averages over its 8 neighbours
        assert np.count_nonzero(dense[:, 4]) == 8


class TestCalculateDelta:
    """Tests for calculate_delta function."""

    def test_delta_basic(self):
        """Test basic delta calculation."""
        present = np.array([0, 10, 20])
        future = np.array([10, 20, 30])
        n_steps = 10

        delta = calculate_delta(present, future, n_steps)

        expected = np.array([1, 1, 1])
        assert np.allclose(delta, expected)

    def test_delta_negative(self):
        """Test delta can be negative."""
        present = np.array([10, 20, 30])
        future = np.array([0, 10, 20])
        n_steps = 10

        delta = calculate_delta(present, future, n_steps)

        expected = np.array([-1, -1, -1])
        assert np.allclose(delta, expected)

    def test_delta_with_float_steps(self):
        """Test delta with float n_steps."""
        present = np.array([0.0])
        future = np.array([1.0])
        n_steps = 0.5

        delta = calculate_delta(present, future, n_steps)

        assert np.allclose(delta, [2.0])

    def test_delta_preserves_shape(self):
        """Test delta preserves input shape."""
        present = np.zeros((3, 4, 5))
        future = np.ones((3, 4, 5))
        n_steps = 10

        delta = calculate_delta(present, future, n_steps)

        assert delta.shape == present.shape

    def test_delta_2d_array(self):
        """Test delta with 2D array."""
        present = np.zeros((10, 20))
        future = np.ones((10, 20))
        n_steps = 5

        delta = calculate_delta(present, future, n_steps)

        assert np.allclose(delta, 0.2)
