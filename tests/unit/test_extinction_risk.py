"""Unit tests for ExtinctionRisk class."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from captain.data.extinction_risk import ExtinctionRisk, ExtinctionRiskStatic


class TestExtinctionRiskInit:
    """Tests for ExtinctionRisk initialization."""

    def test_init_with_numpy(self, rng):
        """Test initialization with numpy array."""
        init_status = rng.integers(0, 5, size=10).astype(np.int64)
        er = ExtinctionRisk(init_status=init_status)

        assert er.init_status.shape == (10,)
        assert isinstance(er.init_status, torch.Tensor)

    def test_init_with_torch(self, rng):
        """Test initialization with torch tensor."""
        init_status = torch.randint(0, 5, (10,))
        er = ExtinctionRisk(init_status=init_status)

        assert er.init_status.shape == (10,)

    def test_init_default_n_classes(self, rng):
        """Test default number of classes is 5."""
        init_status = rng.integers(0, 5, size=10).astype(np.int64)
        er = ExtinctionRisk(init_status=init_status)

        assert er._n_classes == 5

    def test_init_custom_n_classes(self, rng):
        """Test custom number of classes."""
        init_status = rng.integers(0, 3, size=10).astype(np.int64)
        er = ExtinctionRisk(init_status=init_status, n_classes=3)

        assert er._n_classes == 3

    def test_init_default_thresholds(self, rng):
        """Test default threshold generation."""
        init_status = rng.integers(0, 5, size=10).astype(np.int64)
        er = ExtinctionRisk(init_status=init_status, n_classes=5, alpha=1.0)

        # With alpha=1.0, thresholds should be evenly spaced
        expected = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        assert torch.allclose(er._loss_thresholds, expected)
        assert torch.allclose(er._protect_thresholds, expected)

    def test_init_custom_loss_thresholds(self, rng):
        """Test custom loss thresholds."""
        init_status = rng.integers(0, 5, size=10).astype(np.int64)
        custom_thresholds = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 1.0])

        er = ExtinctionRisk(
            init_status=init_status,
            loss_thresholds=custom_thresholds
        )

        assert len(er._loss_thresholds) == 6  # n_classes + 1

    def test_init_invalid_loss_thresholds(self, rng):
        """Test that invalid threshold count raises error."""
        init_status = rng.integers(0, 5, size=10).astype(np.int64)
        wrong_thresholds = np.array([0.0, 0.5, 1.0])  # Wrong count

        with pytest.raises(ValueError, match="Number of loss thresholds"):
            ExtinctionRisk(
                init_status=init_status,
                n_classes=5,
                loss_thresholds=wrong_thresholds
            )

    def test_init_with_device(self, rng):
        """Test initialization with device."""
        init_status = rng.integers(0, 5, size=10).astype(np.int64)
        er = ExtinctionRisk(init_status=init_status, device="cpu")

        assert er.device == torch.device("cpu")
        assert er.init_status.device.type == "cpu"


class TestExtinctionRiskClassify:
    """Tests for classify method."""

    @pytest.fixture
    def setup_ext_risk(self, rng):
        """Create ExtinctionRisk with init values set."""
        n_species = 10
        init_status = torch.zeros(n_species, dtype=torch.long)  # All LC
        er = ExtinctionRisk(init_status=init_status, n_classes=5)

        init_pop = torch.ones(n_species) * 1000
        init_protected_pop = torch.zeros(n_species)
        er.set_init_values(init_pop, init_protected_pop)

        return er

    def test_classify_no_change(self, setup_ext_risk):
        """Test classification with no population decline."""
        er = setup_ext_risk
        current_pop = torch.ones(10) * 1000  # Same as init
        protected_pop = torch.zeros(10)

        risk = er.classify(current_pop, protected_pop)

        # No change should keep species at lowest risk
        assert torch.all(risk == 0)

    def test_classify_population_decline_increases_risk(self, setup_ext_risk):
        """Test that population decline increases risk."""
        er = setup_ext_risk
        current_pop = torch.ones(10) * 100  # 90% decline
        protected_pop = torch.zeros(10)

        risk = er.classify(current_pop, protected_pop)

        # Severe decline should increase risk
        assert torch.all(risk > 0)

    def test_classify_protection_reduces_risk(self, rng):
        """Test that protection reduces risk."""
        n_species = 10
        init_status = torch.ones(n_species, dtype=torch.long) * 2  # Start at VU
        er = ExtinctionRisk(init_status=init_status, n_classes=5)

        init_pop = torch.ones(n_species) * 1000
        er.set_init_values(init_pop, torch.zeros(n_species))

        current_pop = torch.ones(10) * 1000
        protected_pop = torch.ones(10) * 900  # 90% protected

        risk = er.classify(current_pop, protected_pop)

        # High protection should reduce risk below initial
        assert torch.all(risk < 2)

    def test_classify_risk_clamped(self, setup_ext_risk):
        """Test that risk is clamped to valid range."""
        er = setup_ext_risk

        # Extreme decline
        current_pop = torch.ones(10) * 0.001
        protected_pop = torch.zeros(10)

        risk = er.classify(current_pop, protected_pop)

        # Risk should be clamped to max (n_classes - 1 = 4)
        assert torch.all(risk <= 4)
        assert torch.all(risk >= 0)

    def test_classify_extinction(self, setup_ext_risk):
        """Test classification at near-extinction."""
        er = setup_ext_risk
        current_pop = torch.zeros(10)  # Extinct
        protected_pop = torch.zeros(10)

        risk = er.classify(current_pop, protected_pop)

        # Should be at highest risk
        assert torch.all(risk == 4)


class TestExtinctionRiskSetInitValues:
    """Tests for set_init_values method."""

    def test_set_init_values_numpy(self, rng):
        """Test setting init values with numpy arrays."""
        init_status = rng.integers(0, 5, size=10).astype(np.int64)
        er = ExtinctionRisk(init_status=init_status)

        init_pop = rng.random(10).astype(np.float32) * 1000
        init_protected = rng.random(10).astype(np.float32) * 100

        er.set_init_values(init_pop, init_protected)

        assert er.init_pop is not None
        assert er.init_pop.shape == (10,)

    def test_set_init_values_torch(self, rng):
        """Test setting init values with torch tensors."""
        init_status = torch.randint(0, 5, (10,))
        er = ExtinctionRisk(init_status=init_status)

        init_pop = torch.rand(10) * 1000
        init_protected = torch.rand(10) * 100

        er.set_init_values(init_pop, init_protected)

        assert er.init_pop is not None


class TestExtinctionRiskSpeciesPerClass:
    """Tests for species_per_class method."""

    def test_species_per_class_counts(self, rng):
        """Test species count per class."""
        # 3 LC, 2 NT, 2 VU, 2 EN, 1 CR
        init_status = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3, 3, 4])
        er = ExtinctionRisk(init_status=init_status)

        counts = er.species_per_class()

        assert counts[0] == 3  # LC
        assert counts[1] == 2  # NT
        assert counts[4] == 1  # CR

    def test_species_per_class_normalized(self, rng):
        """Test normalized species per class."""
        init_status = torch.tensor([0, 0, 0, 0, 1])  # 4 LC, 1 NT
        er = ExtinctionRisk(init_status=init_status)

        proportions = er.species_per_class(normalize=True)

        assert abs(proportions[0].item() - 0.8) < 1e-5  # 80% LC
        assert abs(proportions[1].item() - 0.2) < 1e-5  # 20% NT
        assert abs(proportions.sum().item() - 1.0) < 1e-5

    def test_species_per_class_custom_labels(self, rng):
        """Test species per class with custom risk labels."""
        init_status = torch.zeros(10, dtype=torch.long)
        er = ExtinctionRisk(init_status=init_status)

        custom_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        counts = er.species_per_class(risk_labels=custom_labels)

        assert torch.all(counts == 2)  # 2 species per class


class TestExtinctionRiskDevice:
    """Tests for device handling."""

    def test_to_cpu(self, rng):
        """Test moving to CPU."""
        init_status = rng.integers(0, 5, size=10).astype(np.int64)
        er = ExtinctionRisk(init_status=init_status)

        er.to("cpu")

        assert er.device == torch.device("cpu")
        assert er.init_status.device.type == "cpu"

    def test_to_returns_self(self, rng):
        """Test that to() returns self for chaining."""
        init_status = rng.integers(0, 5, size=10).astype(np.int64)
        er = ExtinctionRisk(init_status=init_status)

        result = er.to("cpu")

        assert result is er


class TestExtinctionRiskStatic:
    """Tests for ExtinctionRiskStatic subclass."""

    def test_static_never_changes(self, rng):
        """Test that static risk never changes."""
        init_status = torch.tensor([0, 1, 2, 3, 4])
        er = ExtinctionRiskStatic(init_status=init_status)

        # Set init values
        er.set_init_values(
            torch.ones(5) * 1000,
            torch.zeros(5)
        )

        # Even with extinction, status should not change
        current_pop = torch.zeros(5)
        protected_pop = torch.zeros(5)

        risk = er.classify(current_pop, protected_pop)

        assert torch.all(risk == init_status)

    def test_static_ignores_protection(self, rng):
        """Test that static risk ignores protection."""
        init_status = torch.tensor([3, 3, 3])  # All EN
        er = ExtinctionRiskStatic(init_status=init_status)

        current_pop = torch.ones(3) * 1000
        protected_pop = torch.ones(3) * 1000  # 100% protected

        risk = er.classify(current_pop, protected_pop)

        # Still EN despite full protection
        assert torch.all(risk == 3)
