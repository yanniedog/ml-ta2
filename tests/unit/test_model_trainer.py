"""Unit tests for model_trainer.py."""

import os
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.model_trainer import ModelTrainer, train_model


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Return a sample DataFrame for training."""
    return pd.DataFrame(
        {
            "feat1": np.random.randn(100),
            "feat2": np.random.randn(100),
            "target": np.random.randn(100).cumsum(),
        }
    )


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for models."""
    return tmp_path / "models"


# --------------------------------------------------------------------------
# Core functionality tests
# --------------------------------------------------------------------------


def test_trainer_init(model_dir: Path):
    """Test ModelTrainer initialisation."""
    trainer = ModelTrainer(algo="random_forest", model_dir=model_dir)
    assert trainer.algo == "random_forest"
    assert trainer.model_dir == model_dir
    assert trainer.optuna_trials == 0
    assert trainer.model is None


def test_unsupported_algorithm_raises_error():
    """Test that an unsupported algorithm raises a ValueError."""
    with pytest.raises(ValueError, match="Unsupported algorithm: not_an_algo"):
        ModelTrainer(algo="not_an_algo")


@patch("src.model_trainer.lgb", None)
def test_missing_library_falls_back_to_random_forest(model_dir: Path):
    """Test that a missing optional library falls back to RandomForest."""
    trainer = ModelTrainer(algo="lightgbm", model_dir=model_dir)
    assert trainer.algo == "random_forest"


@pytest.mark.parametrize("algo", ModelTrainer.available_algorithms())
def test_train_function_runs_for_all_available_algos(
    algo: str, sample_data: pd.DataFrame, model_dir: Path
):
    """Test that the main `train` function executes for all available algorithms."""
    trainer = ModelTrainer(algo=algo, model_dir=model_dir, optuna_trials=0)
    model, rmse = trainer.train(sample_data, target_column="target")

    assert model is not None
    assert isinstance(rmse, float)
    assert rmse > 0

    # Check model persistence
    saved_models = list(model_dir.glob(f"{algo}_*.pkl"))
    assert len(saved_models) == 1
    with open(saved_models[0], "rb") as f:
        artefact = pickle.load(f)
    assert "model" in artefact
    assert "params" in artefact


# --------------------------------------------------------------------------
# Optuna integration tests
# --------------------------------------------------------------------------


@patch("src.model_trainer.optuna")
def test_optuna_is_called_when_trials_are_set(optuna_mock: MagicMock, sample_data, model_dir):
    """Test that Optuna is invoked when `optuna_trials` > 0."""
    optuna_mock.create_study.return_value.best_params = {"n_estimators": 150}
    trainer = ModelTrainer(
        algo="random_forest", model_dir=model_dir, optuna_trials=5
    )
    trainer.train(sample_data, target_column="target")

    optuna_mock.create_study.assert_called_once_with(direction="minimize")
    study = optuna_mock.create_study.return_value
    study.optimize.assert_called_once()
    assert trainer.params["n_estimators"] == 150


@patch("src.model_trainer.optuna", None)
def test_optuna_is_skipped_if_not_available(sample_data, model_dir):
    """Test that Optuna is skipped if the library is not installed."""
    # No mock for optuna, so it will be None
    trainer = ModelTrainer(
        algo="random_forest", model_dir=model_dir, optuna_trials=5
    )
    # should run without error
    trainer.train(sample_data, target_column="target")
    # Check that default params are used
    assert trainer.params["n_estimators"] == 300


# --------------------------------------------------------------------------
# Convenience wrapper tests
# --------------------------------------------------------------------------

def test_train_model_wrapper(sample_data, model_dir):
    """Test the functional wrapper `train_model`."""
    model, rmse = train_model(
        sample_data,
        target_column="target",
        algo="random_forest",
        model_dir=model_dir,
    )
    assert model is not None
    assert isinstance(rmse, float)


# --------------------------------------------------------------------------
# Utility tests
# --------------------------------------------------------------------------

def test_available_algorithms_utility():
    """Test the `available_algorithms` classmethod."""
    available = ModelTrainer.available_algorithms()
    assert isinstance(available, list)
    assert "random_forest" in available
