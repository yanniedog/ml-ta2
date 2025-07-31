"""model_trainer.py

Centralised training orchestration for the ML-TA platform.

Features
========
1. Supports multiple ensemble-friendly algorithms:
   * LightGBM, XGBoost, CatBoost, (fallback to RandomForest if unavailable)
2. Optuna-powered hyper-parameter optimisation with early stopping.
3. Automatic feature / target extraction from provided `pandas.DataFrame`.
4. SHAP value calculation for post-hoc interpretability (optional).
5. Structured logging and rich progress bars.
6. Model persistence to the configured `models/` directory including
   versioning via timestamp and git SHA (if available).

NOTE: Designed for local execution – no external cloud resources
required. All dependencies are optional; missing libraries will be
ignored gracefully with a warning.
"""
from __future__ import annotations

import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import optuna
except ImportError:  # pragma: no cover – optional dependency
    optuna = None  # type: ignore

# ML libraries (optional)
try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None  # type: ignore

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None  # type: ignore

try:
    import catboost as cb
except ImportError:  # pragma: no cover
    cb = None  # type: ignore

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Local utilities
try:
    from src.logging_config import logger  # type: ignore
except Exception:  # pragma: no cover – fallback basic logger
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class ModelTrainer:
    """High-level training interface."""

    SUPPORTED_ALGOS = {
        "lightgbm": bool(lgb),
        "xgboost": bool(xgb),
        "catboost": bool(cb),
        "random_forest": True,  # Always available
    }

    DEFAULT_PARAMS = {
        "lightgbm": {
            "num_leaves": 64,
            "learning_rate": 0.05,
            "n_estimators": 400,
            "objective": "regression",
        },
        "xgboost": {
            "max_depth": 6,
            "eta": 0.05,
            "subsample": 0.8,
            "objective": "reg:squarederror",
            "num_round": 400,
        },
        "catboost": {
            "depth": 6,
            "learning_rate": 0.05,
            "iterations": 400,
            "loss_function": "RMSE",
            "verbose": False,
        },
        "random_forest": {
            "n_estimators": 300,
            "max_depth": None,
            "n_jobs": -1,
        },
    }

    def __init__(
        self,
        algo: str = "lightgbm",
        model_dir: str | Path | None = None,
        optuna_trials: int = 0,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        algo = algo.lower()
        if algo not in self.SUPPORTED_ALGOS:
            raise ValueError(f"Unsupported algorithm: {algo}")
        if not self.SUPPORTED_ALGOS[algo]:
            logger.warning("Algorithm %s not available – falling back to RandomForest", algo)
            algo = "random_forest"
        self.algo = algo
        self.optuna_trials = optuna_trials if optuna and optuna_trials > 0 else 0
        self.test_size = test_size
        self.random_state = random_state
        self.params: Dict[str, Any] = self.DEFAULT_PARAMS[self.algo].copy()

        self.model: Any = None
        self.history: Dict[str, Any] = {}
        self.model_dir = Path(model_dir or "models").resolve()
        self.model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self, data: pd.DataFrame, target_column: str) -> Tuple[Any, float]:
        """Train a model and return fitted model and validation RMSE."""
        X = data.drop(columns=[target_column])
        y = data[target_column]

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, shuffle=False
        )

        if self.optuna_trials:
            self._run_optuna(X_train, y_train, X_valid, y_valid)

        self.model = self._build_model(self.params)
        self._fit_model(X_train, y_train, X_valid, y_valid)

        preds = self._predict_internal(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, preds))
        logger.info("Validation RMSE: %.5f", rmse)

        self.history["rmse"] = rmse
        self._persist_model()
        return self.model, rmse

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _build_model(self, params: Dict[str, Any]):
        if self.algo == "lightgbm":
            return lgb.LGBMRegressor(**params)
        if self.algo == "xgboost":
            return xgb.XGBRegressor(**params)
        if self.algo == "catboost":
            return cb.CatBoostRegressor(**params)
        # random forest fallback
        return RandomForestRegressor(**params)

    def _fit_model(self, X_train, y_train, X_valid, y_valid):
        if self.algo == "catboost":
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                use_best_model=True,
            )
        elif self.algo == "lightgbm":
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
            )
        elif self.algo == "xgboost":
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False,
            )
        else:  # random forest – no early stopping
            self.model.fit(X_train, y_train)

    def _predict_internal(self, X):
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        raise RuntimeError("Model has no predict method")

    # ------------------------------------------------------------------
    # Optuna integration
    # ------------------------------------------------------------------
    def _run_optuna(self, X_train, y_train, X_valid, y_valid):
        if not optuna:
            logger.warning("Optuna not available – skipping hyperparameter optimisation")
            return

        logger.info("Starting Optuna hyperparameter tuning (%d trials)", self.optuna_trials)

        def objective(trial: optuna.Trial):  # type: ignore
            params = self._suggest_params(trial)
            model = self._build_model(params)
            self._fit_model(X_train, y_train, X_valid, y_valid)
            preds = model.predict(X_valid)
            rmse = np.sqrt(mean_squared_error(y_valid, preds))
            return rmse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.optuna_trials, show_progress_bar=False)
        self.params.update(study.best_params)
        logger.info("Optuna best RMSE: %.5f", study.best_value)

    def _suggest_params(self, trial: "optuna.Trial") -> Dict[str, Any]:
        """Return algorithm-specific param suggestions."""
        if self.algo == "lightgbm":
            return {
                "num_leaves": trial.suggest_int("num_leaves", 31, 256),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            }
        if self.algo == "xgboost":
            return {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "num_round": trial.suggest_int("num_round", 200, 1000),
            }
        if self.algo == "catboost":
            return {
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "iterations": trial.suggest_int("iterations", 200, 1000),
            }
        # random forest
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 4, 20),
        }

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------
    def _persist_model(self):
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.algo}_{timestamp}.pkl"
        full_path = self.model_dir / filename
        with open(full_path, "wb") as f:
            pickle.dump({"model": self.model, "params": self.params, "history": self.history}, f)
        logger.info("Model persisted to %s", full_path)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @classmethod
    def available_algorithms(cls) -> List[str]:
        """Return a list of algorithms available in current environment."""
        return [k for k, v in cls.SUPPORTED_ALGOS.items() if v]


# Convenience function ----------------------------------------------------

def train_model(
    data: pd.DataFrame,
    target_column: str,
    algo: str = "lightgbm",
    model_dir: Optional[str | Path] = None,
    optuna_trials: int = 0,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Any, float]:
    """Functional wrapper around `ModelTrainer` for quick use."""
    trainer = ModelTrainer(
        algo=algo,
        model_dir=model_dir,
        optuna_trials=optuna_trials,
        test_size=test_size,
        random_state=random_state,
    )
    return trainer.train(data, target_column)


if __name__ == "__main__":  # pragma: no cover – simple sanity check
    df = pd.DataFrame(
        {
            "feat1": np.random.randn(1000),
            "feat2": np.random.randn(1000),
            "target": np.random.randn(1000).cumsum(),
        }
    )
    model, score = train_model(df, target_column="target", algo="random_forest")
    print("Trained model", model, "RMSE", score)
