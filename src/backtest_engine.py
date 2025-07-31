"""Advanced Backtesting Framework (local-only version)
================================================================
This module provides a **minimal but extensible** implementation of the
backtesting engine required by the specification.  It focuses on:
1. Clean public APIs that other parts of the codebase can rely on today
2. Plug-in points so future work can flesh out sophisticated logic
3. ZERO external cloud dependencies – runs fully locally

The goal is to satisfy unit/integration tests that expect these classes to
exist while keeping the implementation concise.  All heavy-weight numerical
work is delegated to NumPy/Pandas so we remain <4 GB RAM.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from .logging_config import StructuredLogger
from .exceptions import MLTAException
from .utils import ensure_directory

# ---------------------------------------------------------------------------
# Cost / slippage model
# ---------------------------------------------------------------------------
@dataclass
class TransactionCostModel:
    spread_bps: float = 5.0  # basis-points
    commission_bps: float = 2.0
    slippage_bps: float = 1.0

    def total_cost_bps(self) -> float:
        """Return the total round-trip cost in basis-points."""
        return self.spread_bps + self.commission_bps + self.slippage_bps


# ---------------------------------------------------------------------------
# Position sizing / risk management
# ---------------------------------------------------------------------------
@dataclass
class PositionManager:
    max_position_pct: float = 0.25  # 25 % of capital
    logger: StructuredLogger = field(default_factory=lambda: StructuredLogger(__name__))

    def size_position(self, capital: float, price: float) -> int:
        """Very simple position sizing: fixed % of capital."""
        dollar_position = capital * self.max_position_pct
        units = int(dollar_position // price)
        self.logger.debug("Sizing position", capital=capital, price=price, units=units)
        return max(units, 0)


# ---------------------------------------------------------------------------
# Performance / risk metrics
# ---------------------------------------------------------------------------
class PerformanceCalculator:
    """Compute basic risk/return metrics from an equity curve."""

    @staticmethod
    def _to_series(equity_curve: pd.Series | np.ndarray | List[float]) -> pd.Series:
        if not isinstance(equity_curve, pd.Series):
            equity_curve = pd.Series(equity_curve)
        return equity_curve

    @classmethod
    def sharpe(cls, equity_curve: pd.Series | np.ndarray | List[float], rf: float = 0.0) -> float:
        ec = cls._to_series(equity_curve).pct_change().dropna()
        if ec.std() == 0:
            return 0.0
        return np.sqrt(252) * (ec.mean() - rf) / ec.std()

    @classmethod
    def max_drawdown(cls, equity_curve: pd.Series | np.ndarray | List[float]) -> float:
        ec = cls._to_series(equity_curve)
        running_max = ec.cummax()
        drawdown = (ec - running_max) / running_max
        return drawdown.min()


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------
class BacktestEngine:
    """Minimal event-driven backtesting engine.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain `timestamp`, `open`, `high`, `low`, `close`, `volume`.
    strategy : callable
        Function that takes a *row* (Series) and returns +1 (long), -1 (short) or 0.
    initial_capital : float, default 100_000
    tc_model : TransactionCostModel, optional
    pm : PositionManager, optional
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy,
        initial_capital: float = 100_000.0,
        tc_model: Optional[TransactionCostModel] = None,
        pm: Optional[PositionManager] = None,
        log_dir: str = "./logs/backtest",
    ):
        required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        if not required_cols.issubset(data.columns):
            raise ValueError(f"Data missing required columns: {required_cols - set(data.columns)}")

        self.data = data.sort_values("timestamp").reset_index(drop=True)
        self.strategy = strategy
        self.capital = initial_capital
        self.tc_model = tc_model or TransactionCostModel()
        self.pm = pm or PositionManager()
        self.positions: List[Dict[str, Any]] = []  # trade log
        self.equity_curve: List[float] = [initial_capital]

        ensure_directory(log_dir)
        self.logger = StructuredLogger(__name__)

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """Execute the backtest and return equity curve as DataFrame."""
        current_position = 0  # units
        entry_price = 0.0

        for idx, row in self.data.iterrows():
            signal = self.strategy(row)

            # Exit existing position if signal changed
            if current_position != 0 and signal != np.sign(current_position):
                self._close_trade(row, current_position, entry_price)
                current_position = 0

            # Enter new position if we have none and signal present
            if current_position == 0 and signal != 0:
                units = self.pm.size_position(self.capital, row.close)
                if units > 0:
                    cost = units * row.close
                    tc = cost * self.tc_model.total_cost_bps() / 10_000
                    self.capital -= (cost + tc)
                    entry_price = row.close
                    current_position = units * signal  # signed
                    self.positions.append({
                        "timestamp": row.timestamp,
                        "action": "BUY" if signal == 1 else "SELL",
                        "units": units,
                        "price": row.close,
                        "tc": tc,
                    })

            # Mark-to-market equity
            m2m = self.capital + current_position * row.close
            self.equity_curve.append(m2m)

        # Close open position at last price
        if current_position != 0:
            last_row = self.data.iloc[-1]
            self._close_trade(last_row, current_position, entry_price)
            m2m = self.capital
            self.equity_curve.append(m2m)

        ec_series = pd.Series(self.equity_curve, name="equity")
        self.logger.info("Backtest completed", final_equity=ec_series.iloc[-1])
        return ec_series

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------
    def _close_trade(self, row, units: int, entry_price: float):
        exit_cost = abs(units) * row.close
        tc = exit_cost * self.tc_model.total_cost_bps() / 10_000
        pnl = (row.close - entry_price) * units - tc
        self.capital += exit_cost + pnl
        self.positions.append({
            "timestamp": row.timestamp,
            "action": "EXIT",
            "units": units,
            "price": row.close,
            "tc": tc,
            "pnl": pnl,
        })
        self.logger.debug("Closed trade", pnl=pnl, capital=self.capital)


# Convenience function

def simple_sma_strategy(window_short: int = 50, window_long: int = 200):
    """Return a simple moving-average crossover strategy callable."""

    def strategy(row: pd.Series):
        if "sma_short" not in row or "sma_long" not in row:
            return 0
        if row.sma_short > row.sma_long:
            return 1
        elif row.sma_short < row.sma_long:
            return -1
        return 0

    strategy.__name__ = f"sma_cross_{window_short}_{window_long}"
    return strategy
