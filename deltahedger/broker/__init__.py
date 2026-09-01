"""Execution handlers: simulated for backtests, IBKR for live routing."""

from .base import ExecutionError, ExecutionHandler, Fill
from .paper import SimulatedExecution

__all__ = ["ExecutionError", "ExecutionHandler", "Fill", "SimulatedExecution"]
