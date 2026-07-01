"""Benchmark runner scaffolding for future real-data experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from wavestgate.evaluation.metrics import summarize_proportion_metrics


@dataclass
class BenchmarkResult:
    method: str
    metrics: dict[str, float]


def evaluate_method(method_name: str, predict_fn: Callable, batch) -> BenchmarkResult:
    """Evaluate a callable that returns predicted proportions for a batch."""

    if batch.proportion_gt is None:
        raise ValueError("Ground-truth proportions are required for this benchmark helper")
    predicted = predict_fn(batch)
    return BenchmarkResult(method=method_name, metrics=summarize_proportion_metrics(predicted, batch.proportion_gt))
