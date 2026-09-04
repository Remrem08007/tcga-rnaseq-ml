from __future__ import annotations

from contextlib import contextmanager
import sys
import threading
import time
from typing import Callable, Iterable, Mapping, TextIO

import joblib.parallel
import numpy as np
from sklearn.model_selection import cross_validate


class ProgressReporter:
    """Report task/unit progress in terminals and line-oriented scheduler logs."""

    def __init__(
        self,
        *,
        prefix: str,
        task_name: str,
        unit_name: str,
        total_tasks: int,
        total_units: int,
        stream: TextIO | None = None,
        heartbeat_seconds: float = 60.0,
    ) -> None:
        if total_tasks < 1:
            raise ValueError("total_tasks must be >= 1")
        if total_units < 1:
            raise ValueError("total_units must be >= 1")
        self.prefix = prefix
        self.task_name = task_name
        self.unit_name = unit_name
        self.total_tasks = total_tasks
        self.total_units = total_units
        self.stream = stream if stream is not None else sys.stderr
        self.heartbeat_seconds = heartbeat_seconds
        self._is_terminal = bool(getattr(self.stream, "isatty", lambda: False)())
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._heartbeat: threading.Thread | None = None
        self._task_index = 0
        self._task_label = ""
        self._completed_units = 0
        self._task_started = 0.0

    @staticmethod
    def _format_duration(seconds: float) -> str:
        rounded = max(0, int(seconds))
        hours, remainder = divmod(rounded, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _line(self, *, state: str) -> str:
        fraction = self._completed_units / self.total_units
        filled = min(24, int(24 * fraction))
        bar = "#" * filled + "-" * (24 - filled)
        elapsed = self._format_duration(time.perf_counter() - self._task_started)
        return (
            f"[{self.prefix}] {self.task_name} {self._task_index}/{self.total_tasks} "
            f"({self._task_label}) [{bar}] "
            f"{self.unit_name} {self._completed_units}/{self.total_units} "
            f"elapsed {elapsed} {state}"
        )

    def _render(self, *, state: str, final: bool = False) -> None:
        with self._lock:
            prefix = "\r" if self._is_terminal else ""
            suffix = "\n" if final or not self._is_terminal else ""
            print(
                prefix + self._line(state=state),
                file=self.stream,
                end=suffix,
                flush=True,
            )

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            self._render(state="running")

    def start_task(
        self,
        *,
        task_index: int,
        task_label: str,
        state: str,
    ) -> None:
        self._task_index = task_index
        self._task_label = task_label
        self._completed_units = 0
        self._task_started = time.perf_counter()
        self._stop.clear()
        self._render(state=state)
        if self.heartbeat_seconds > 0:
            self._heartbeat = threading.Thread(
                target=self._heartbeat_loop,
                name=f"{self.prefix}-progress",
                daemon=True,
            )
            self._heartbeat.start()

    def units_completed(self, count: int) -> None:
        self._completed_units = min(
            self.total_units,
            self._completed_units + count,
        )
        self._render(state="running")

    def _stop_heartbeat(self) -> None:
        self._stop.set()
        if self._heartbeat is not None:
            self._heartbeat.join(timeout=max(1.0, self.heartbeat_seconds + 1.0))
            self._heartbeat = None

    def finish_task(self) -> None:
        self._stop_heartbeat()
        self._completed_units = self.total_units
        self._render(state="complete", final=True)

    def fail_task(self) -> None:
        self._stop_heartbeat()
        self._render(state="failed", final=True)


@contextmanager
def joblib_progress(callback: Callable[[int], None] | None):
    """Forward completed Joblib batches to a parent-process callback."""

    if callback is None:
        yield
        return

    original_callback = joblib.parallel.BatchCompletionCallBack

    class ProgressBatchCompletionCallback(original_callback):
        def __call__(self, *args, **kwargs):
            result = super().__call__(*args, **kwargs)
            callback(int(self.batch_size))
            return result

    joblib.parallel.BatchCompletionCallBack = ProgressBatchCompletionCallback
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = original_callback


def cross_validate_with_progress(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    *,
    scoring: Mapping[str, object],
    cv_splits: Iterable[tuple[np.ndarray, np.ndarray]],
    n_jobs: int,
    return_estimator: bool,
    error_score: str | float,
    progress_callback: Callable[[int], None] | None = None,
) -> dict[str, np.ndarray]:
    """Run cross-validation while reporting every completed fold.

    Joblib completion callbacks cover parallel execution.  Its sequential
    backend does not expose those callbacks, so one-fold calls are combined
    when progress is requested with ``n_jobs=1``.
    """

    splits = list(cv_splits)
    if progress_callback is not None and n_jobs == 1:
        chunks: list[dict[str, np.ndarray]] = []
        for split in splits:
            chunk = cross_validate(
                estimator,
                X,
                y,
                scoring=scoring,
                cv=[split],
                n_jobs=1,
                return_estimator=return_estimator,
                error_score=error_score,
            )
            chunks.append(chunk)
            progress_callback(1)
        combined: dict[str, np.ndarray] = {}
        for key in chunks[0]:
            if key == "estimator":
                estimators = [item for chunk in chunks for item in chunk[key]]
                values = np.empty(len(estimators), dtype=object)
                values[:] = estimators
                combined[key] = values
            else:
                combined[key] = np.concatenate(
                    [np.atleast_1d(chunk[key]) for chunk in chunks]
                )
        return combined

    with joblib_progress(progress_callback):
        return cross_validate(
            estimator,
            X,
            y,
            scoring=scoring,
            cv=splits,
            n_jobs=n_jobs,
            return_estimator=return_estimator,
            error_score=error_score,
        )
