from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..camera_interface import CameraInterface
from ..metrics import variance_of_laplacian
from ..diagnostics import DiagnosticsLogger


MetricFn = Callable[[np.ndarray], float]


@dataclass
class ContrastMaximizationStrategy:
    """Simple hill-climbing strategy to maximize a focus-sharpness metric.

    This approach samples the image sharpness around the current focus and
    moves focus in the direction that increases the metric. Step size can be
    reduced adaptively if no improvement is found.
    """

    camera: CameraInterface
    step: float = 5.0
    metric: MetricFn = variance_of_laplacian
    settle_time_s: float = 0.02
    min_step: float = 0.5
    max_iters: int = 50

    diag: DiagnosticsLogger | None = None

    def _score(self, phase: str) -> float:
        frame = self.camera.get_frame()
        val = float(self.metric(frame))
        if self.diag is not None:
            self.diag.log_measurement(phase=phase, z=self.camera.get_focus(), value=val)
        return val

    def find_best_focus(self, *, timeout_s: float | None = None) -> float:
        start_time = time.perf_counter()
        fmin, fmax = self.camera.get_focus_range()
        focus = float(self.camera.get_focus())
        focus = float(np.clip(focus, fmin, fmax))
        step = float(self.step)

        # Initialize score at current focus
        self.camera.set_focus(focus)
        time.sleep(self.settle_time_s)
        best_score = self._score(phase="init")
        best_focus = focus

        direction = 1.0  # try increasing focus first

        for _ in range(self.max_iters):
            if timeout_s is not None and (time.perf_counter() - start_time) > timeout_s:
                break

            # Probe next focus in current direction
            next_focus = float(np.clip(best_focus + direction * step, fmin, fmax))
            if abs(next_focus - best_focus) < 1e-6:
                # Reached boundary; flip direction and reduce step
                direction *= -1.0
                step *= 0.5
                if step < self.min_step:
                    break
                continue

            self.camera.set_focus(next_focus)
            time.sleep(self.settle_time_s)
            score = self._score(phase="walk")

            if score > best_score:
                best_score = score
                best_focus = next_focus
                # keep the same direction
            else:
                # reverse and reduce step
                direction *= -1.0
                step *= 0.5
                if step < self.min_step:
                    break

        # Ensure device ends at the chosen best focus
        self.camera.set_focus(best_focus)
        time.sleep(self.settle_time_s)
        return best_focus
