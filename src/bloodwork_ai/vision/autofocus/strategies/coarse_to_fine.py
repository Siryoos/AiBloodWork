from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Tuple

import numpy as np

from ..camera_interface import CameraInterface
from ..metrics import tenengrad
from ..diagnostics import DiagnosticsLogger


MetricFn = Callable[[np.ndarray], float]


def _crop_rois(img: np.ndarray, roi_fracs: Sequence[Tuple[float, float, float, float]]):
    h, w = img.shape[:2]
    rois = []
    for fx, fy, fw, fh in roi_fracs:
        x = int(round(fx * w))
        y = int(round(fy * h))
        ww = int(round(fw * w))
        hh = int(round(fh * h))
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        x2 = max(x + 1, min(w, x + ww))
        y2 = max(y + 1, min(h, y + hh))
        rois.append(img[y:y2, x:x2])
    return rois


def _median_metric(img: np.ndarray, metric: MetricFn, roi_fracs: Sequence[Tuple[float, float, float, float]]):
    vals = [float(metric(roi)) for roi in _crop_rois(img, roi_fracs)]
    return float(np.median(vals))


def _polyfit_peak(zs: Sequence[float], vs: Sequence[float]) -> float:
    zs = np.asarray(zs, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.float64)
    if len(zs) < 3 or len(vs) < 3:
        return float(zs[np.argmax(vs)])
    # Normalize for numerical stability
    z0 = zs.mean()
    zn = zs - z0
    try:
        coeffs = np.polyfit(zn, vs, 2)
    except Exception:
        return float(zs[np.argmax(vs)])
    a, b, c = coeffs
    if abs(a) < 1e-12:
        return float(zs[np.argmax(vs)])
    z_peak = -b / (2.0 * a)
    return float(z_peak + z0)


@dataclass
class CoarseToFineStrategy:
    """Coarse sweep -> quadratic refine -> micro hill-climb.

    Implements the blood-smear-friendly routine described in the user brief.
    """

    camera: CameraInterface
    bracket: float = 8.0
    coarse: float = 2.0
    fine: float = 0.3
    metric: MetricFn = tenengrad
    settle_time_s: float = 0.02
    roi_fracs: Tuple[Tuple[float, float, float, float], ...] = (
        (0.35, 0.35, 0.30, 0.30),  # center
        (0.10, 0.10, 0.25, 0.25),  # top-left
        (0.65, 0.10, 0.25, 0.25),  # top-right
        (0.10, 0.65, 0.25, 0.25),  # bottom-left
    )
    max_micro_iters: int = 20
    last_max_coarse: float | None = None
    last_best_value: float | None = None

    diag: DiagnosticsLogger | None = None

    def _sharpness(self, z: float, phase: str) -> float:
        self.camera.set_focus(z)
        time.sleep(self.settle_time_s)
        frame = self.camera.get_frame()
        val = _median_metric(frame, self.metric, self.roi_fracs)
        if self.diag is not None:
            self.diag.log_measurement(phase=phase, z=z, value=val)
        return val

    # for diagnostics export
    last_coarse_zs: list[float] | None = None
    last_coarse_vals: list[float] | None = None

    def find_best_focus(self, *, timeout_s: float | None = None) -> float:
        start = time.perf_counter()
        fmin, fmax = self.camera.get_focus_range()
        z_guess = float(np.clip(self.camera.get_focus(), fmin, fmax))

        # 1) coarse sweep
        zs = np.arange(z_guess - self.bracket, z_guess + self.bracket + 1e-9, self.coarse)
        zs = np.clip(zs, fmin, fmax)
        vals: list[float] = []
        best_idx = 0
        best_val = -np.inf
        for i, z in enumerate(zs):
            if timeout_s is not None and (time.perf_counter() - start) > timeout_s:
                break
            v = self._sharpness(float(z), phase="coarse")
            vals.append(v)
            if v > best_val:
                best_val = v
                best_idx = i

        if not vals:
            # fallback: just return current focus
            return z_guess
        self.last_max_coarse = float(np.max(vals))
        self.last_coarse_zs = [float(z) for z in zs]
        self.last_coarse_vals = [float(v) for v in vals]

        # 2) quadratic refine around best 3–5 points
        lo = max(0, best_idx - 2)
        hi = min(len(zs), best_idx + 3)
        z_fit = _polyfit_peak(zs[lo:hi], vals[lo:hi])
        z_fit = float(np.clip(z_fit, fmin, fmax))

        # 3) micro hill-climb
        best_z = z_fit
        best_v = self._sharpness(best_z, phase="refine")
        for _ in range(self.max_micro_iters):
            if timeout_s is not None and (time.perf_counter() - start) > timeout_s:
                break
            improved = False
            for step in (self.fine, -self.fine):
                z_try = float(np.clip(best_z + step, fmin, fmax))
                v = self._sharpness(z_try, phase="micro")
                if v > best_v:
                    best_v = v
                    best_z = z_try
                    improved = True
            if not improved:
                break

        # Finish at best_z
        self.camera.set_focus(best_z)
        time.sleep(self.settle_time_s)
        self.last_best_value = float(best_v)
        return float(best_z)
