#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple, Callable

import numpy as np
import cv2

from bloodwork_ai.vision.autofocus import (
    AutoFocusController,
    CameraInterface,
    CoarseToFineStrategy,
    ContrastMaximizationStrategy,
    variance_of_laplacian,
    tenengrad,
    brenner_gradient,
    high_frequency_energy,
    FocusSurfaceBuilder,
    FocusSurfaceModel,
    save_surface,
    load_surface,
    fit_rbf,
    save_surface_cache,
    load_surface_from_cache,
)
from bloodwork_ai.vision.autofocus.stage_interface import StageInterface
from bloodwork_ai.vision.autofocus.stages.gcode import SerialGCodeStage
from bloodwork_ai.vision.autofocus.diagnostics import DiagnosticsLogger


# ---------------------- Camera Drivers ----------------------


@dataclass
class OpenCVUvcCamera(CameraInterface):
    device_index: int = 0
    focus_min: float = 0.0
    focus_max: float = 255.0
    width: Optional[int] = None
    height: Optional[int] = None
    use_v4l2: bool = False
    v4l2_dev: str = "/dev/video0"
    lock_exposure: bool = False
    exposure_value: float | None = None  # backend-specific units

    def __post_init__(self) -> None:
        self.cap = cv2.VideoCapture(self.device_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera index {self.device_index}")
        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Disable auto-focus if supported
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self._focus = float(self.cap.get(cv2.CAP_PROP_FOCUS) or 0.0)
        if self.lock_exposure:
            # Best effort: try to lock exposure across backends
            # Note: OpenCV exposure controls vary by OS/driver
            try:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual on some backends
            except Exception:
                pass
            if self.exposure_value is not None:
                try:
                    self.cap.set(cv2.CAP_PROP_EXPOSURE, float(self.exposure_value))
                except Exception:
                    pass

    def get_frame(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame")
        return frame

    def _v4l2_set_focus(self, value: float) -> None:
        val = int(round(value))
        try:
            subprocess.run([
                "v4l2-ctl", "-d", self.v4l2_dev,
                "-c", "focus_auto=0",
                "-c", f"focus_absolute={val}"
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            raise RuntimeError(f"v4l2-ctl failed: {e}")

    def _v4l2_set_exposure(self) -> None:
        if not self.lock_exposure:
            return
        try:
            args = ["v4l2-ctl", "-d", self.v4l2_dev, "-c", "exposure_auto=1"]
            if self.exposure_value is not None:
                args += ["-c", f"exposure_absolute={int(round(self.exposure_value))}"]
            subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            raise RuntimeError(f"v4l2-ctl exposure failed: {e}")

    def _v4l2_get_focus(self) -> float:
        try:
            p = subprocess.run([
                "v4l2-ctl", "-d", self.v4l2_dev, "-C", "focus_absolute"
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            out = p.stdout.strip()
            # Expect lines like: 'focus_absolute: 123'
            parts = out.split(":")
            if len(parts) == 2:
                return float(parts[1].strip())
        except Exception as e:
            raise RuntimeError(f"v4l2-ctl failed: {e}")
        return self._focus

    def set_focus(self, value: float) -> None:
        v = max(self.focus_min, min(self.focus_max, value))
        if self.use_v4l2:
            self._v4l2_set_focus(v)
        else:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.cap.set(cv2.CAP_PROP_FOCUS, v)
        self._focus = v

    def get_focus(self) -> float:
        if self.use_v4l2:
            try:
                return self._v4l2_get_focus()
            except Exception:
                return self._focus
        return float(self.cap.get(cv2.CAP_PROP_FOCUS) or self._focus)

    def get_focus_range(self) -> Tuple[float, float]:
        return (self.focus_min, self.focus_max)


@dataclass
class SimulatedCamera(CameraInterface):
    image_path: Optional[str] = None
    focus_min: float = 0.0
    focus_max: float = 255.0
    best_focus: float = 128.0
    blur_scale: float = 25.0  # larger = blur grows slower with |z - z*|
    noise_std: float = 0.0

    def __post_init__(self) -> None:
        if self.image_path and os.path.isfile(self.image_path):
            img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read image: {self.image_path}")
            self._sharp_img = img
        else:
            # Generate a synthetic high-frequency pattern
            h, w = 720, 960
            yy, xx = np.mgrid[0:h, 0:w]
            pattern = ((np.sin(xx / 3.0) + np.sin(yy / 2.0)) * 0.5 + 0.5) * 255
            pattern = pattern.astype(np.uint8)
            self._sharp_img = cv2.merge([pattern, pattern, pattern])
        self._focus = float(self.best_focus)

    def get_frame(self) -> np.ndarray:
        # Blur proportional to distance from best_focus
        dist = abs(self._focus - self.best_focus)
        sigma = max(0.0, dist / self.blur_scale)
        if sigma < 0.1:
            img = self._sharp_img.copy()
        else:
            k = int(max(3, 2 * int(2 * sigma) + 1))
            img = cv2.GaussianBlur(self._sharp_img, (k, k), sigmaX=sigma, sigmaY=sigma)
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img

    def set_focus(self, value: float) -> None:
        self._focus = max(self.focus_min, min(self.focus_max, float(value)))

    def get_focus(self) -> float:
        return self._focus

    def get_focus_range(self) -> Tuple[float, float]:
        return (self.focus_min, self.focus_max)


@dataclass
class NoopStage(StageInterface):
    x: float = 0.0
    y: float = 0.0

    def move_xy(self, x: float, y: float) -> None:
        self.x, self.y = float(x), float(y)

    def get_xy(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class V4L2PanTiltStage(StageInterface):
    """Stage driver skeleton mapped to V4L2 pan/tilt controls.

    This is a convenience for development using cameras that expose pan/tilt via
    V4L2 (e.g., webcams). It treats pan as X and tilt as Y.
    """

    device: str = "/dev/video0"
    pan_min: int = -36000
    pan_max: int = 36000
    tilt_min: int = -36000
    tilt_max: int = 36000
    _x: float = 0.0
    _y: float = 0.0

    def move_xy(self, x: float, y: float) -> None:
        xi = int(round(max(self.pan_min, min(self.pan_max, x))))
        yi = int(round(max(self.tilt_min, min(self.tilt_max, y))))
        try:
            subprocess.run([
                "v4l2-ctl", "-d", self.device,
                "-c", f"pan_absolute={xi}",
                "-c", f"tilt_absolute={yi}",
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            raise RuntimeError(f"v4l2-ctl pan/tilt failed: {e}")
        self._x, self._y = float(xi), float(yi)

    def get_xy(self) -> Tuple[float, float]:
        try:
            p = subprocess.run(["v4l2-ctl", "-d", self.device, "-C", "pan_absolute"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            pan = int(p.stdout.split(":")[1])
            t = subprocess.run(["v4l2-ctl", "-d", self.device, "-C", "tilt_absolute"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            tilt = int(t.stdout.split(":")[1])
            self._x, self._y = float(pan), float(tilt)
        except Exception:
            pass
        return (self._x, self._y)


# ---------------------- CLI ----------------------


METRICS: dict[str, Callable[[np.ndarray], float]] = {
    "tenengrad": tenengrad,
    "lapvar": variance_of_laplacian,
    "brenner": brenner_gradient,
    "hf": high_frequency_energy,
}


def fused_metric_builder(primary: str, secondary: str | None, alpha: float, normalize: bool) -> Callable[[np.ndarray], float]:
    m1 = METRICS[primary]
    m2 = METRICS[secondary] if secondary else None

    def _normalize(img: np.ndarray) -> np.ndarray:
        if not normalize:
            return img
        # divide by local mean to mitigate illumination/stain
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        mean = float(gray.mean()) or 1.0
        scale = 128.0 / mean
        img2 = np.clip(img.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        return img2

    def metric(img: np.ndarray) -> float:
        im = _normalize(img)
        v1 = float(m1(im))
        if m2 is None:
            return v1
        v2 = float(m2(im))
        # Match scales on-the-fly to keep contributions comparable
        scale = v1 / (v2 + 1e-9)
        return (1 - alpha) * v1 + alpha * (scale * v2)

    return metric


def build_strategy(name: str, camera: CameraInterface, args, diag: DiagnosticsLogger | None) -> object:
    metric = fused_metric_builder(args.metric, args.metric2, args.alpha, args.normalize)
    if name == "coarse2fine":
        return CoarseToFineStrategy(
            camera=camera,
            bracket=args.bracket,
            coarse=args.coarse,
            fine=args.fine,
            metric=metric,
            settle_time_s=args.settle,
            diag=diag,
        )
    elif name == "contrast":
        return ContrastMaximizationStrategy(
            camera=camera,
            step=args.step,
            metric=metric,
            settle_time_s=args.settle,
            diag=diag,
        )
    else:
        raise ValueError(f"Unknown strategy: {name}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-focus CLI (Mac + Ubuntu)")
    p.add_argument("--mode", choices=["focus", "surface", "qa"], default="focus", help="Operation mode")
    sp = p.add_subparsers(dest="driver", required=True)

    # OpenCV/UVC driver (works on many USB cams; focus control may vary)
    uvc = sp.add_parser("uvc", help="OpenCV/UVC camera driver")
    uvc.add_argument("--device", type=int, default=0, help="Camera index (default 0)")
    uvc.add_argument("--width", type=int, default=None)
    uvc.add_argument("--height", type=int, default=None)
    uvc.add_argument("--focus-min", type=float, default=0.0)
    uvc.add_argument("--focus-max", type=float, default=255.0)
    uvc.add_argument("--use-v4l2", action="store_true", help="Use v4l2-ctl for focus (Linux)")
    uvc.add_argument("--v4l2-dev", type=str, default="/dev/video0")
    uvc.add_argument("--lock-exposure", action="store_true", help="Attempt to lock exposure (manual)")
    uvc.add_argument("--exposure", type=float, default=None, help="Exposure value (backend-specific)")

    # Simulated driver (for macOS or dev without hardware)
    sim = sp.add_parser("sim", help="Simulated camera driver (uses blurred sharp image)")
    sim.add_argument("--image", type=str, default=None, help="Path to sharp reference image")
    sim.add_argument("--focus-min", type=float, default=0.0)
    sim.add_argument("--focus-max", type=float, default=255.0)
    sim.add_argument("--best-focus", type=float, default=128.0)
    sim.add_argument("--blur-scale", type=float, default=25.0)
    sim.add_argument("--noise-std", type=float, default=0.0)

    # Strategy and metrics
    p.add_argument("--strategy", choices=["coarse2fine", "contrast"], default="coarse2fine")
    p.add_argument("--metric", choices=list(METRICS.keys()), default="tenengrad")
    p.add_argument("--metric2", choices=["none"] + list(METRICS.keys()), default="none")
    p.add_argument("--alpha", type=float, default=0.5, help="Weight for secondary metric [0..1]")
    p.add_argument("--normalize", action="store_true", help="Normalize input for metric computation")
    # Coarse2fine params
    p.add_argument("--bracket", type=float, default=8.0)
    p.add_argument("--coarse", type=float, default=2.0)
    p.add_argument("--fine", type=float, default=0.3)
    # Contrast params
    p.add_argument("--step", type=float, default=5.0)
    # Common
    p.add_argument("--settle", type=float, default=0.02, help="Settle time after focus move (s)")
    p.add_argument("--timeout", type=float, default=None, help="Overall timeout (s)")
    p.add_argument("--log-csv", type=str, default=None, help="Diagnostics CSV path for z/metric curves")
    # Predictive surface options
    p.add_argument("--surface-cache", type=str, default=".autofocus_cache/surface.json", help="Path to save/load surface model")
    p.add_argument("--surface-model", choices=["plane", "quad", "rbf"], default="quad")
    p.add_argument("--rbf-eps", type=float, default=200.0, help="RBF epsilon (length scale)")
    p.add_argument("--rbf-auto", action="store_true", help="Automatically select RBF epsilon by LOOCV")
    p.add_argument("--rbf-eps-cands", type=str, default="100,150,200,250,300", help="Comma-separated epsilon candidates for auto selection")
    p.add_argument("--bbox", type=float, nargs=4, metavar=("x0", "y0", "x1", "y1"), help="Tile bounding box for surface build")
    p.add_argument("--grid", type=int, default=4, help="Grid size per side for surface build")
    p.add_argument("--use-surface", action="store_true", help="Use cached surface to seed z and check outliers")
    p.add_argument("--outlier-thresh", type=float, default=8.0, help="If |z-best - z-pred| exceeds this, redo coarse sweep")
    p.add_argument("--tile-id", type=str, default=None, help="Tile ID to key predictive surface cache")
    p.add_argument("--surface-residuals-csv", type=str, default=None, help="Write surface fit residuals to CSV")
    p.add_argument("--curve-csv", type=str, default=None, help="Write coarse sweep z,value curve to CSV (coarse2fine)")
    # Stage selection for surface building
    p.add_argument("--stage", choices=["noop", "v4l2pantilt", "gcode"], default="noop")
    p.add_argument("--stage-dev", type=str, default="/dev/video0")
    p.add_argument("--stage-pan-range", type=int, nargs=2, metavar=("min", "max"), default=[-36000, 36000])
    p.add_argument("--stage-tilt-range", type=int, nargs=2, metavar=("min", "max"), default=[-36000, 36000])
    p.add_argument("--stage-port", type=str, default="/dev/ttyUSB0")
    p.add_argument("--stage-baud", type=int, default=115200)
    p.add_argument("--stage-feed", type=float, default=None)
    p.add_argument("--stage-ok-timeout", type=float, default=5.0)
    p.add_argument("--stage-wait-ok", action="store_true")
    p.add_argument("--stage-status-poll", action="store_true")
    p.add_argument("--accept-min-frac", type=float, default=0.0, help="Acceptance: best metric must be >= frac of coarse max (coarse2fine)")
    # QA options
    p.add_argument("--repeats", type=int, default=10, help="QA: repeat autofocus N times")
    p.add_argument("--rand-offset", type=float, default=4.0, help="QA: random start offset range")
    p.add_argument("--dense-ref", type=float, default=None, help="QA: dense sweep step for ground truth (sim only)")
    p.add_argument("--run-json", type=str, default=None, help="Write run summary JSON (append)")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    metric2 = None if args.metric2 == "none" else args.metric2
    # Build camera
    if args.driver == "uvc":
        cam = OpenCVUvcCamera(
            device_index=args.device,
            width=args.width,
            height=args.height,
            focus_min=args.focus_min,
            focus_max=args.focus_max,
            use_v4l2=args.use_v4l2,
            v4l2_dev=args.v4l2_dev,
            lock_exposure=getattr(args, "lock_exposure", False),
            exposure_value=getattr(args, "exposure", None),
        )
    elif args.driver == "sim":
        cam = SimulatedCamera(
            image_path=args.image,
            focus_min=args.focus_min,
            focus_max=args.focus_max,
            best_focus=args.best_focus,
            blur_scale=args.blur_scale,
            noise_std=args.noise_std,
        )
    else:
        raise ValueError("Unknown driver")

    diag = DiagnosticsLogger(args.log_csv) if args.log_csv else None
    strategy = build_strategy(args.strategy, cam, args, diag)
    controller = AutoFocusController(camera=cam, strategy=strategy)

    # Noop XY stage placeholder; replace with your hardware driver
    if args.stage == "noop":
        stage = NoopStage()
    elif args.stage == "v4l2pantilt":
        stage = V4L2PanTiltStage(
            device=args.stage_dev,
            pan_min=args.stage_pan_range[0],
            pan_max=args.stage_pan_range[1],
            tilt_min=args.stage_tilt_range[0],
            tilt_max=args.stage_tilt_range[1],
        )
    elif args.stage == "gcode":
        stage = SerialGCodeStage(
            port=args.stage_port,
            baud=args.stage_baud,
            feed_xy=args.stage_feed,
            wait_ok=args.stage_wait_ok,
            use_status_poll=args.stage_status_poll,
            ok_timeout_s=args.stage_ok_timeout,
        )
    else:
        raise SystemExit("Unknown stage")

    if args.mode == "focus":
        # Optionally use predictive surface
        z_pred = None
        if args.use_surface and os.path.exists(args.surface_cache):
            if args.tile_id:
                try:
                    model, _ = load_surface_from_cache(args.surface_cache, args.tile_id)
                except Exception as e:
                    model, _ = load_surface(args.surface_cache)
            else:
                model, _ = load_surface(args.surface_cache)
            x, y = stage.get_xy()
            z_pred = model.predict(x, y)
            cam.set_focus(z_pred)

        t0 = time.perf_counter()
        best = controller.autofocus(timeout_s=args.timeout)
        dt = (time.perf_counter() - t0) * 1000.0
        # Outlier check
        if args.use_surface and os.path.exists(args.surface_cache) and z_pred is not None:
            if abs(best - z_pred) > args.outlier_thresh:
                # redo once with full bracket around current
                best = controller.autofocus(timeout_s=args.timeout)
                retried = True
            else:
                retried = False
        else:
            retried = False
        # Acceptance check for coarse2fine
        if args.accept_min_frac > 0 and isinstance(strategy, CoarseToFineStrategy):
            if strategy.last_max_coarse is not None and strategy.last_best_value is not None:
                if strategy.last_best_value < args.accept_min_frac * strategy.last_max_coarse:
                    print("Warning: acceptance check failed (sharpness below threshold)")
                    accept_pass = False
                else:
                    accept_pass = True
            else:
                accept_pass = True
        else:
            accept_pass = True
        # Export coarse curve if requested
        if args.curve_csv and isinstance(strategy, CoarseToFineStrategy):
            if strategy.last_coarse_zs and strategy.last_coarse_vals:
                os.makedirs(os.path.dirname(args.curve_csv) or ".", exist_ok=True)
                with open(args.curve_csv, "w") as f:
                    f.write("z,value\n")
                    for z, v in zip(strategy.last_coarse_zs, strategy.last_coarse_vals):
                        f.write(f"{z:.6f},{v:.6f}\n")

        print(f"best_focus={best:.3f}  elapsed_ms={dt:.1f}")
        # Run summary JSON
        if args.run_json:
            os.makedirs(os.path.dirname(args.run_json) or ".", exist_ok=True)
            rec = {
                "ts": time.time(),
                "mode": "focus",
                "driver": args.driver,
                "strategy": args.strategy,
                "metric": args.metric,
                "metric2": metric2,
                "alpha": args.alpha,
                "elapsed_ms": dt,
                "best_focus": float(best),
                "z_pred": float(z_pred) if z_pred is not None else None,
                "outlier_thresh": args.outlier_thresh,
                "retried": retried,
                "accept_min_frac": args.accept_min_frac,
                "accept_pass": accept_pass,
                "coarse_max": getattr(strategy, "last_max_coarse", None) if isinstance(strategy, CoarseToFineStrategy) else None,
                "final_metric": getattr(strategy, "last_best_value", None) if isinstance(strategy, CoarseToFineStrategy) else None,
                "curve_csv": args.curve_csv,
                "diag_csv": args.log_csv,
            }
            try:
                import json
                if os.path.exists(args.run_json):
                    with open(args.run_json, "r") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        data.append(rec)
                    else:
                        data = [data, rec]
                else:
                    data = [rec]
                with open(args.run_json, "w") as f:
                    json.dump(data, f)
            except Exception:
                pass
        if diag:
            diag.close()
        return 0

    elif args.mode == "surface":
        if not args.bbox:
            raise SystemExit("--bbox x0 y0 x1 y1 required for surface mode")
        builder = FocusSurfaceBuilder(
            camera=cam,
            stage=stage,
            autofocus_fn=lambda z_guess: controller.autofocus(timeout_s=args.timeout),
        )
        model_kind = args.surface_model
        if model_kind == "rbf" and args.rbf_auto:
            # Auto-select epsilon by LOOCV over candidate list
            try:
                eps_cands = [float(x) for x in args.rbf_eps_cands.split(",") if x]
            except Exception:
                eps_cands = [args.rbf_eps]
            # Build samples to choose eps: we first gather with default autofocus
            model_tmp, samples_tmp = builder.build(tuple(args.bbox), grid=args.grid, model="plane")
            xs = [x for x, y, z in samples_tmp]
            ys = [y for x, y, z in samples_tmp]
            zs = [z for x, y, z in samples_tmp]
            from bloodwork_ai.vision.autofocus import select_rbf_epsilon, fit_rbf
            best_eps, best_rmse, scores = select_rbf_epsilon(xs, ys, zs, eps_cands)
            model = fit_rbf(xs, ys, zs, best_eps)
            samples = samples_tmp
            rbf_info = {"best_eps": best_eps, "cv_rmse": best_rmse, "scores": scores}
        else:
            model, samples = builder.build(tuple(args.bbox), grid=args.grid, model=model_kind)
            rbf_info = None
        # Save cache by tile-id or to a single file
        if args.tile_id:
            save_surface_cache(args.surface_cache, args.tile_id, model, samples)
        else:
            save_surface(model, samples, args.surface_cache)
        # Residuals CSV
        if args.surface_residuals_csv:
            os.makedirs(os.path.dirname(args.surface_residuals_csv) or ".", exist_ok=True)
            with open(args.surface_residuals_csv, "w") as f:
                f.write("x,y,z_true,z_pred,residual\n")
                for (x, y, z) in samples:
                    z_pred = model.predict(x, y)
                    res = z - z_pred
                    f.write(f"{x:.6f},{y:.6f},{z:.6f},{z_pred:.6f},{res:.6f}\n")
        print(f"Surface saved to {args.surface_cache} with {len(samples)} samples (model={model.kind})")
        # Run summary JSON
        if args.run_json:
            os.makedirs(os.path.dirname(args.run_json) or ".", exist_ok=True)
            resids = [float((z - model.predict(x, y))) for (x, y, z) in samples]
            summary = {
                "ts": time.time(),
                "mode": "surface",
                "model": model.kind,
                "n_samples": len(samples),
                "residual_median": float(np.median(resids)),
                "residual_p95": float(np.percentile(resids, 95)),
                "tile_id": args.tile_id,
                "surface_cache": args.surface_cache,
                "rbf_info": rbf_info,
                "surface_residuals_csv": args.surface_residuals_csv,
            }
            try:
                import json
                if os.path.exists(args.run_json):
                    with open(args.run_json, "r") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        data.append(summary)
                    else:
                        data = [data, summary]
                else:
                    data = [summary]
                with open(args.run_json, "w") as f:
                    json.dump(data, f)
            except Exception:
                pass
        if diag:
            diag.close()
        return 0

    elif args.mode == "qa":
        results = []
        true_best = getattr(cam, "best_focus", None) if isinstance(cam, SimulatedCamera) else None
        for i in range(args.repeats):
            # randomize start focus
            fmin, fmax = cam.get_focus_range()
            offset = np.random.uniform(-args.rand_offset, args.rand_offset)
            start = float(np.clip(cam.get_focus() + offset, fmin, fmax))
            cam.set_focus(start)
            t0 = time.perf_counter()
            best = controller.autofocus(timeout_s=args.timeout)
            dt = (time.perf_counter() - t0) * 1000.0
            err = None
            if true_best is not None:
                err = float(abs(best - true_best))
            results.append((best, dt, err))

        bests = np.array([r[0] for r in results], dtype=np.float64)
        times = np.array([r[1] for r in results], dtype=np.float64)
        errs = np.array([r[2] for r in results if r[2] is not None], dtype=np.float64) if any(r[2] is not None for r in results) else None
        summary = {
            "mean_z": float(bests.mean()),
            "std_z": float(bests.std(ddof=1)) if len(bests) > 1 else 0.0,
            "median_time_ms": float(np.median(times)),
            "p95_time_ms": float(np.percentile(times, 95)),
            "mean_abs_err": float(errs.mean()) if errs is not None else None,
            "p95_abs_err": float(np.percentile(errs, 95)) if errs is not None else None,
        }
        print("QA summary:", summary)
        if args.run_json:
            try:
                import json
                os.makedirs(os.path.dirname(args.run_json) or ".", exist_ok=True)
                rec = {"ts": time.time(), "mode": "qa", "summary": summary}
                if os.path.exists(args.run_json):
                    with open(args.run_json, "r") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        data.append(rec)
                    else:
                        data = [data, rec]
                else:
                    data = [rec]
                with open(args.run_json, "w") as f:
                    json.dump(data, f)
            except Exception:
                pass
        if diag:
            diag.close()
        return 0

    else:
        raise SystemExit("Unknown mode")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
