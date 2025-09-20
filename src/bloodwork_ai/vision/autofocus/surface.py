from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple, Dict, Any, Optional

import json
import os
import numpy as np

from .camera_interface import CameraInterface
from .stage_interface import StageInterface


PredictFn = Callable[[float, float], float]


@dataclass
class FocusSurfaceModel:
    kind: str  # "plane", "quad", "rbf"
    coeffs: np.ndarray  # shape depends on kind; for rbf this may be empty
    ref_origin: Tuple[float, float]  # reference origin for numerical stability
    # RBF-specific fields
    centers: Optional[np.ndarray] = None  # (N,2)
    weights: Optional[np.ndarray] = None  # (N,)
    epsilon: Optional[float] = None

    def predict(self, x: float, y: float) -> float:
        xo, yo = self.ref_origin
        X = x - xo
        Y = y - yo
        if self.kind == "plane":
            a0, ax, ay = self.coeffs
            return float(a0 + ax * X + ay * Y)
        elif self.kind == "quad":
            a0, ax, ay, axx, ayy, axy = self.coeffs
            return float(a0 + ax * X + ay * Y + axx * X * X + ayy * Y * Y + axy * X * Y)
        elif self.kind == "rbf":
            if self.centers is None or self.weights is None or self.epsilon is None:
                raise RuntimeError("RBF model missing parameters")
            X = np.asarray([X, Y])  # (2,)
            C = self.centers  # (N,2)
            d2 = np.sum((C - X) ** 2, axis=1)  # (N,)
            # Gaussian RBF
            phi = np.exp(-d2 / (2.0 * float(self.epsilon) ** 2))  # (N,)
            return float(np.dot(self.weights, phi))
        else:
            raise ValueError(f"Unknown model kind: {self.kind}")

    def to_json(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "kind": self.kind,
            "coeffs": self.coeffs.tolist(),
            "ref_origin": list(self.ref_origin),
        }
        if self.kind == "rbf":
            data["centers"] = None if self.centers is None else self.centers.tolist()
            data["weights"] = None if self.weights is None else self.weights.tolist()
            data["epsilon"] = self.epsilon
        return data

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "FocusSurfaceModel":
        kind = data["kind"]
        centers = None
        weights = None
        epsilon = None
        if kind == "rbf":
            centers = None if data.get("centers") is None else np.asarray(data["centers"], dtype=np.float64)
            weights = None if data.get("weights") is None else np.asarray(data["weights"], dtype=np.float64)
            epsilon = None if data.get("epsilon") is None else float(data["epsilon"])
        return FocusSurfaceModel(
            kind=kind,
            coeffs=np.asarray(data["coeffs"], dtype=np.float64),
            ref_origin=(float(data["ref_origin"][0]), float(data["ref_origin"][1])),
            centers=centers,
            weights=weights,
            epsilon=epsilon,
        )


def fit_plane(xs: Sequence[float], ys: Sequence[float], zs: Sequence[float]) -> FocusSurfaceModel:
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    zs = np.asarray(zs, dtype=np.float64)
    xo, yo = float(xs.mean()), float(ys.mean())
    X = xs - xo
    Y = ys - yo
    A = np.stack([np.ones_like(X), X, Y], axis=1)
    coeffs, *_ = np.linalg.lstsq(A, zs, rcond=None)
    return FocusSurfaceModel("plane", coeffs, (xo, yo))


def fit_quad(xs: Sequence[float], ys: Sequence[float], zs: Sequence[float]) -> FocusSurfaceModel:
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    zs = np.asarray(zs, dtype=np.float64)
    xo, yo = float(xs.mean()), float(ys.mean())
    X = xs - xo
    Y = ys - yo
    A = np.stack([np.ones_like(X), X, Y, X * X, Y * Y, X * Y], axis=1)
    coeffs, *_ = np.linalg.lstsq(A, zs, rcond=None)
    return FocusSurfaceModel("quad", coeffs, (xo, yo))


def fit_rbf(xs: Sequence[float], ys: Sequence[float], zs: Sequence[float], epsilon: float = 200.0) -> FocusSurfaceModel:
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    zs = np.asarray(zs, dtype=np.float64)
    centers = np.stack([xs, ys], axis=1)  # (N,2)
    # Build Gaussian RBF kernel matrix
    # Phi_ij = exp(-||pi - pj||^2 / (2*epsilon^2))
    diff = centers[:, None, :] - centers[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    Phi = np.exp(-d2 / (2.0 * float(epsilon) ** 2))
    # Regularize slightly for numerical stability
    lam = 1e-8
    A = Phi + lam * np.eye(Phi.shape[0])
    weights = np.linalg.solve(A, zs)
    return FocusSurfaceModel("rbf", coeffs=np.array([], dtype=np.float64), ref_origin=(0.0, 0.0), centers=centers, weights=weights, epsilon=float(epsilon))


def rbf_cv_rmse(xs: Sequence[float], ys: Sequence[float], zs: Sequence[float], epsilon: float, lam: float = 1e-8) -> float:
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    zs = np.asarray(zs, dtype=np.float64)
    n = len(zs)
    errs = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        xi, yi = xs[~mask], ys[~mask]
        zi = zs[~mask]
        centers = np.stack([xi, yi], axis=1)
        # kernel matrix for train
        diff = centers[:, None, :] - centers[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        Phi = np.exp(-d2 / (2.0 * float(epsilon) ** 2))
        A = Phi + lam * np.eye(Phi.shape[0])
        w = np.linalg.solve(A, zi)
        # predict left-out
        px = xs[i]
        py = ys[i]
        d2v = np.sum((centers - np.array([px, py])) ** 2, axis=1)
        phi = np.exp(-d2v / (2.0 * float(epsilon) ** 2))
        zhat = float(np.dot(w, phi))
        errs.append(float(zs[i] - zhat))
    rmse = float(np.sqrt(np.mean(np.square(errs))))
    return rmse


def select_rbf_epsilon(xs: Sequence[float], ys: Sequence[float], zs: Sequence[float], eps_candidates: Sequence[float]) -> Tuple[float, float, Dict[float, float]]:
    scores: Dict[float, float] = {}
    best_eps = None
    best_rmse = float("inf")
    for eps in eps_candidates:
        rmse = rbf_cv_rmse(xs, ys, zs, epsilon=float(eps))
        scores[float(eps)] = rmse
        if rmse < best_rmse:
            best_rmse = rmse
            best_eps = float(eps)
    assert best_eps is not None
    return best_eps, best_rmse, scores


def grid_points(x0: float, y0: float, x1: float, y1: float, n: int) -> List[Tuple[float, float]]:
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    pts = []
    for y in ys:
        for x in xs:
            pts.append((float(x), float(y)))
    return pts


@dataclass
class FocusSurfaceBuilder:
    camera: CameraInterface
    stage: StageInterface
    autofocus_fn: Callable[[float | None], float]  # takes z_guess -> returns best_z

    def build(self, tile_bbox: Tuple[float, float, float, float], grid: int = 4, model: str = "quad") -> Tuple[FocusSurfaceModel, List[Tuple[float, float, float]]]:
        x0, y0, x1, y1 = tile_bbox
        pts = grid_points(x0, y0, x1, y1, grid)
        xs: List[float] = []
        ys: List[float] = []
        zs: List[float] = []
        for (x, y) in pts:
            self.stage.move_xy(x, y)
            z_guess = float(self.camera.get_focus())
            z = float(self.autofocus_fn(z_guess))
            xs.append(x)
            ys.append(y)
            zs.append(z)

        if model == "plane":
            m = fit_plane(xs, ys, zs)
        elif model == "quad":
            m = fit_quad(xs, ys, zs)
        else:
            raise ValueError("Unsupported model: " + model)

        samples = list(zip(xs, ys, zs))
        return m, samples


def save_surface(model: FocusSurfaceModel, samples: List[Tuple[float, float, float]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {"model": model.to_json(), "samples": samples}
    with open(path, "w") as f:
        json.dump(data, f)


def load_surface(path: str) -> Tuple[FocusSurfaceModel, List[Tuple[float, float, float]]]:
    with open(path, "r") as f:
        data = json.load(f)
    model = FocusSurfaceModel.from_json(data["model"])
    samples = [(float(x), float(y), float(z)) for x, y, z in data["samples"]]
    return model, samples


def save_surface_cache(path: str, tile_id: str, model: FocusSurfaceModel, samples: List[Tuple[float, float, float]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    db: Dict[str, Any] = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                db = json.load(f)
            except Exception:
                db = {}
    db[tile_id] = {"model": model.to_json(), "samples": samples}
    with open(path, "w") as f:
        json.dump(db, f)


def load_surface_from_cache(path: str, tile_id: str) -> Tuple[FocusSurfaceModel, List[Tuple[float, float, float]]]:
    with open(path, "r") as f:
        db = json.load(f)
    if tile_id not in db:
        raise KeyError(f"tile_id {tile_id!r} not found in cache")
    entry = db[tile_id]
    model = FocusSurfaceModel.from_json(entry["model"])
    samples = [(float(x), float(y), float(z)) for x, y, z in entry["samples"]]
    return model, samples
