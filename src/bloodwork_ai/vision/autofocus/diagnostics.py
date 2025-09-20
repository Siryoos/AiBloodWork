from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class DiagnosticsLogger:
    """Lightweight CSV logger for autofocus diagnostics and acceptance checks.

    Each log_measurement() call appends one row with timestamp, phase, z, value,
    and any extra metadata provided (e.g., ROI layout, strategy parameters).
    """

    csv_path: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)
    _writer: Optional[csv.DictWriter] = field(default=None, init=False, repr=False)
    _fh: Optional[Any] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.csv_path is not None:
            dirname = os.path.dirname(self.csv_path)
            if dirname and not os.path.isdir(dirname):
                os.makedirs(dirname, exist_ok=True)
            new_file = not os.path.exists(self.csv_path)
            self._fh = open(self.csv_path, "a", newline="")
            # Add common field names that might be used
            fieldnames = ["ts", "phase", "z", "value", "x", "y", "duration_s", "error"] + sorted(self.extras.keys())
            self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames, extrasaction='ignore')
            if new_file:
                self._writer.writeheader()

    def log_measurement(self, *, phase: str, z: float, value: float, **kwargs: Any) -> None:
        if self._writer is None:
            return
        row = {"ts": f"{time.time():.6f}", "phase": phase, "z": float(z), "value": float(value)}
        # prefer per-call kwargs; fall back to global extras
        for k, v in {**self.extras, **kwargs}.items():
            row[k] = v
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None

