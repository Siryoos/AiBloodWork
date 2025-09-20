Auto-Focus Module
=================

Overview
--------
- Provides a typed `CameraInterface`, focus sharpness metrics, and a pluggable
  strategy to automatically set the camera focus to maximize image sharpness.
- Default strategy uses contrast-based maximization via variance of Laplacian
  (and optionally Tenengrad) computed with OpenCV.

Structure
---------
- `camera_interface.py`: Protocol describing required camera methods.
- `metrics.py`: Focus metrics (variance of Laplacian, Tenengrad).
- `strategies/contrast.py`: Hill-climbing style contrast maximization.
- `controller.py`: Thin wrapper to run a strategy end-to-end.

Usage (example)
---------------
```python
import time
import cv2
from bloodwork_ai.vision.autofocus import (
    AutoFocusController,
    ContrastMaximizationStrategy,
    variance_of_laplacian,
    CameraInterface,
)


class OpenCVCamera(CameraInterface):
    def __init__(self, device_index: int = 0) -> None:
        self.cap = cv2.VideoCapture(device_index)
        # Replace the following with actual camera focus control (e.g., V4L2)
        self._focus = 0.0
        self._min, self._max = 0.0, 255.0

    def get_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame")
        return frame

    def set_focus(self, value: float) -> None:
        # TODO: integrate hardware-specific focus control here
        self._focus = max(self._min, min(self._max, value))
        time.sleep(0.02)  # small settle time

    def get_focus(self) -> float:
        return self._focus

    def get_focus_range(self):
        return (self._min, self._max)


camera = OpenCVCamera(0)
strategy = ContrastMaximizationStrategy(
    camera=camera,
    step=8.0,
    metric=variance_of_laplacian,
    settle_time_s=0.02,
)
controller = AutoFocusController(camera=camera, strategy=strategy)
best_focus = controller.autofocus(timeout_s=2.0)
print(f"Best focus: {best_focus:.2f}")
```

Notes
-----
- Real camera focus control varies by platform and device. Replace `set_focus`/
  `get_focus` with actual driver calls (e.g., V4L2 on Linux, vendor SDKs).
- Tune `step`, `max_iters`, and `settle_time_s` for your device.
- Consider adding bounds checks and safety interlocks for motorized lenses.

CLI
---
- Focus once (with optional predictive surface and diagnostics):
  - `python scripts/autofocus.py sim --mode focus --metric tenengrad --log-csv runs/af_curve.csv`
  - `python scripts/autofocus.py uvc --mode focus --device 0 --use-surface --surface-cache .autofocus_cache/surface.json`
- Build predictive surface (plane/quad) over a tile bbox with an N×N grid:
  - `python scripts/autofocus.py sim --mode surface --bbox 0 0 1000 1000 --grid 4 --surface-model quad`
  - Saves to `.autofocus_cache/surface.json` by default
- QA harness (repeats, throughput, repeatability; sim includes absolute error):
  - `python scripts/autofocus.py sim --mode qa --repeats 20 --rand-offset 4`
