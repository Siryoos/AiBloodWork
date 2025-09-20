from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple

from ..stage_interface import StageInterface


@dataclass
class SerialGCodeStage(StageInterface):
    """Minimal serial/GCode stage driver (GRBL/Marlin-like).

    Assumptions:
    - Absolute positioning (G90) is supported.
    - Motion command: G0 X.. Y.. [F..]
    - Acknowledge: lines containing 'ok' (GRBL) or 'ok\n' (Marlin).
    - Optional status polling with '?' returning lines like '<Idle|MPos:x,y,...>'.

    Note: Requires 'pyserial'. Install with `pip install pyserial`.
    """

    port: str
    baud: int = 115200
    feed_xy: float | None = None  # mm/min or device units per min
    wait_ok: bool = True
    use_status_poll: bool = True
    status_poll_cmd: str = "?"
    ok_timeout_s: float = 5.0
    idle_keyword: str = "Idle"
    # internal
    _x: float = 0.0
    _y: float = 0.0

    def __post_init__(self) -> None:
        try:
            import serial  # type: ignore
        except Exception as e:  # pragma: no cover - import error path
            raise RuntimeError(
                "pyserial not installed. Install with `pip install pyserial`"
            ) from e
        self._ser = serial.Serial(self.port, self.baud, timeout=0.2)
        time.sleep(0.2)
        self._flush_input()
        # Absolute mode
        self._send_line("G90")
        self._drain_ok()

    def _flush_input(self) -> None:
        try:
            self._ser.reset_input_buffer()
        except Exception:
            pass

    def _send_line(self, line: str) -> None:
        data = (line.strip() + "\n").encode("ascii", errors="ignore")
        self._ser.write(data)

    def _drain_ok(self) -> None:
        if not self.wait_ok:
            return
        start = time.perf_counter()
        buf = b""
        while (time.perf_counter() - start) < self.ok_timeout_s:
            b = self._ser.readline()
            if not b:
                continue
            buf += b
            if b.strip().lower().startswith(b"ok"):
                return
            # Some firmwares echo the command before ok; keep reading
        # Timed out; not fatal but warn via exception for caller
        raise TimeoutError("Timed out waiting for ok from stage")

    def _wait_idle(self) -> None:
        if not self.use_status_poll:
            return
        start = time.perf_counter()
        while (time.perf_counter() - start) < self.ok_timeout_s:
            self._send_line(self.status_poll_cmd)
            line = self._ser.readline().decode(errors="ignore").strip()
            if not line:
                continue
            if self.idle_keyword.lower() in line.lower():
                return
            time.sleep(0.05)

    def move_xy(self, x: float, y: float) -> None:
        self._x, self._y = float(x), float(y)
        if self.feed_xy is not None:
            cmd = f"G0 X{self._x:.3f} Y{self._y:.3f} F{self.feed_xy:.1f}"
        else:
            cmd = f"G0 X{self._x:.3f} Y{self._y:.3f}"
        self._send_line(cmd)
        try:
            self._drain_ok()
        except TimeoutError:
            # Fall back to status polling if available
            pass
        self._wait_idle()

    def get_xy(self) -> Tuple[float, float]:
        # Best-effort: query a single status line and parse MPos or WPos
        if self.use_status_poll:
            self._send_line(self.status_poll_cmd)
            line = self._ser.readline().decode(errors="ignore")
            # Example: <Idle|MPos:1.234,5.678,0.000|...>
            if "MPos:" in line:
                try:
                    seg = line.split("MPos:")[1].split("|")[0]
                    parts = seg.split(",")
                    return float(parts[0]), float(parts[1])
                except Exception:
                    pass
            if "WPos:" in line:
                try:
                    seg = line.split("WPos:")[1].split("|")[0]
                    parts = seg.split(",")
                    return float(parts[0]), float(parts[1])
                except Exception:
                    pass
        return (self._x, self._y)

