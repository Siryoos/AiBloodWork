from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable, Optional, List, Tuple
import numpy as np


@runtime_checkable
class TemperatureSensor(Protocol):
    """Interface for temperature sensors."""

    def get_temperature(self) -> float:
        """Return current temperature in Celsius."""


@dataclass
class ThermalCompensationModel:
    """Linear thermal compensation model for focus drift.

    Model: Δz = a * ΔT + b
    where ΔT is temperature change from reference.
    """

    # Thermal coefficient (µm/°C)
    thermal_coeff: float = 0.0
    # Offset (µm)
    offset: float = 0.0
    # Reference temperature (°C)
    ref_temperature: float = 20.0

    def predict_drift(self, current_temp: float) -> float:
        """Predict focus drift based on temperature change."""
        delta_t = current_temp - self.ref_temperature
        return self.thermal_coeff * delta_t + self.offset


@dataclass
class DriftTracker:
    """Tracks and predicts focus drift over time and temperature."""

    temperature_sensor: Optional[TemperatureSensor] = None
    thermal_model: ThermalCompensationModel = field(default_factory=ThermalCompensationModel)
    max_history: int = 100

    # Internal state
    _focus_history: List[Tuple[float, float, float]] = field(default_factory=list)  # (time, temp, z)
    _last_calibration_time: float = 0.0
    _calibration_interval: float = 300.0  # 5 minutes

    def record_focus(self, z_position: float, temperature: Optional[float] = None) -> None:
        """Record a focus measurement with timestamp and temperature."""
        timestamp = time.time()

        if temperature is None and self.temperature_sensor is not None:
            temperature = self.temperature_sensor.get_temperature()
        elif temperature is None:
            temperature = self.thermal_model.ref_temperature

        self._focus_history.append((timestamp, temperature, z_position))

        # Limit history size
        if len(self._focus_history) > self.max_history:
            self._focus_history = self._focus_history[-self.max_history:]

    def get_predicted_focus(self, base_z: float, current_temp: Optional[float] = None) -> float:
        """Get temperature-compensated focus prediction."""
        if current_temp is None and self.temperature_sensor is not None:
            current_temp = self.temperature_sensor.get_temperature()
        elif current_temp is None:
            current_temp = self.thermal_model.ref_temperature

        drift = self.thermal_model.predict_drift(current_temp)
        return base_z + drift

    def update_thermal_model(self) -> bool:
        """Update thermal compensation model from recent history.

        Returns:
            True if model was updated, False if insufficient data.
        """
        current_time = time.time()

        # Only update if enough time has passed
        if current_time - self._last_calibration_time < self._calibration_interval:
            return False

        if len(self._focus_history) < 5:
            return False

        # Extract recent data
        times = np.array([h[0] for h in self._focus_history])
        temps = np.array([h[1] for h in self._focus_history])
        focuses = np.array([h[2] for h in self._focus_history])

        # Only use data from last hour for thermal model
        recent_mask = times > (current_time - 3600)
        if np.sum(recent_mask) < 3:
            return False

        temps_recent = temps[recent_mask]
        focuses_recent = focuses[recent_mask]

        # Fit linear model: z = a*T + b
        if len(set(temps_recent)) < 2:  # Need temperature variation
            return False

        try:
            coeffs = np.polyfit(temps_recent - self.thermal_model.ref_temperature,
                              focuses_recent, 1)
            self.thermal_model.thermal_coeff = float(coeffs[0])
            self.thermal_model.offset = float(coeffs[1])
            self._last_calibration_time = current_time
            return True
        except Exception:
            return False

    def get_drift_statistics(self) -> dict:
        """Get statistics about recent focus drift."""
        if len(self._focus_history) < 2:
            return {"status": "insufficient_data"}

        times = np.array([h[0] for h in self._focus_history])
        temps = np.array([h[1] for h in self._focus_history])
        focuses = np.array([h[2] for h in self._focus_history])

        current_time = time.time()
        recent_mask = times > (current_time - 1800)  # Last 30 minutes

        if np.sum(recent_mask) < 2:
            return {"status": "insufficient_recent_data"}

        recent_focuses = focuses[recent_mask]
        recent_temps = temps[recent_mask]

        return {
            "status": "ok",
            "focus_std": float(np.std(recent_focuses)),
            "focus_range": float(np.ptp(recent_focuses)),
            "temp_range": float(np.ptp(recent_temps)),
            "thermal_coeff": self.thermal_model.thermal_coeff,
            "n_samples": int(np.sum(recent_mask))
        }


@dataclass
class GlobalFocusOffset:
    """Tracks global focus offset that slowly varies with time and conditions."""

    _offset: float = 0.0
    _last_update: float = 0.0
    _update_interval: float = 600.0  # 10 minutes
    _offset_history: List[Tuple[float, float]] = field(default_factory=list)  # (time, offset)
    max_history: int = 50

    def update_offset(self, measured_offset: float) -> None:
        """Update global offset based on recent measurements."""
        current_time = time.time()

        # Add to history
        self._offset_history.append((current_time, measured_offset))
        if len(self._offset_history) > self.max_history:
            self._offset_history = self._offset_history[-self.max_history:]

        # Update global offset with exponential moving average
        if current_time - self._last_update > self._update_interval:
            recent_offsets = [h[1] for h in self._offset_history
                            if h[0] > current_time - 3600]  # Last hour

            if recent_offsets:
                # Use median to reduce outlier impact
                new_offset = float(np.median(recent_offsets))
                # Exponential moving average
                alpha = 0.1
                self._offset = (1 - alpha) * self._offset + alpha * new_offset
                self._last_update = current_time

    def get_offset(self) -> float:
        """Get current global focus offset."""
        return self._offset

    def get_statistics(self) -> dict:
        """Get offset statistics."""
        if not self._offset_history:
            return {"status": "no_data"}

        current_time = time.time()
        recent_offsets = [h[1] for h in self._offset_history
                         if h[0] > current_time - 3600]

        if not recent_offsets:
            return {"status": "no_recent_data"}

        return {
            "status": "ok",
            "current_offset": self._offset,
            "recent_std": float(np.std(recent_offsets)),
            "recent_range": float(np.ptp(recent_offsets)),
            "n_samples": len(recent_offsets)
        }