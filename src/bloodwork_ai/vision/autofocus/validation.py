from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
import numpy as np
import cv2


@dataclass
class FocusAccuracyTest:
    """Test autofocus accuracy against ground truth."""

    name: str
    z_positions: List[float]  # Ground truth focus positions to test
    expected_accuracy_um: float = 1.0  # Maximum acceptable error
    expected_repeatability_um: float = 0.5  # Maximum acceptable std dev
    num_repeats: int = 10

    results: Dict[str, Any] = field(default_factory=dict)

    def run(self, autofocus_fn: Callable[[float], float], verbose: bool = True) -> Dict[str, Any]:
        """Run accuracy test.

        Args:
            autofocus_fn: Function that takes z_guess and returns best_z
            verbose: Print progress information

        Returns:
            Test results dictionary
        """
        results = {
            "test_name": self.name,
            "timestamp": time.time(),
            "positions_tested": len(self.z_positions),
            "repeats_per_position": self.num_repeats,
            "measurements": [],
            "summary": {}
        }

        all_errors = []
        all_repeatabilities = []

        for i, z_true in enumerate(self.z_positions):
            if verbose:
                print(f"Testing position {i+1}/{len(self.z_positions)}: z={z_true:.2f}um")

            position_measurements = []

            for repeat in range(self.num_repeats):
                # Add some random offset to starting guess
                z_guess = z_true + np.random.normal(0, 2.0)

                try:
                    z_measured = autofocus_fn(z_guess)
                    error = abs(z_measured - z_true)
                    position_measurements.append({
                        "repeat": repeat,
                        "z_true": z_true,
                        "z_guess": z_guess,
                        "z_measured": z_measured,
                        "error_um": error
                    })
                    all_errors.append(error)
                except Exception as e:
                    if verbose:
                        print(f"  Repeat {repeat} failed: {e}")
                    position_measurements.append({
                        "repeat": repeat,
                        "z_true": z_true,
                        "z_guess": z_guess,
                        "z_measured": None,
                        "error_um": None,
                        "error": str(e)
                    })

            # Calculate repeatability for this position
            valid_measurements = [m["z_measured"] for m in position_measurements
                                if m["z_measured"] is not None]
            if len(valid_measurements) >= 2:
                repeatability = float(np.std(valid_measurements))
                all_repeatabilities.append(repeatability)
            else:
                repeatability = None

            results["measurements"].append({
                "z_true": z_true,
                "measurements": position_measurements,
                "repeatability_um": repeatability,
                "success_rate": len(valid_measurements) / self.num_repeats
            })

        # Calculate summary statistics
        if all_errors:
            results["summary"] = {
                "mean_error_um": float(np.mean(all_errors)),
                "max_error_um": float(np.max(all_errors)),
                "error_std_um": float(np.std(all_errors)),
                "accuracy_pass": float(np.max(all_errors)) <= self.expected_accuracy_um,
                "mean_repeatability_um": float(np.mean(all_repeatabilities)) if all_repeatabilities else None,
                "max_repeatability_um": float(np.max(all_repeatabilities)) if all_repeatabilities else None,
                "repeatability_pass": (float(np.max(all_repeatabilities)) <= self.expected_repeatability_um
                                     if all_repeatabilities else False),
                "overall_pass": (float(np.max(all_errors)) <= self.expected_accuracy_um and
                               (float(np.max(all_repeatabilities)) <= self.expected_repeatability_um
                                if all_repeatabilities else False))
            }
        else:
            results["summary"] = {"error": "No successful measurements"}

        self.results = results
        return results


@dataclass
class MTFMeasurement:
    """Modulation Transfer Function measurement for image quality validation."""

    @staticmethod
    def slanted_edge_mtf(image: np.ndarray, edge_angle_deg: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
        """Measure MTF using slanted edge method.

        Args:
            image: Grayscale image containing a slanted edge
            edge_angle_deg: Expected edge angle in degrees

        Returns:
            Tuple of (frequencies, mtf_values)
        """
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h, w = image.shape

        # Find edge using gradient
        grad_x = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Find strongest edge region
        edge_strength = cv2.GaussianBlur(grad_mag, (5, 5), 1.0)
        max_pos = np.unravel_index(np.argmax(edge_strength), edge_strength.shape)

        # Extract ROI around edge
        roi_size = min(64, h//4, w//4)
        y0 = max(0, max_pos[0] - roi_size//2)
        y1 = min(h, y0 + roi_size)
        x0 = max(0, max_pos[1] - roi_size//2)
        x1 = min(w, x0 + roi_size)

        roi = image[y0:y1, x0:x1].astype(np.float32)

        # Compute edge spread function (ESF)
        # For simplicity, average along the edge direction
        if abs(edge_angle_deg) < 45:
            esf = np.mean(roi, axis=0)
        else:
            esf = np.mean(roi, axis=1)

        # Differentiate to get line spread function (LSF)
        lsf = np.diff(esf)

        # Apply window function
        window = np.hanning(len(lsf))
        lsf_windowed = lsf * window

        # FFT to get MTF
        fft_result = np.fft.fft(lsf_windowed, n=len(lsf_windowed)*4)
        mtf = np.abs(fft_result)

        # Normalize
        if mtf[0] > 0:
            mtf = mtf / mtf[0]

        # Generate frequency axis (cycles per pixel)
        freqs = np.fft.fftfreq(len(mtf), d=1.0)

        # Keep only positive frequencies up to Nyquist
        n_half = len(freqs) // 2
        freqs = freqs[:n_half]
        mtf = mtf[:n_half]

        return freqs, mtf

    @staticmethod
    def mtf50(freqs: np.ndarray, mtf: np.ndarray) -> float:
        """Find MTF50 frequency (where MTF drops to 50%)."""
        # Find where MTF crosses 0.5
        below_half = mtf < 0.5
        if not np.any(below_half):
            return float(freqs[-1])  # MTF never drops below 50%

        # Linear interpolation to find exact crossing
        idx = np.where(below_half)[0][0]
        if idx == 0:
            return 0.0

        # Interpolate between idx-1 and idx
        f0, f1 = freqs[idx-1], freqs[idx]
        m0, m1 = mtf[idx-1], mtf[idx]

        if abs(m1 - m0) < 1e-10:
            return f0

        mtf50_freq = f0 + (0.5 - m0) * (f1 - f0) / (m1 - m0)
        return float(mtf50_freq)


@dataclass
class ImageQualityValidator:
    """Validates image quality at different focus positions."""

    def measure_sharpness_curve(self,
                               camera_interface,
                               z_positions: List[float],
                               metric_fn: Callable[[np.ndarray], float],
                               settle_time_s: float = 0.02) -> Dict[str, Any]:
        """Measure sharpness vs focus position curve.

        Args:
            camera_interface: Camera interface
            z_positions: List of focus positions to test
            metric_fn: Sharpness metric function
            settle_time_s: Time to wait after focus change

        Returns:
            Dictionary with positions, sharpness values, and analysis
        """
        positions = []
        sharpness_values = []

        for z in z_positions:
            camera_interface.set_focus(z)
            time.sleep(settle_time_s)

            # Take multiple frames and average
            scores = []
            for _ in range(3):
                frame = camera_interface.get_frame()
                score = metric_fn(frame)
                scores.append(score)
                time.sleep(0.01)

            avg_score = np.mean(scores)
            positions.append(z)
            sharpness_values.append(avg_score)

        positions = np.array(positions)
        sharpness_values = np.array(sharpness_values)

        # Find peak
        peak_idx = np.argmax(sharpness_values)
        peak_z = positions[peak_idx]
        peak_sharpness = sharpness_values[peak_idx]

        # Estimate curve width (FWHM)
        half_max = peak_sharpness * 0.5
        above_half = sharpness_values >= half_max
        if np.any(above_half):
            indices = np.where(above_half)[0]
            fwhm_indices = indices[-1] - indices[0] if len(indices) > 1 else 0
            fwhm_um = float((positions[indices[-1]] - positions[indices[0]])
                          if len(indices) > 1 else 0)
        else:
            fwhm_um = 0.0

        return {
            "positions": positions.tolist(),
            "sharpness_values": sharpness_values.tolist(),
            "peak_z": float(peak_z),
            "peak_sharpness": float(peak_sharpness),
            "fwhm_um": fwhm_um,
            "curve_snr": float(peak_sharpness / np.std(sharpness_values)) if np.std(sharpness_values) > 0 else 0
        }

    def validate_mtf_performance(self,
                                camera_interface,
                                target_z: float,
                                autofocus_fn: Callable[[float], float],
                                expected_mtf50_ratio: float = 0.95) -> Dict[str, Any]:
        """Validate MTF performance at autofocus position vs ground truth.

        Args:
            camera_interface: Camera interface
            target_z: Ground truth optimal focus position
            autofocus_fn: Autofocus function
            expected_mtf50_ratio: Minimum acceptable ratio of AF MTF50 to GT MTF50

        Returns:
            Validation results
        """
        # Measure MTF at ground truth position
        camera_interface.set_focus(target_z)
        time.sleep(0.05)
        gt_frame = camera_interface.get_frame()

        try:
            gt_freqs, gt_mtf = MTFMeasurement.slanted_edge_mtf(gt_frame)
            gt_mtf50 = MTFMeasurement.mtf50(gt_freqs, gt_mtf)
        except Exception as e:
            return {"error": f"Failed to measure ground truth MTF: {e}"}

        # Run autofocus from offset position
        z_guess = target_z + np.random.normal(0, 3.0)
        try:
            af_z = autofocus_fn(z_guess)
        except Exception as e:
            return {"error": f"Autofocus failed: {e}"}

        # Measure MTF at autofocus position
        camera_interface.set_focus(af_z)
        time.sleep(0.05)
        af_frame = camera_interface.get_frame()

        try:
            af_freqs, af_mtf = MTFMeasurement.slanted_edge_mtf(af_frame)
            af_mtf50 = MTFMeasurement.mtf50(af_freqs, af_mtf)
        except Exception as e:
            return {"error": f"Failed to measure autofocus MTF: {e}"}

        # Calculate ratio and pass/fail
        mtf50_ratio = af_mtf50 / gt_mtf50 if gt_mtf50 > 0 else 0
        passed = mtf50_ratio >= expected_mtf50_ratio

        return {
            "target_z": target_z,
            "af_z": af_z,
            "focus_error_um": abs(af_z - target_z),
            "gt_mtf50": gt_mtf50,
            "af_mtf50": af_mtf50,
            "mtf50_ratio": mtf50_ratio,
            "expected_ratio": expected_mtf50_ratio,
            "passed": passed
        }


@dataclass
class AutofocusTestSuite:
    """Complete test suite for autofocus validation."""

    def __init__(self):
        self.tests: List[FocusAccuracyTest] = []
        self.results: List[Dict[str, Any]] = []

    def add_test(self, test: FocusAccuracyTest) -> None:
        """Add a test to the suite."""
        self.tests.append(test)

    def run_all_tests(self,
                     autofocus_fn: Callable[[float], float],
                     camera_interface=None,
                     verbose: bool = True) -> Dict[str, Any]:
        """Run all tests in the suite.

        Args:
            autofocus_fn: Autofocus function to test
            camera_interface: Camera interface for image quality tests
            verbose: Print progress information

        Returns:
            Combined test results
        """
        suite_results = {
            "timestamp": time.time(),
            "tests_run": len(self.tests),
            "test_results": [],
            "summary": {}
        }

        all_passed = True
        total_measurements = 0
        successful_measurements = 0

        for i, test in enumerate(self.tests):
            if verbose:
                print(f"\nRunning test {i+1}/{len(self.tests)}: {test.name}")

            result = test.run(autofocus_fn, verbose=verbose)
            suite_results["test_results"].append(result)

            # Track overall statistics
            if "summary" in result and "overall_pass" in result["summary"]:
                if not result["summary"]["overall_pass"]:
                    all_passed = False

            # Count measurements
            for pos_result in result.get("measurements", []):
                for measurement in pos_result.get("measurements", []):
                    total_measurements += 1
                    if measurement.get("z_measured") is not None:
                        successful_measurements += 1

        # Calculate suite summary
        suite_results["summary"] = {
            "all_tests_passed": all_passed,
            "total_measurements": total_measurements,
            "successful_measurements": successful_measurements,
            "success_rate": successful_measurements / total_measurements if total_measurements > 0 else 0
        }

        self.results = suite_results["test_results"]
        return suite_results

    @classmethod
    def create_standard_suite(cls) -> "AutofocusTestSuite":
        """Create a standard test suite for blood smear autofocus."""
        suite = cls()

        # Basic accuracy test
        suite.add_test(FocusAccuracyTest(
            name="basic_accuracy",
            z_positions=[-10, -5, 0, 5, 10, 15],
            expected_accuracy_um=1.0,
            expected_repeatability_um=0.5,
            num_repeats=5
        ))

        # Extended range test
        suite.add_test(FocusAccuracyTest(
            name="extended_range",
            z_positions=[-20, -15, -10, -5, 0, 5, 10, 15, 20, 25],
            expected_accuracy_um=1.5,
            expected_repeatability_um=0.7,
            num_repeats=3
        ))

        # Precision test at optimal position
        suite.add_test(FocusAccuracyTest(
            name="precision_test",
            z_positions=[0],
            expected_accuracy_um=0.5,
            expected_repeatability_um=0.3,
            num_repeats=20
        ))

        return suite