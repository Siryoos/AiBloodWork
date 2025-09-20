from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from enum import Enum


class IlluminationMode(Enum):
    """Standard illumination modes for computational photography."""
    BRIGHTFIELD = "BRIGHTFIELD"
    DARKFIELD = "DARKFIELD"
    LED_ANGLE_0 = "LED_ANGLE_0"
    LED_ANGLE_25 = "LED_ANGLE_25"
    LED_ANGLE_45 = "LED_ANGLE_45"
    LED_ANGLE_90 = "LED_ANGLE_90"
    MULTI_ANGLE_STACK = "MULTI_ANGLE_STACK"
    FOURIER_PTYCHOGRAPHY = "FOURIER_PTYCHOGRAPHY"


class ReconstructionMethod(Enum):
    """Computational photography reconstruction methods."""
    SYNTHETIC_APERTURE = "SYNTHETIC_APERTURE"
    FOURIER_PTYCHOGRAPHY = "FOURIER_PTYCHOGRAPHY"
    FOCUS_STACKING = "FOCUS_STACKING"
    PHASE_RETRIEVAL = "PHASE_RETRIEVAL"


@dataclass
class ChromaticFocusCompensation:
    """Chromatic focus compensation for different wavelengths."""

    # Reference wavelength (nm) for focus measurement
    reference_wavelength: float = 550.0

    # Focus offsets for different channels (um)
    red_offset_um: float = 0.0      # ~650nm
    green_offset_um: float = 0.0    # ~550nm (reference)
    blue_offset_um: float = 0.0     # ~450nm

    # Custom wavelength offsets
    wavelength_offsets: Dict[float, float] = field(default_factory=dict)

    def get_focus_offset(self, wavelength: float) -> float:
        """Get focus offset for given wavelength."""
        if wavelength in self.wavelength_offsets:
            return self.wavelength_offsets[wavelength]

        # Approximate chromatic aberration with simple model
        # Focus shift ~ 1/wavelength^2 (simplified)
        ref_inv_sq = 1.0 / (self.reference_wavelength ** 2)
        wave_inv_sq = 1.0 / (wavelength ** 2)

        # Linear approximation around reference
        if abs(wavelength - 650) < 50:  # Red channel
            return self.red_offset_um
        elif abs(wavelength - 450) < 50:  # Blue channel
            return self.blue_offset_um
        else:  # Green or other
            return self.green_offset_um


@dataclass
class CompPhotoConfig:
    """Configuration for computational photography integration."""

    reconstruction_method: ReconstructionMethod = ReconstructionMethod.SYNTHETIC_APERTURE
    illumination_mode: IlluminationMode = IlluminationMode.LED_ANGLE_25

    # Focus constraints for reconstruction
    defocus_tolerance_um: float = 2.0  # Max acceptable defocus for reconstruction
    focus_consistency_requirement: bool = True  # Require consistent focus across angles

    # Multi-focus capture settings
    enable_focus_stacking: bool = False
    focus_stack_range_um: float = 4.0
    focus_stack_steps: int = 5

    # Chromatic compensation
    chromatic_compensation: ChromaticFocusCompensation = field(
        default_factory=ChromaticFocusCompensation
    )

    # Illumination coupling
    af_illumination_mode: IlluminationMode = IlluminationMode.LED_ANGLE_25
    capture_illumination_modes: List[IlluminationMode] = field(default_factory=lambda: [
        IlluminationMode.LED_ANGLE_25
    ])

    # Phase retrieval constraints
    phase_retrieval_focus_accuracy_um: float = 0.5


@dataclass
class FocusStackPlan:
    """Plan for multi-focus capture."""

    center_focus_um: float
    focus_positions_um: List[float]
    illumination_sequence: List[IlluminationMode]
    expected_defocus_range_um: float

    @classmethod
    def create_symmetric_stack(cls,
                             center_z: float,
                             range_um: float,
                             num_steps: int,
                             illumination: IlluminationMode) -> "FocusStackPlan":
        """Create symmetric focus stack around center position."""
        if num_steps <= 1:
            positions = [center_z]
        else:
            half_range = range_um / 2
            positions = np.linspace(center_z - half_range,
                                  center_z + half_range,
                                  num_steps).tolist()

        return cls(
            center_focus_um=center_z,
            focus_positions_um=positions,
            illumination_sequence=[illumination] * len(positions),
            expected_defocus_range_um=range_um
        )


class CompPhotoAutofocus:
    """Autofocus integration for computational photography workflows."""

    def __init__(self, config: CompPhotoConfig):
        self.config = config

    def compute_optimal_focus(self,
                            base_af_result: float,
                            illumination_mode: IlluminationMode,
                            wavelength: Optional[float] = None) -> float:
        """Compute optimal focus for given illumination and wavelength.

        Args:
            base_af_result: Base autofocus result (typically at reference wavelength)
            illumination_mode: Target illumination mode
            wavelength: Target wavelength in nm (if None, uses reference)

        Returns:
            Optimal focus position for computational photography
        """
        optimal_z = base_af_result

        # Apply chromatic compensation
        if wavelength is not None:
            chromatic_offset = self.config.chromatic_compensation.get_focus_offset(wavelength)
            optimal_z += chromatic_offset

        # Apply illumination-specific corrections
        # Different illumination angles may have different optimal focus
        illum_offset = self._get_illumination_focus_offset(illumination_mode)
        optimal_z += illum_offset

        return optimal_z

    def _get_illumination_focus_offset(self, mode: IlluminationMode) -> float:
        """Get focus offset for different illumination modes."""
        # Different illumination angles may shift optimal focus slightly
        offsets = {
            IlluminationMode.BRIGHTFIELD: 0.0,
            IlluminationMode.DARKFIELD: 0.2,  # Slightly higher for edge contrast
            IlluminationMode.LED_ANGLE_0: 0.0,
            IlluminationMode.LED_ANGLE_25: 0.0,
            IlluminationMode.LED_ANGLE_45: 0.1,
            IlluminationMode.LED_ANGLE_90: 0.2,
        }
        return offsets.get(mode, 0.0)

    def create_focus_stack_plan(self,
                              base_af_result: float,
                              target_illumination: Optional[IlluminationMode] = None) -> FocusStackPlan:
        """Create focus stack plan for multi-focus reconstruction."""
        if not self.config.enable_focus_stacking:
            # Single focus position
            illumination = target_illumination or self.config.capture_illumination_modes[0]
            optimal_z = self.compute_optimal_focus(base_af_result, illumination)

            return FocusStackPlan(
                center_focus_um=optimal_z,
                focus_positions_um=[optimal_z],
                illumination_sequence=[illumination],
                expected_defocus_range_um=0.0
            )

        # Multi-focus stack
        center_z = base_af_result
        if target_illumination:
            center_z = self.compute_optimal_focus(base_af_result, target_illumination)

        return FocusStackPlan.create_symmetric_stack(
            center_z=center_z,
            range_um=self.config.focus_stack_range_um,
            num_steps=self.config.focus_stack_steps,
            illumination=target_illumination or self.config.capture_illumination_modes[0]
        )

    def validate_focus_for_reconstruction(self,
                                        af_results: Dict[IlluminationMode, float],
                                        target_method: Optional[ReconstructionMethod] = None) -> Dict[str, Any]:
        """Validate focus results meet reconstruction requirements.

        Args:
            af_results: Dictionary mapping illumination modes to focus positions
            target_method: Target reconstruction method

        Returns:
            Validation results with pass/fail and metrics
        """
        method = target_method or self.config.reconstruction_method

        if len(af_results) < 2:
            # Single illumination mode
            return {
                "status": "pass",
                "focus_consistency": True,
                "max_defocus_error": 0.0,
                "reconstruction_ready": True
            }

        # Check focus consistency across illumination modes
        focus_values = list(af_results.values())
        focus_range = max(focus_values) - min(focus_values)
        focus_std = float(np.std(focus_values))

        # Validation criteria depend on reconstruction method
        if method == ReconstructionMethod.FOURIER_PTYCHOGRAPHY:
            max_allowed_defocus = self.config.phase_retrieval_focus_accuracy_um
        else:
            max_allowed_defocus = self.config.defocus_tolerance_um

        consistency_pass = focus_range <= max_allowed_defocus

        # Check individual defocus errors
        mean_focus = float(np.mean(focus_values))
        max_defocus_error = max(abs(f - mean_focus) for f in focus_values)

        reconstruction_ready = (consistency_pass and
                              max_defocus_error <= max_allowed_defocus)

        return {
            "status": "pass" if reconstruction_ready else "fail",
            "focus_consistency": consistency_pass,
            "focus_range_um": focus_range,
            "focus_std_um": focus_std,
            "max_defocus_error": max_defocus_error,
            "mean_focus_um": mean_focus,
            "reconstruction_ready": reconstruction_ready,
            "method": method.value,
            "max_allowed_defocus_um": max_allowed_defocus
        }

    def get_illumination_sequence_for_af(self) -> List[IlluminationMode]:
        """Get illumination sequence for autofocus measurement.

        Returns sequence that best represents the final capture illumination.
        """
        if self.config.af_illumination_mode == IlluminationMode.MULTI_ANGLE_STACK:
            # Use all capture illumination modes for AF
            return self.config.capture_illumination_modes
        else:
            # Use single representative mode
            return [self.config.af_illumination_mode]

    def compute_synthetic_aperture_focus_target(self,
                                              af_results: Dict[IlluminationMode, float]) -> float:
        """Compute optimal focus for synthetic aperture reconstruction.

        For synthetic aperture, we want to center the defocus distribution.
        """
        if not af_results:
            return 0.0

        focus_values = list(af_results.values())

        if len(focus_values) == 1:
            return focus_values[0]

        # For synthetic aperture, use median to center the distribution
        return float(np.median(focus_values))

    def estimate_phase_retrieval_requirements(self,
                                            tile_size_um: Tuple[float, float],
                                            numerical_aperture: float) -> Dict[str, float]:
        """Estimate focus accuracy requirements for phase retrieval.

        Args:
            tile_size_um: Tile dimensions in microns (width, height)
            numerical_aperture: Objective numerical aperture

        Returns:
            Dictionary with estimated requirements
        """
        # Depth of field estimation
        wavelength_um = 0.55  # Green light
        dof_um = wavelength_um / (numerical_aperture ** 2)

        # For phase retrieval, typically need focus accuracy << DOF
        required_accuracy = dof_um / 4.0  # Quarter DOF rule

        # Tile size affects reconstruction robustness
        min_tile_dim = min(tile_size_um)
        max_acceptable_defocus = min_tile_dim / 100.0  # Empirical rule

        return {
            "estimated_dof_um": dof_um,
            "recommended_focus_accuracy_um": min(required_accuracy, max_acceptable_defocus),
            "max_acceptable_defocus_um": max_acceptable_defocus,
            "numerical_aperture": numerical_aperture,
            "wavelength_um": wavelength_um
        }