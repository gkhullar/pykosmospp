"""
Flux calibration module for pyKOSMOS++.

Per tasks.md T107-T108 and FR-009: Optional flux calibration including
atmospheric extinction correction and sensitivity function application.
"""

from .extinction import apply_extinction_correction
from .sensitivity import compute_sensitivity_function, apply_sensitivity_correction

__all__ = [
    'apply_extinction_correction',
    'compute_sensitivity_function',
    'apply_sensitivity_correction',
]
