"""
Trace detection and spectral extraction module for pykosmos-spec-ai.
"""

from .trace import detect_traces_cross_correlation
from .profile import fit_spatial_profile
from .sky import estimate_sky_background
from .extract import extract_optimal

__all__ = [
    'detect_traces_cross_correlation',
    'fit_spatial_profile',
    'estimate_sky_background',
    'extract_optimal',
]
