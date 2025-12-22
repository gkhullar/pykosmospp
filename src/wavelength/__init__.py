"""
Wavelength calibration module for pykosmos-spec-ai.
"""

from .identify import detect_arc_lines
from .match import load_linelist, match_lines_to_catalog
from .fit import fit_wavelength_solution
from .apply import apply_wavelength_to_spectrum

__all__ = [
    'detect_arc_lines',
    'load_linelist',
    'match_lines_to_catalog',
    'fit_wavelength_solution',
    'apply_wavelength_to_spectrum',
]


__all__ = ['identify', 'match', 'fit', 'apply']
