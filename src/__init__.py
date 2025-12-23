"""
pyKOSMOS Spectroscopic Reduction Pipeline
==========================================

A Python package for reducing APO-KOSMOS long-slit spectroscopy data,
with focus on faint galaxy observations and robust wavelength calibration.

Main components:
- calibration: Bias, flat, cosmic ray processing
- extraction: Trace detection and optimal extraction
- wavelength: Arc line identification and wavelength solutions
- flux_calibration: Standard star flux calibration and extinction correction
- quality: Validation and quality metrics
- io: FITS I/O, configuration, logging

See documentation at: specs/001-galaxy-spec-pipeline/
"""

__version__ = "0.2.1"
__author__ = "APO KOSMOS Team"

# Package-level imports will be added as modules are implemented
