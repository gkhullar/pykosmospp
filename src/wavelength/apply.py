"""
Apply wavelength calibration to extracted spectra.

Per tasks.md T035: Evaluate wavelength solution at pixel positions.
"""

import numpy as np
from specutils import Spectrum1D
import astropy.units as u

from ..models import WavelengthSolution


def apply_wavelength_to_spectrum(flux: np.ndarray,
                                 uncertainty: np.ndarray,
                                 wavelength_solution: WavelengthSolution,
                                 pixel_array: np.ndarray = None) -> Spectrum1D:
    """
    Apply wavelength calibration to 1D spectrum.
    
    Creates specutils Spectrum1D object with wavelength axis from solution.
    
    Parameters
    ----------
    flux : np.ndarray
        Flux array (arbitrary units or electrons)
    uncertainty : np.ndarray
        Uncertainty array (same units as flux)
    wavelength_solution : WavelengthSolution
        Fitted wavelength solution
    pixel_array : np.ndarray, optional
        Pixel positions. If None, uses np.arange(len(flux))
        
    Returns
    -------
    Spectrum1D
        Wavelength-calibrated spectrum with flux and uncertainty
    """
    if pixel_array is None:
        pixel_array = np.arange(len(flux))
    
    # Evaluate wavelength solution at pixel positions
    wavelengths = wavelength_solution.wavelength(pixel_array)
    
    # Create Spectrum1D with wavelength axis
    from astropy.nddata import StdDevUncertainty
    
    spectrum = Spectrum1D(
        spectral_axis=wavelengths * u.Angstrom,
        flux=flux * u.electron,  # or u.adu depending on calibration state
        uncertainty=StdDevUncertainty(uncertainty * u.electron)
    )
    
    # Add metadata
    spectrum.meta = {
        'wavelength_rms': wavelength_solution.rms_residual,
        'n_arc_lines': wavelength_solution.n_lines_identified,
        'poly_order': wavelength_solution.order,
        'poly_type': wavelength_solution.poly_type,
        'wavelength_range': wavelength_solution.wavelength_range,
    }
    
    return spectrum
