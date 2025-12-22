"""
Apply wavelength calibration to extracted spectra.

Per tasks.md T035: Evaluate wavelength solution at pixel positions.
Per tasks.md T106: Spectral binning support for improved SNR.
"""

import numpy as np
from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler
import astropy.units as u
from typing import Optional

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


def bin_spectral(spectrum: Spectrum1D,
                bin_width_angstrom: float = 2.0) -> Spectrum1D:
    """
    Bin spectrum spectrally to improve SNR.
    
    Per tasks.md T106 and research.md §6: Post-extraction spectral binning
    using flux-conserving resampling from specutils.
    
    Parameters
    ----------
    spectrum : Spectrum1D
        Input spectrum with wavelength calibration
    bin_width_angstrom : float, optional
        Bin width in Angstroms (default: 2.0 Å)
        
    Returns
    -------
    Spectrum1D
        Binned spectrum with reduced spectral resolution
        
    Notes
    -----
    Spectral binning trades spectral resolution for SNR improvement.
    SNR improves by approximately √(R_old/R_new) where R is resolving power.
    
    Recommended for:
    - Faint sources where spectral resolution is not critical
    - Continuum studies
    - Broad emission/absorption feature analysis
    
    Uses flux-conserving resampling to preserve total flux.
    
    Examples
    --------
    >>> # Bin to 5 Angstrom resolution for faint galaxy continuum
    >>> binned_spectrum = bin_spectral(spectrum, bin_width_angstrom=5.0)
    >>> print(f"Original: {len(spectrum.flux)} pixels")
    >>> print(f"Binned: {len(binned_spectrum.flux)} pixels")
    """
    # Get wavelength range
    wave_min = spectrum.spectral_axis.min().to(u.Angstrom).value
    wave_max = spectrum.spectral_axis.max().to(u.Angstrom).value
    
    # Create new wavelength grid
    n_bins = int((wave_max - wave_min) / bin_width_angstrom)
    new_wavelengths = np.linspace(wave_min, wave_max, n_bins) * u.Angstrom
    
    # Use flux-conserving resampler from specutils
    resampler = FluxConservingResampler()
    binned_spectrum = resampler(spectrum, new_wavelengths)
    
    # Update metadata
    if hasattr(binned_spectrum, 'meta'):
        binned_spectrum.meta['spectral_bin_width'] = bin_width_angstrom
        binned_spectrum.meta['original_n_pixels'] = len(spectrum.flux)
        binned_spectrum.meta['binned_n_pixels'] = len(binned_spectrum.flux)
    
    return binned_spectrum

