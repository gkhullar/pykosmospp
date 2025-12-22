"""
Sensitivity function computation and application.

Per tasks.md T108 and FR-009: Compute sensitivity function from standard
star observations for absolute flux calibration.
"""

import numpy as np
from specutils import Spectrum1D
import astropy.units as u
from astropy.nddata import StdDevUncertainty
from scipy.interpolate import UnivariateSpline
from typing import Optional
import warnings


def compute_sensitivity_function(observed_spectrum: Spectrum1D,
                                 standard_star_spectrum: Spectrum1D,
                                 smooth_order: int = 3,
                                 smooth_factor: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute sensitivity function from standard star observation.
    
    Per tasks.md T108 and FR-009: Compares observed standard star spectrum
    to known flux-calibrated spectrum to derive instrument sensitivity.
    
    Parameters
    ----------
    observed_spectrum : Spectrum1D
        Observed standard star spectrum (instrumental units)
    standard_star_spectrum : Spectrum1D
        Known flux-calibrated standard star spectrum
        Must have flux in physical units (erg/s/cm²/Å or similar)
    smooth_order : int, optional
        Spline order for smoothing (default: 3)
    smooth_factor : float, optional
        Smoothing factor (default: 1.0, more = smoother)
        
    Returns
    -------
    wavelengths : np.ndarray
        Wavelength array for sensitivity function
    sensitivity : np.ndarray
        Sensitivity function (conversion factor from instrumental to physical units)
        
    Notes
    -----
    Sensitivity function S(λ) is defined such that:
    
    .. math::
        F_{physical}(\\lambda) = F_{instrumental}(\\lambda) \\times S(\\lambda)
    
    The function is computed as:
    
    .. math::
        S(\\lambda) = \\frac{F_{standard}(\\lambda)}{F_{observed}(\\lambda)}
    
    where F_standard is the known flux-calibrated spectrum and F_observed
    is the observed instrumental spectrum.
    
    Examples
    --------
    >>> # Compute sensitivity from standard star
    >>> wavelengths, sensitivity = compute_sensitivity_function(
    ...     observed_std_star,
    ...     catalog_std_star
    ... )
    
    >>> # Save for later use
    >>> np.savetxt('sensitivity.dat', np.column_stack([wavelengths, sensitivity]))
    """
    # Ensure spectra are on same wavelength grid
    obs_wave = observed_spectrum.spectral_axis.to(u.Angstrom).value
    std_wave = standard_star_spectrum.spectral_axis.to(u.Angstrom).value
    
    # Interpolate standard star to observed wavelength grid
    std_flux_interp = np.interp(obs_wave, std_wave, 
                                standard_star_spectrum.flux.value)
    
    # Compute raw sensitivity
    # Avoid division by zero
    obs_flux = observed_spectrum.flux.value
    nonzero = obs_flux > 0
    
    sensitivity_raw = np.zeros_like(obs_flux)
    sensitivity_raw[nonzero] = std_flux_interp[nonzero] / obs_flux[nonzero]
    
    # Smooth sensitivity function with spline
    # Remove extreme outliers first
    valid = (sensitivity_raw > 0) & (sensitivity_raw < np.nanpercentile(sensitivity_raw, 95) * 3)
    
    if np.sum(valid) < 10:
        warnings.warn("Too few valid points for sensitivity function fitting")
        return obs_wave, sensitivity_raw
    
    try:
        # Fit univariate spline
        spline = UnivariateSpline(
            obs_wave[valid],
            sensitivity_raw[valid],
            k=smooth_order,
            s=smooth_factor * np.sum(valid)
        )
        
        sensitivity_smooth = spline(obs_wave)
        
        # Ensure positive
        sensitivity_smooth = np.maximum(sensitivity_smooth, 1e-20)
        
        return obs_wave, sensitivity_smooth
        
    except Exception as e:
        warnings.warn(f"Spline fitting failed: {e}. Using raw sensitivity.")
        return obs_wave, sensitivity_raw


def apply_sensitivity_correction(spectrum: Spectrum1D,
                                 sensitivity_wavelengths: np.ndarray,
                                 sensitivity_values: np.ndarray) -> Spectrum1D:
    """
    Apply sensitivity function to convert to physical flux units.
    
    Per tasks.md T108 and FR-009: Applies pre-computed sensitivity function
    to convert instrumental flux to physical units.
    
    Parameters
    ----------
    spectrum : Spectrum1D
        Input spectrum in instrumental units
    sensitivity_wavelengths : np.ndarray
        Wavelength array for sensitivity function (Angstroms)
    sensitivity_values : np.ndarray
        Sensitivity function values
        
    Returns
    -------
    Spectrum1D
        Flux-calibrated spectrum in physical units
        
    Notes
    -----
    The sensitivity function is interpolated to the spectrum's wavelength grid
    and multiplied by the flux:
    
    .. math::
        F_{calibrated}(\\lambda) = F_{instrumental}(\\lambda) \\times S(\\lambda)
    
    Examples
    --------
    >>> # Load pre-computed sensitivity
    >>> data = np.loadtxt('sensitivity.dat')
    >>> sens_wave, sens_values = data[:, 0], data[:, 1]
    
    >>> # Apply to science spectrum
    >>> calibrated = apply_sensitivity_correction(
    ...     science_spectrum,
    ...     sens_wave,
    ...     sens_values
    ... )
    >>> print(calibrated.flux.unit)
    erg / (Angstrom cm2 s)
    """
    # Get spectrum wavelengths
    spec_wavelengths = spectrum.spectral_axis.to(u.Angstrom).value
    
    # Interpolate sensitivity to spectrum wavelengths
    sensitivity_interp = np.interp(
        spec_wavelengths,
        sensitivity_wavelengths,
        sensitivity_values,
        left=sensitivity_values[0],
        right=sensitivity_values[-1]
    )
    
    # Apply sensitivity correction
    calibrated_flux = spectrum.flux.value * sensitivity_interp
    
    # Assume output in erg/s/cm²/Å (standard flux units)
    calibrated_flux = calibrated_flux * u.erg / (u.s * u.cm**2 * u.Angstrom)
    
    # Propagate uncertainty
    if spectrum.uncertainty is not None:
        calibrated_uncertainty = spectrum.uncertainty.array * sensitivity_interp
        calibrated_uncertainty = calibrated_uncertainty * u.erg / (u.s * u.cm**2 * u.Angstrom)
        calibrated_uncertainty = StdDevUncertainty(calibrated_uncertainty)
    else:
        calibrated_uncertainty = None
    
    # Create calibrated spectrum
    calibrated_spectrum = Spectrum1D(
        spectral_axis=spectrum.spectral_axis,
        flux=calibrated_flux,
        uncertainty=calibrated_uncertainty
    )
    
    # Update metadata
    if hasattr(spectrum, 'meta'):
        calibrated_spectrum.meta = spectrum.meta.copy()
    else:
        calibrated_spectrum.meta = {}
    
    calibrated_spectrum.meta['flux_calibrated'] = True
    calibrated_spectrum.meta['flux_units'] = 'erg/s/cm2/Angstrom'
    calibrated_spectrum.meta['mean_sensitivity'] = np.median(sensitivity_interp)
    
    return calibrated_spectrum


def load_standard_star_spectrum(star_name: str) -> Optional[Spectrum1D]:
    """
    Load standard star spectrum from catalog.
    
    Placeholder function - would load from standard star catalogs
    (e.g., HST CALSPEC, ESO standards).
    
    Parameters
    ----------
    star_name : str
        Standard star name (e.g., 'BD+17d4708', 'Feige110')
        
    Returns
    -------
    Spectrum1D or None
        Standard star spectrum if found
        
    Notes
    -----
    This is a placeholder. In production, would load from:
    - HST CALSPEC: https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/calspec
    - ESO spectrophotometric standards
    - SDSS spectrophotometric standards
    """
    warnings.warn(
        f"Standard star {star_name} loading not yet implemented. "
        f"User must provide flux-calibrated spectrum manually."
    )
    return None
