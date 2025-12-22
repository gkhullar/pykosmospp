"""
Optimal extraction following Horne (1986).

Per tasks.md T043: Implements variance-weighted optimal extraction using
spatial profile for maximum SNR.
"""

import numpy as np
from specutils import Spectrum1D
import astropy.units as u
from astropy.nddata import StdDevUncertainty


def extract_optimal(data_2d: np.ndarray,
                   variance_2d: np.ndarray,
                   trace,  # Type: Trace
                   aperture_width: int = 10) -> Spectrum1D:
    """
    Optimal extraction of 1D spectrum from 2D data.
    
    Implements Horne (1986) optimal extraction algorithm:
    - Uses spatial profile as extraction weights
    - Variance-weighted combination
    - Propagates uncertainties correctly
    - Maximizes SNR compared to simple aperture extraction
    
    Parameters
    ----------
    data_2d : np.ndarray
        2D spectral data (spatial x spectral)
    variance_2d : np.ndarray
        2D variance map
    trace : Trace
        Trace object with spatial profile
    aperture_width : int, optional
        Extraction aperture width (default: 10)
        
    Returns
    -------
    Spectrum1D
        Extracted 1D spectrum with wavelength axis if available
        
    References
    ----------
    Horne, K. 1986, PASP, 98, 609
    \"An Optimal Extraction Algorithm for CCD Spectroscopy\"
    """
    ny, nx = data_2d.shape
    
    # Check if trace has fitted spatial profile
    if trace.spatial_profile is None:
        # Fall back to aperture extraction
        return _extract_aperture(data_2d, variance_2d, trace, aperture_width)
    
    # Extract flux and variance for each spectral pixel
    flux = np.zeros(nx)
    variance = np.zeros(nx)
    
    aperture_half = aperture_width // 2
    
    for x_idx in range(nx):
        # Get trace center at this spectral pixel
        y_center = trace.spatial_positions[x_idx]
        
        # Define aperture
        y_start = int(max(0, y_center - aperture_half))
        y_end = int(min(ny, y_center + aperture_half + 1))
        
        # Extract column
        data_column = data_2d[y_start:y_end, x_idx]
        var_column = variance_2d[y_start:y_end, x_idx]
        
        # Spatial positions relative to trace center
        y_positions = np.arange(y_start, y_end) - y_center
        
        # Evaluate spatial profile at these positions
        profile_weights = trace.spatial_profile.evaluate(y_positions)
        
        # Normalize profile (sum to 1)
        if profile_weights.sum() > 0:
            profile_weights /= profile_weights.sum()
        else:
            # Fallback to uniform weights
            profile_weights = np.ones_like(profile_weights) / len(profile_weights)
        
        # Optimal extraction weights (profile^2 / variance)
        # Per Horne 1986 equation
        optimal_weights = profile_weights**2 / (var_column + 1e-10)
        
        # Normalize weights
        weight_sum = optimal_weights.sum()
        if weight_sum > 0:
            optimal_weights /= weight_sum
        
        # Extract flux (weighted sum)
        flux[x_idx] = np.sum(data_column * optimal_weights)
        
        # Propagate variance
        # Variance of weighted sum: sum(w^2 * var)
        variance[x_idx] = np.sum(optimal_weights**2 * var_column)
    
    # Create Spectrum1D
    if trace.wavelength_solution is not None:
        # Apply wavelength calibration
        from ..wavelength.apply import apply_wavelength_to_spectrum
        spectrum = apply_wavelength_to_spectrum(
            flux,
            np.sqrt(variance),
            trace.wavelength_solution
        )
    else:
        # Pixel-based spectrum
        spectral_axis = np.arange(nx) * u.pixel
        spectrum = Spectrum1D(
            spectral_axis=spectral_axis,
            flux=flux * u.electron,
            uncertainty=StdDevUncertainty(np.sqrt(variance) * u.electron)
        )
    
    # Add metadata
    spectrum.meta['extraction_method'] = 'optimal'
    spectrum.meta['trace_id'] = trace.trace_id
    spectrum.meta['aperture_width'] = aperture_width
    if trace.spatial_profile is not None:
        spectrum.meta['profile_type'] = trace.spatial_profile.profile_type
        spectrum.meta['profile_chi_squared'] = trace.spatial_profile.chi_squared
    
    return spectrum


def _extract_aperture(data_2d: np.ndarray,
                     variance_2d: np.ndarray,
                     trace,  # Type: Trace
                     aperture_width: int) -> Spectrum1D:
    """
    Simple aperture extraction (fallback when no profile available).
    
    Sums flux in fixed-width aperture centered on trace.
    """
    ny, nx = data_2d.shape
    
    flux = np.zeros(nx)
    variance = np.zeros(nx)
    
    aperture_half = aperture_width // 2
    
    for x_idx in range(nx):
        y_center = int(trace.spatial_positions[x_idx])
        
        y_start = max(0, y_center - aperture_half)
        y_end = min(ny, y_center + aperture_half + 1)
        
        # Simple sum
        flux[x_idx] = np.sum(data_2d[y_start:y_end, x_idx])
        variance[x_idx] = np.sum(variance_2d[y_start:y_end, x_idx])
    
    # Create spectrum
    if trace.wavelength_solution is not None:
        from ..wavelength.apply import apply_wavelength_to_spectrum
        spectrum = apply_wavelength_to_spectrum(
            flux,
            np.sqrt(variance),
            trace.wavelength_solution
        )
    else:
        spectral_axis = np.arange(nx) * u.pixel
        spectrum = Spectrum1D(
            spectral_axis=spectral_axis,
            flux=flux * u.electron,
            uncertainty=StdDevUncertainty(np.sqrt(variance) * u.electron)
        )
    
    spectrum.meta['extraction_method'] = 'aperture'
    spectrum.meta['trace_id'] = trace.trace_id
    spectrum.meta['aperture_width'] = aperture_width
    
    return spectrum
