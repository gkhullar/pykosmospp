"""
Spectrum extraction methods: optimal (Horne 1986) and boxcar.

Per tasks.md T043, T109: Implements variance-weighted optimal extraction using
spatial profile for maximum SNR, with boxcar extraction as fallback/alternative.
Per tasks.md T105: Spatial binning support for low-SNR data.
"""

import numpy as np
from specutils import Spectrum1D
import astropy.units as u
from astropy.nddata import StdDevUncertainty
from typing import Literal, Optional


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


def extract_boxcar(data_2d: np.ndarray,
                  variance_2d: np.ndarray,
                  trace,  # Type: Trace
                  aperture_width: int = 10) -> Spectrum1D:
    """
    Boxcar extraction: simple aperture summation.
    
    Per tasks.md T109: Alternative to optimal extraction using simple
    summation in fixed aperture. Faster and more robust for bright sources
    or when spatial profile is poorly constrained.
    
    Parameters
    ----------
    data_2d : np.ndarray
        2D spectral data (spatial x spectral)
    variance_2d : np.ndarray
        2D variance map
    trace : Trace
        Trace object with spatial positions
    aperture_width : int, optional
        Extraction aperture width (default: 10)
        
    Returns
    -------
    Spectrum1D
        Extracted 1D spectrum
        
    Notes
    -----
    Boxcar extraction is recommended for:
    - Bright sources (SNR > 50)
    - Extended sources with complex spatial structure
    - Quick-look reductions
    - Cases where optimal extraction profile fit fails
    
    Optimal extraction typically provides 10-30% better SNR for faint sources.
    
    Examples
    --------
    >>> spectrum = extract_boxcar(data_2d, variance_2d, trace, aperture_width=12)
    >>> print(spectrum.meta['extraction_method'])
    'boxcar'
    """
    ny, nx = data_2d.shape
    
    flux = np.zeros(nx)
    variance = np.zeros(nx)
    
    aperture_half = aperture_width // 2
    
    for x_idx in range(nx):
        y_center = int(trace.spatial_positions[x_idx])
        
        y_start = max(0, y_center - aperture_half)
        y_end = min(ny, y_center + aperture_half + 1)
        
        # Simple sum across aperture
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
    
    # Add extraction metadata
    spectrum.meta['extraction_method'] = 'boxcar'
    spectrum.meta['trace_id'] = trace.trace_id
    spectrum.meta['aperture_width'] = aperture_width
    
    return spectrum


def extract_spectrum(data_2d: np.ndarray,
                    variance_2d: np.ndarray,
                    trace,  # Type: Trace
                    method: Literal['optimal', 'boxcar', 'auto'] = 'auto',
                    aperture_width: int = 10) -> Spectrum1D:
    """
    Extract 1D spectrum using specified method.
    
    Per tasks.md T109: Unified interface for extraction method selection.
    
    Parameters
    ----------
    data_2d : np.ndarray
        2D spectral data
    variance_2d : np.ndarray
        2D variance map
    trace : Trace
        Trace object
    method : {'optimal', 'boxcar', 'auto'}, optional
        Extraction method (default: 'auto')
        - 'optimal': Horne 1986 variance-weighted extraction
        - 'boxcar': Simple aperture summation
        - 'auto': Use optimal if profile available, else boxcar
    aperture_width : int, optional
        Extraction aperture width (default: 10)
        
    Returns
    -------
    Spectrum1D
        Extracted 1D spectrum
        
    Examples
    --------
    >>> # Force boxcar for bright source
    >>> spectrum = extract_spectrum(data, variance, trace, method='boxcar')
    
    >>> # Auto-select based on profile availability
    >>> spectrum = extract_spectrum(data, variance, trace, method='auto')
    """
    if method == 'optimal':
        return extract_optimal(data_2d, variance_2d, trace, aperture_width)
    elif method == 'boxcar':
        return extract_boxcar(data_2d, variance_2d, trace, aperture_width)
    elif method == 'auto':
        # Use optimal if spatial profile is available and good quality
        if (hasattr(trace, 'spatial_profile') and 
            trace.spatial_profile is not None and
            trace.spatial_profile.chi_squared < 10.0):
            return extract_optimal(data_2d, variance_2d, trace, aperture_width)
        else:
            return extract_boxcar(data_2d, variance_2d, trace, aperture_width)
    else:
        raise ValueError(f"Unknown extraction method: {method}. "
                        f"Must be 'optimal', 'boxcar', or 'auto'.")


def bin_spatial(data_2d: np.ndarray,
               variance_2d: Optional[np.ndarray] = None,
               bin_factor: int = 2) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Bin 2D spectrum spatially to improve SNR for faint sources.
    
    Per tasks.md T105 and research.md §6: Pre-extraction spatial binning
    for low-SNR data. Bins spatial direction by summing adjacent pixels.
    
    Parameters
    ----------
    data_2d : np.ndarray
        2D spectral data (spatial x spectral)
    variance_2d : np.ndarray, optional
        2D variance map. If provided, properly propagates uncertainties.
    bin_factor : int, optional
        Spatial binning factor (default: 2)
        bin_factor=2 combines 2 pixels → 1 (reduces spatial resolution by 2×)
        
    Returns
    -------
    binned_data : np.ndarray
        Spatially binned 2D spectrum
    binned_variance : np.ndarray or None
        Spatially binned variance (if variance_2d provided)
        
    Notes
    -----
    Spatial binning reduces spatial resolution but improves SNR by √bin_factor.
    Recommended for:
    - Faint sources (SNR < 5)
    - When spatial resolution is not critical
    - Point sources where PSF spans multiple pixels
    
    Examples
    --------
    >>> binned_data, binned_var = bin_spatial(data_2d, variance_2d, bin_factor=2)
    >>> print(f"Original shape: {data_2d.shape}, Binned shape: {binned_data.shape}")
    Original shape: (515, 2048), Binned shape: (257, 2048)
    """
    ny, nx = data_2d.shape
    
    # Calculate new spatial size
    ny_binned = ny // bin_factor
    
    # Reshape and sum spatial pixels
    # Trim to multiple of bin_factor
    ny_trim = ny_binned * bin_factor
    data_trimmed = data_2d[:ny_trim, :]
    
    # Reshape to (ny_binned, bin_factor, nx) and sum along bin_factor axis
    binned_data = data_trimmed.reshape(ny_binned, bin_factor, nx).sum(axis=1)
    
    # Propagate variance if provided
    if variance_2d is not None:
        variance_trimmed = variance_2d[:ny_trim, :]
        # Variance sums when adding (not averaging)
        binned_variance = variance_trimmed.reshape(ny_binned, bin_factor, nx).sum(axis=1)
        return binned_data, binned_variance
    else:
        return binned_data, None


