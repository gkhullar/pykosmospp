"""
Trace detection using cross-correlation method.

Per tasks.md T039 and research.md §1: Detect spectral traces by correlating
2D spectrum with Gaussian kernel, find peaks in collapsed profile.
"""

from typing import List, Optional
import numpy as np
from scipy import ndimage, signal


def detect_traces_cross_correlation(data_2d: np.ndarray,
                                    variance_2d: np.ndarray,
                                    min_snr: float = 3.0,
                                    expected_fwhm: float = 5.0,
                                    mask: Optional[np.ndarray] = None,
                                    exclude_emission_gradient: float = 0.1,
                                    max_traces: Optional[int] = None) -> List:
    """
    Detect spectral traces using cross-correlation with Gaussian kernel.
    
    Per research.md §1: Cross-correlation identifies spatial position of
    continuum sources. Excludes emission lines from profile estimation.
    
    Algorithm:
    1. Collapse spectrum along dispersion (mask emission lines via gradient)
    2. Cross-correlate with Gaussian kernel (FWHM ~ expected PSF)
    3. Find peaks above min_snr threshold
    4. Refine trace position along dispersion axis
    
    Parameters
    ----------
    data_2d : np.ndarray
        2D spectral data (spatial x spectral)
    variance_2d : np.ndarray
        Variance map
    min_snr : float, optional
        Minimum SNR for detection (default: 3.0)
    expected_fwhm : float, optional
        Expected FWHM of spatial profile in pixels (default: 5.0)
    mask : np.ndarray, optional
        Bad pixel mask (True = bad)
    exclude_emission_gradient : float, optional
        Gradient threshold for emission line exclusion (default: 0.1)
    max_traces : int, optional
        Maximum number of traces to return (most significant)
        
    Returns
    -------
    List[Trace]
        Detected traces sorted by SNR (highest first)
    """
    from ..models import Trace
    
    ny, nx = data_2d.shape
    
    if mask is None:
        mask = np.zeros_like(data_2d, dtype=bool)
    
    # Create emission line mask via gradient threshold
    # High gradients indicate emission/absorption features
    spectral_gradient = np.abs(np.gradient(data_2d, axis=1))
    gradient_threshold = exclude_emission_gradient * np.nanmedian(spectral_gradient)
    emission_mask = spectral_gradient > gradient_threshold
    
    # Combined mask
    full_mask = mask | emission_mask
    
    # Collapse along spectral axis (median of unmasked continuum)
    data_masked = np.ma.masked_array(data_2d, mask=full_mask)
    spatial_profile = np.ma.median(data_masked, axis=1).filled(0)
    
    # Estimate noise from variance
    variance_masked = np.ma.masked_array(variance_2d, mask=full_mask)
    noise_profile = np.sqrt(np.ma.median(variance_masked, axis=1).filled(1))
    
    # Create Gaussian kernel for cross-correlation
    sigma = expected_fwhm / 2.355  # FWHM to sigma
    kernel_size = int(4 * sigma)
    if kernel_size % 2 == 0:
        kernel_size += 1  # Make odd
    kernel = signal.windows.gaussian(kernel_size, sigma)
    kernel /= kernel.sum()
    
    # Cross-correlate spatial profile with kernel
    correlated = ndimage.correlate1d(spatial_profile, kernel, mode='constant')
    
    # SNR of correlation
    snr_profile = correlated / noise_profile
    
    # Find peaks above threshold
    peak_indices, properties = signal.find_peaks(
        snr_profile,
        height=min_snr,
        distance=int(2 * expected_fwhm),  # Minimum separation between traces
        prominence=min_snr * 0.5
    )
    
    if len(peak_indices) == 0:
        return []
    
    # Sort by SNR (highest first)
    peak_snrs = snr_profile[peak_indices]
    sorted_indices = np.argsort(peak_snrs)[::-1]
    peak_indices = peak_indices[sorted_indices]
    peak_snrs = peak_snrs[sorted_indices]
    
    # Limit number of traces
    if max_traces is not None:
        peak_indices = peak_indices[:max_traces]
        peak_snrs = peak_snrs[:max_traces]
    
    # For each detected peak, trace spatial position along dispersion
    traces = []
    spectral_pixels = np.arange(nx)
    
    for trace_id, (peak_y, peak_snr) in enumerate(zip(peak_indices, peak_snrs)):
        # Trace center position along dispersion axis
        # Use centroid in window around peak for each spectral pixel
        spatial_positions = []
        
        window_half = int(2 * expected_fwhm)
        
        for x in spectral_pixels:
            # Extract spatial column
            y_start = max(0, peak_y - window_half)
            y_end = min(ny, peak_y + window_half + 1)
            
            column = data_2d[y_start:y_end, x]
            
            # Skip if masked
            if mask[y_start:y_end, x].all():
                spatial_positions.append(float(peak_y))
                continue
            
            # Compute centroid
            y_positions = np.arange(y_start, y_end)
            if np.sum(column) > 0:
                centroid = np.sum(y_positions * column) / np.sum(column)
                spatial_positions.append(centroid)
            else:
                spatial_positions.append(float(peak_y))
        
        # Create Trace object
        trace = Trace(
            trace_id=trace_id,
            spatial_positions=np.array(spatial_positions),
            spectral_pixels=spectral_pixels,
            snr_estimate=float(peak_snr),
            user_selected=False
        )
        traces.append(trace)
    
    return traces
