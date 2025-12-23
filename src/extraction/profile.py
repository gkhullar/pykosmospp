"""
Spatial profile fitting for optimal extraction.

Per tasks.md T041 and research.md §4: Fit Gaussian or Moffat profile to
spatial cross-section, with fallback to empirical profile if fit fails.
"""

from typing import Optional
import numpy as np
from scipy import optimize
import warnings


def fit_spatial_profile(data_2d: np.ndarray,
                       variance_2d: np.ndarray,
                       trace,  # Type: Trace
                       aperture_width: int = 10,
                       profile_type: str = 'gaussian',
                       chi_sq_threshold: float = 10.0,
                       continuum_fraction: float = 0.5):
    """
    Fit spatial profile to trace for optimal extraction.
    
    Per research.md §4: Fits Gaussian profile on continuum regions,
    masks emission lines via gradient, falls back to empirical if
    chi-squared >10.
    
    Parameters
    ----------
    data_2d : np.ndarray
        2D spectral data
    variance_2d : np.ndarray
        Variance map
    trace : Trace
        Trace object with spatial positions
    aperture_width : int, optional
        Extraction aperture width (default: 10)
    profile_type : str, optional
        Profile type to fit ('gaussian' or 'moffat', default: 'gaussian')
    chi_sq_threshold : float, optional
        Chi-squared threshold for fallback to empirical (default: 10.0)
    continuum_fraction : float, optional
        Fraction of spectral pixels to use (lowest gradient = continuum)
        
    Returns
    -------
    SpatialProfile
        Fitted spatial profile
    """
    from ..models import SpatialProfile
    
    # Shape: data_2d[y, x] where x=spatial, y=spectral/wavelength
    ny_spectral, nx_spatial = data_2d.shape
    
    # Select continuum regions (low spectral gradient along Y axis)
    spectral_gradient = np.abs(np.gradient(data_2d, axis=0))  # Gradient along Y (spectral)
    gradient_median = np.median(spectral_gradient, axis=1)  # Median across spatial (X)
    
    # Use spectral pixels with lowest gradient (continuum)
    n_continuum = int(continuum_fraction * ny_spectral)
    continuum_indices = np.argsort(gradient_median)[:n_continuum]
    
    # Extract spatial profiles at continuum wavelengths
    aperture_half = aperture_width // 2
    profiles = []
    weights = []
    
    for y_idx in continuum_indices:
        # Get trace center (X position) at this spectral/Y pixel
        x_center = int(trace.spatial_positions[y_idx])
        
        # Extract aperture in spatial/X direction
        x_start = max(0, x_center - aperture_half)
        x_end = min(nx_spatial, x_center + aperture_half + 1)
        
        profile = data_2d[y_idx, x_start:x_end]  # Extract along X at fixed Y
        variance = variance_2d[y_idx, x_start:x_end]
        
        # Weight by inverse variance
        weight = 1.0 / np.sqrt(variance + 1e-10)
        
        profiles.append(profile)
        weights.append(weight)
    
    # Stack and collapse profiles
    max_len = max(len(p) for p in profiles)
    stacked_profile = np.zeros(max_len)
    stacked_weights = np.zeros(max_len)
    
    for profile, weight in zip(profiles, weights):
        pad_start = (max_len - len(profile)) // 2
        stacked_profile[pad_start:pad_start+len(profile)] += profile * weight
        stacked_weights[pad_start:pad_start+len(profile)] += weight
    
    # Normalize
    valid = stacked_weights > 0
    stacked_profile[valid] /= stacked_weights[valid]
    
    # Spatial pixel positions relative to center
    spatial_pixels = np.arange(max_len) - max_len // 2
    
    # Fit profile
    if profile_type == 'gaussian':
        profile_func, params, chi_sq = _fit_gaussian_profile(
            spatial_pixels,
            stacked_profile,
            stacked_weights
        )
    elif profile_type == 'moffat':
        profile_func, params, chi_sq = _fit_moffat_profile(
            spatial_pixels,
            stacked_profile,
            stacked_weights
        )
    else:
        raise ValueError(f"Unknown profile type: {profile_type}")
    
    # Check fit quality, fallback to empirical if poor
    if chi_sq > chi_sq_threshold:
        warnings.warn(
            f"Profile fit chi-squared {chi_sq:.2f} exceeds threshold {chi_sq_threshold}. "
            f"Using empirical profile."
        )
        profile_func = lambda x: np.interp(x, spatial_pixels, stacked_profile,
                                          left=0, right=0)
        profile_type = 'empirical'
        params = {'center': 0, 'width': aperture_width / 2.355, 'amplitude': stacked_profile.max()}
    
    # Create SpatialProfile object
    spatial_profile = SpatialProfile(
        profile_type=profile_type,
        center=params['center'],
        width=params['width'],
        amplitude=params['amplitude'],
        profile_function=profile_func,
        chi_squared=chi_sq
    )
    
    return spatial_profile


def _fit_gaussian_profile(x, y, weights):
    """Fit Gaussian profile to data."""
    # Initial guess
    amplitude = y.max()
    center = x[np.argmax(y)]
    sigma = 2.0  # ~5 pixel FWHM
    
    # Gaussian function
    def gaussian(x, amp, cen, sig):
        return amp * np.exp(-0.5 * ((x - cen) / sig)**2)
    
    try:
        # Fit with scipy
        popt, _ = optimize.curve_fit(
            gaussian,
            x,
            y,
            p0=[amplitude, center, sigma],
            sigma=1.0/weights,
            absolute_sigma=True,
            maxfev=1000
        )
        
        amp, cen, sig = popt
        
        # Compute chi-squared
        model = gaussian(x, amp, cen, sig)
        residuals = (y - model) * weights
        chi_sq = np.sum(residuals**2) / max(1, len(x) - 3)
        
        # Create profile function
        profile_func = lambda pos: gaussian(pos, amp, cen, sig)
        
        params = {
            'center': cen,
            'width': sig * 2.355,  # Convert sigma to FWHM
            'amplitude': amp
        }
        
        return profile_func, params, chi_sq
        
    except (RuntimeError, ValueError):
        # Fit failed, return empirical
        warnings.warn("Gaussian fit failed, using empirical profile")
        profile_func = lambda pos: np.interp(pos, x, y, left=0, right=0)
        params = {
            'center': center,
            'width': sigma * 2.355,
            'amplitude': amplitude
        }
        return profile_func, params, 999.0


def _fit_moffat_profile(x, y, weights):
    """Fit Moffat profile to data."""
    # Initial guess
    amplitude = y.max()
    center = x[np.argmax(y)]
    alpha = 2.0
    beta = 2.5
    
    # Moffat function
    def moffat(x, amp, cen, alp, bet):
        return amp / (1 + ((x - cen) / alp)**2)**bet
    
    try:
        popt, _ = optimize.curve_fit(
            moffat,
            x,
            y,
            p0=[amplitude, center, alpha, beta],
            sigma=1.0/weights,
            absolute_sigma=True,
            maxfev=1000
        )
        
        amp, cen, alp, bet = popt
        
        # Compute chi-squared
        model = moffat(x, amp, cen, alp, bet)
        residuals = (y - model) * weights
        chi_sq = np.sum(residuals**2) / max(1, len(x) - 4)
        
        profile_func = lambda pos: moffat(pos, amp, cen, alp, bet)
        
        # Convert to FWHM
        fwhm = 2 * alp * np.sqrt(2**(1/bet) - 1)
        
        params = {
            'center': cen,
            'width': fwhm,
            'amplitude': amp
        }
        
        return profile_func, params, chi_sq
        
    except (RuntimeError, ValueError):
        warnings.warn("Moffat fit failed, using empirical profile")
        profile_func = lambda pos: np.interp(pos, x, y, left=0, right=0)
        params = {
            'center': center,
            'width': alpha * 2,
            'amplitude': amplitude
        }
        return profile_func, params, 999.0
