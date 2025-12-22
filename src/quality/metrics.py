"""
Quality metrics computation for reduced spectra.

Per data-model.md §14, computes SNR, wavelength RMS, sky residuals,
and assigns overall quality grade.
"""

from typing import Dict, Optional
import numpy as np
from scipy.ndimage import median_filter


def compute_quality_metrics(
    spectrum_1d,
    spectrum_2d: Optional['Spectrum2D'] = None
) -> Dict:
    """
    Compute quality metrics for extracted spectrum.
    
    Parameters
    ----------
    spectrum_1d : Spectrum1D
        Extracted 1D spectrum
    spectrum_2d : Spectrum2D, optional
        Source 2D spectrum for cosmic ray fraction
    
    Returns
    -------
    Dict
        Quality metrics dictionary
    """
    metrics = {}
    
    # Compute median SNR in continuum regions
    # Use gradient to identify emission lines (avoid them)
    flux = spectrum_1d.flux.value
    if hasattr(spectrum_1d, 'uncertainty') and spectrum_1d.uncertainty is not None:
        uncertainty = spectrum_1d.uncertainty.array
    else:
        # Estimate from flux
        uncertainty = np.sqrt(np.abs(flux))
        uncertainty[uncertainty == 0] = 1.0
    
    snr = flux / uncertainty
    
    # Identify continuum (low gradient regions)
    gradient = np.abs(np.gradient(flux))
    gradient_threshold = np.percentile(gradient, 50)
    continuum_mask = gradient < gradient_threshold
    
    if np.sum(continuum_mask) > 0:
        median_snr = np.median(snr[continuum_mask])
    else:
        median_snr = np.median(snr)
    
    metrics['median_snr'] = float(median_snr)
    
    # Wavelength RMS from metadata
    if hasattr(spectrum_1d, 'meta') and 'wavelength_rms' in spectrum_1d.meta:
        metrics['wavelength_rms'] = spectrum_1d.meta['wavelength_rms']
    
    # Sky residual RMS from metadata
    if hasattr(spectrum_1d, 'meta') and 'sky_residual_rms' in spectrum_1d.meta:
        metrics['sky_residual_rms'] = spectrum_1d.meta['sky_residual_rms']
    
    # Cosmic ray fraction from 2D spectrum
    if spectrum_2d is not None and spectrum_2d.cosmic_ray_mask is not None:
        cosmic_ray_fraction = np.sum(spectrum_2d.cosmic_ray_mask) / spectrum_2d.cosmic_ray_mask.size
        metrics['cosmic_ray_fraction'] = float(cosmic_ray_fraction)
    
    # Spatial profile consistency score from chi-squared
    # Per remediation fix C1: FR-017 requires spatial trace profile consistency
    if hasattr(spectrum_1d, 'meta') and 'spatial_profile_chi_squared' in spectrum_1d.meta:
        chi_squared = spectrum_1d.meta['spatial_profile_chi_squared']
        # Convert chi-squared to consistency score (0-1, higher is better)
        # Good fit: chi_squared ~ 1, score ~ 1
        # Poor fit: chi_squared >> 1, score → 0
        profile_consistency_score = np.exp(-abs(chi_squared - 1.0))
        metrics['profile_consistency_score'] = float(profile_consistency_score)
    
    # Saturation flag
    saturation_threshold = 65535
    if np.any(flux >= saturation_threshold * 0.95):
        metrics['saturation_flag'] = True
    else:
        metrics['saturation_flag'] = False
    
    # Assign overall grade
    grade = _assign_grade(metrics)
    metrics['overall_grade'] = grade
    
    return metrics


def _assign_grade(metrics: Dict) -> str:
    """
    Assign overall quality grade based on metrics.
    
    Grading criteria:
    - Excellent: SNR > 20, wavelength RMS < 0.1 Å, no saturation
    - Good: SNR > 10, wavelength RMS < 0.2 Å
    - Fair: SNR > 5, wavelength RMS < 0.3 Å
    - Poor: Otherwise
    
    Parameters
    ----------
    metrics : Dict
        Quality metrics
    
    Returns
    -------
    str
        Grade ('Excellent', 'Good', 'Fair', 'Poor')
    """
    snr = metrics.get('median_snr', 0)
    wavelength_rms = metrics.get('wavelength_rms', 1.0)
    saturated = metrics.get('saturation_flag', False)
    
    if snr > 20 and wavelength_rms < 0.1 and not saturated:
        return 'Excellent'
    elif snr > 10 and wavelength_rms < 0.2:
        return 'Good'
    elif snr > 5 and wavelength_rms < 0.3:
        return 'Fair'
    else:
        return 'Poor'


def compute_ab_subtraction_quality(
    spectrum_a,
    spectrum_b,
    spectrum_ab
) -> float:
    """
    Compute quality metric for A-B subtraction.
    
    Parameters
    ----------
    spectrum_a, spectrum_b, spectrum_ab : Spectrum1D
        A, B, and A-B spectra
    
    Returns
    -------
    float
        Quality score (0-1, higher is better)
    """
    # Compute residuals in continuum regions
    flux_a = spectrum_a.flux.value
    flux_b = spectrum_b.flux.value
    flux_ab = spectrum_ab.flux.value
    
    expected_ab = flux_a - flux_b
    residuals = flux_ab - expected_ab
    
    # Normalize by typical flux level
    typical_flux = np.median(np.abs(flux_a))
    if typical_flux > 0:
        normalized_rms = np.std(residuals) / typical_flux
    else:
        normalized_rms = np.inf
    
    # Convert to quality score (0-1)
    quality = np.exp(-normalized_rms)
    
    return float(quality)
