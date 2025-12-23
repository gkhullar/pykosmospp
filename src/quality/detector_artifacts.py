"""
Detector artifact detection and diagnostics.

Per user feedback: KOSMOS detector shows different background levels
in left vs right halves. Check if flat fielding removes this artifact.
"""

import numpy as np
from astropy.nddata import CCDData
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def check_left_right_bias(data_2d: np.ndarray,
                          threshold: float = 5.0) -> Dict[str, float]:
    """
    Check for left/right background level difference in detector.
    
    KOSMOS detector shows different background levels in left (X < nx/2)
    vs right (X >= nx/2) halves. This should be corrected by flat fielding,
    but if not, additional correction may be needed.
    
    Parameters
    ----------
    data_2d : np.ndarray
        2D frame with shape (ny_spectral, nx_spatial)
    threshold : float, optional
        Sigma threshold for significant difference (default: 5.0)
        
    Returns
    -------
    dict
        Statistics including left_median, right_median, difference, 
        difference_sigma, is_significant
    """
    ny_spectral, nx_spatial = data_2d.shape
    
    # Split into left and right halves
    mid_x = nx_spatial // 2
    left_half = data_2d[:, :mid_x]
    right_half = data_2d[:, mid_x:]
    
    # Compute robust statistics for each half
    left_median = np.median(left_half)
    right_median = np.median(right_half)
    
    # Estimate noise in each half using MAD
    left_mad = 1.4826 * np.median(np.abs(left_half - left_median))
    right_mad = 1.4826 * np.median(np.abs(right_half - right_median))
    
    # Combined noise estimate
    combined_noise = np.sqrt(left_mad**2 + right_mad**2) / np.sqrt(left_half.size + right_half.size)
    
    # Difference and significance
    difference = right_median - left_median
    difference_sigma = difference / combined_noise if combined_noise > 0 else 0
    
    is_significant = np.abs(difference_sigma) > threshold
    
    result = {
        'left_median': float(left_median),
        'right_median': float(right_median),
        'left_mad': float(left_mad),
        'right_mad': float(right_mad),
        'difference': float(difference),
        'difference_sigma': float(difference_sigma),
        'is_significant': bool(is_significant),
        'threshold_sigma': threshold
    }
    
    if is_significant:
        logger.warning(
            f"Significant left/right background difference detected: "
            f"{difference:.2f} ADU ({difference_sigma:.1f}σ)"
        )
    
    return result


def diagnose_detector_artifacts(science_frame_raw: CCDData,
                                science_frame_calibrated: CCDData,
                                master_flat: CCDData) -> Dict[str, Dict]:
    """
    Comprehensive check for detector artifacts before and after calibration.
    
    Checks if flat fielding removes left/right background bias and other
    systematic patterns.
    
    Parameters
    ----------
    science_frame_raw : CCDData
        Raw science frame (before calibration)
    science_frame_calibrated : CCDData
        Calibrated science frame (after bias subtraction and flat fielding)
    master_flat : CCDData
        Master flat field used for calibration
        
    Returns
    -------
    dict
        Dictionary with keys 'raw', 'calibrated', 'flat', each containing
        left/right bias check results
    """
    results = {}
    
    # Check raw frame
    logger.info("Checking raw science frame for left/right bias...")
    results['raw'] = check_left_right_bias(science_frame_raw.data)
    
    # Check calibrated frame
    logger.info("Checking calibrated science frame for left/right bias...")
    results['calibrated'] = check_left_right_bias(science_frame_calibrated.data)
    
    # Check flat field itself
    logger.info("Checking master flat for left/right pattern...")
    results['flat'] = check_left_right_bias(master_flat.data)
    
    # Summary
    logger.info("\n=== Detector Artifact Diagnosis ===")
    logger.info(f"Raw frame left/right bias: {results['raw']['difference']:.2f} ADU "
               f"({results['raw']['difference_sigma']:.1f}σ)")
    logger.info(f"Flat field left/right pattern: {results['flat']['difference']:.4f} "
               f"({results['flat']['difference_sigma']:.1f}σ)")
    logger.info(f"Calibrated frame left/right bias: {results['calibrated']['difference']:.2f} ADU "
               f"({results['calibrated']['difference_sigma']:.1f}σ)")
    
    if results['calibrated']['is_significant']:
        logger.warning(
            "⚠️  Left/right bias persists after flat fielding! "
            "Additional correction may be needed."
        )
    else:
        logger.info("✓ Flat fielding successfully removes left/right bias.")
    
    return results


def create_spatial_profile_map(data_2d: np.ndarray,
                               n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create spatial profile map to visualize detector artifacts.
    
    Collapses spectrum along Y (wavelength) axis to show spatial structure,
    useful for identifying left/right bias, vignetting, or other patterns.
    
    Parameters
    ----------
    data_2d : np.ndarray
        2D frame with shape (ny_spectral, nx_spatial)
    n_bins : int, optional
        Number of bins along spectral direction for averaging
        
    Returns
    -------
    spatial_positions : np.ndarray
        X pixel positions (spatial direction)
    spatial_profile : np.ndarray
        Median counts at each spatial position
    """
    ny_spectral, nx_spatial = data_2d.shape
    
    # Collapse along spectral/Y axis
    spatial_profile = np.median(data_2d, axis=0)
    spatial_positions = np.arange(nx_spatial)
    
    return spatial_positions, spatial_profile
