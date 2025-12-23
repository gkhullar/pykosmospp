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


def detect_vignetted_edges(data_2d: np.ndarray,
                           threshold: float = 0.1,
                           edge_buffer: int = 10) -> Tuple[int, int]:
    """
    Automatically detect vignetted edges where normalized response drops to ~0.
    
    KOSMOS detector shows vignetting at left and right edges where the
    flat field response drops to near zero. This function detects these
    regions automatically and returns pixel boundaries for clipping.
    
    Parameters
    ----------
    data_2d : np.ndarray
        2D frame (preferably master flat) with shape (ny_spectral, nx_spatial)
    threshold : float, optional
        Fraction of median response below which pixels are considered vignetted
        (default: 0.1, i.e., < 10% of median response)
    edge_buffer : int, optional
        Additional buffer pixels to clip beyond detected edge (default: 10)
        
    Returns
    -------
    x_min : int
        Left edge pixel to clip (start keeping data from here)
    x_max : int
        Right edge pixel to clip (stop keeping data before here)
        
    Notes
    -----
    Uses spatial profile (collapsed along Y/spectral axis) to identify
    where response drops below threshold. Adds buffer for safety.
    
    Examples
    --------
    >>> x_min, x_max = detect_vignetted_edges(master_flat.data.data)
    >>> clipped_data = data_2d[:, x_min:x_max]
    """
    ny_spectral, nx_spatial = data_2d.shape
    
    # Get spatial profile
    spatial_x, spatial_profile = create_spatial_profile_map(data_2d)
    
    # Normalize to median
    median_response = np.median(spatial_profile)
    normalized_profile = spatial_profile / median_response
    
    # Find where response > threshold
    valid_mask = normalized_profile > threshold
    
    # Find first and last valid pixels
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        logger.warning("No valid pixels found above threshold - using full detector")
        return 0, nx_spatial
    
    x_min_detected = valid_indices[0]
    x_max_detected = valid_indices[-1] + 1  # +1 for Python slicing (exclusive end)
    
    # Apply buffer (but stay within detector bounds)
    x_min = max(0, x_min_detected + edge_buffer)
    x_max = min(nx_spatial, x_max_detected - edge_buffer)
    
    # Sanity check
    if x_max <= x_min:
        logger.warning("Edge detection resulted in invalid range - using full detector")
        return 0, nx_spatial
    
    pixels_clipped_left = x_min
    pixels_clipped_right = nx_spatial - x_max
    total_clipped = pixels_clipped_left + pixels_clipped_right
    
    logger.info(f"Detected vignetted edges: clip {pixels_clipped_left} pixels from left, "
               f"{pixels_clipped_right} pixels from right ({total_clipped} total)")
    logger.info(f"Valid detector range: X = [{x_min}:{x_max}] "
               f"({x_max - x_min} pixels retained)")
    
    return x_min, x_max


def clip_detector_edges(ccd_data: CCDData,
                        x_min: int,
                        x_max: int) -> CCDData:
    """
    Clip vignetted edges from detector frame.
    
    Creates a new CCDData object with spatial (X) dimension trimmed
    to remove vignetted regions. Updates header with clipping info.
    
    Parameters
    ----------
    ccd_data : CCDData
        Input frame with shape (ny_spectral, nx_spatial)
    x_min : int
        Left edge pixel (start of valid region)
    x_max : int
        Right edge pixel (end of valid region, exclusive)
        
    Returns
    -------
    CCDData
        Clipped frame with shape (ny_spectral, x_max - x_min)
        
    Notes
    -----
    This operation should be applied consistently to all frames
    (bias, flat, arc, science) to maintain alignment.
    
    Examples
    --------
    >>> x_min, x_max = detect_vignetted_edges(master_flat.data.data)
    >>> clipped_bias = clip_detector_edges(master_bias.data, x_min, x_max)
    >>> clipped_flat = clip_detector_edges(master_flat.data, x_min, x_max)
    """
    # Clip spatial (X) dimension: data[y, x]
    clipped_data = ccd_data.data[:, x_min:x_max]
    
    # Create new CCDData with clipped data
    clipped_ccd = CCDData(
        clipped_data,
        unit=ccd_data.unit,
        header=ccd_data.header.copy()
    )
    
    # Update header with clipping info
    clipped_ccd.header['EDGECLIP'] = (True, 'Vignetted edges clipped')
    clipped_ccd.header['XCLIPMIN'] = (x_min, 'Left edge clipped at X pixel')
    clipped_ccd.header['XCLIPMAX'] = (x_max, 'Right edge clipped before X pixel')
    clipped_ccd.header['XCLIPPED'] = (x_min + (ccd_data.data.shape[1] - x_max),
                                      'Total X pixels clipped')
    
    # Update NAXIS1 if present (spatial dimension)
    if 'NAXIS1' in clipped_ccd.header:
        original_naxis1 = clipped_ccd.header['NAXIS1']
        clipped_ccd.header['NAXIS1'] = x_max - x_min
        clipped_ccd.header['ONAXIS1'] = (original_naxis1, 'Original NAXIS1 before clipping')
    
    return clipped_ccd
