"""
Flat field processing.

Creates master flat from multiple flat frames with normalization and validation.
Per tasks.md T026 and data-model.md §8.
"""

from typing import List, Tuple
import numpy as np
from astropy.nddata import CCDData

from ..models import FlatFrame, MasterFlat, MasterBias
from .combine import sigma_clipped_median_combine


def create_master_flat(flat_frames: List[FlatFrame],
                       master_bias: MasterBias,
                       sigma: float = 3.0,
                       normalization_region: Tuple[float, float, float, float] = (0.25, 0.75, 0.25, 0.75),
                       max_bad_pixel_fraction: float = 0.05,
                       expected_counts_min: float = 10000.0,
                       expected_counts_max: float = 50000.0) -> MasterFlat:
    """
    Create master flat from multiple flat frames.
    
    Per data-model.md §8: Bias-subtracts flats, combines, normalizes to median=1.0,
    identifies bad pixels, validates bad_pixel_fraction <0.05.
    
    Parameters
    ----------
    flat_frames : list of FlatFrame
        Flat frames to combine (≥3 recommended)
    master_bias : MasterBias
        Master bias for bias subtraction
    sigma : float, optional
        Sigma clipping threshold (default: 3.0)
    normalization_region : tuple, optional
        Fractional region for normalization (spatial_start, spatial_end, 
        spectral_start, spectral_end), default: (0.25, 0.75, 0.25, 0.75)
    max_bad_pixel_fraction : float, optional
        Maximum acceptable bad pixel fraction (default: 0.05)
    expected_counts_min : float, optional
        Minimum median counts for well-exposed flat (default: 10000 ADU)
    expected_counts_max : float, optional
        Maximum median counts for well-exposed flat (default: 50000 ADU)
        
    Returns
    -------
    MasterFlat
        Combined normalized master flat with bad pixel mask
        
    Raises
    ------
    ValueError
        If too few frames, bad exposure, or too many bad pixels
    """
    if len(flat_frames) < 2:
        raise ValueError(f"Need at least 2 flat frames, got {len(flat_frames)}")
    
    # Bias-subtract each flat frame
    bias_subtracted = []
    for ff in flat_frames:
        calibrated = ff.data.subtract(master_bias.data)
        bias_subtracted.append(calibrated)
        
        # Check exposure level
        median_counts = float(np.median(calibrated.data))
        if median_counts < expected_counts_min or median_counts > expected_counts_max:
            raise ValueError(
                f"Flat {ff.file_path} poorly exposed: {median_counts:.0f} ADU "
                f"(expected {expected_counts_min:.0f}-{expected_counts_max:.0f})"
            )
    
    # Combine using sigma-clipped median
    combined = sigma_clipped_median_combine(bias_subtracted, sigma=sigma)
    
    # Normalize to unity in specified region
    ny, nx = combined.data.shape
    y_start = int(ny * normalization_region[0])
    y_end = int(ny * normalization_region[1])
    x_start = int(nx * normalization_region[2])
    x_end = int(nx * normalization_region[3])
    
    normalization_region_data = combined.data[y_start:y_end, x_start:x_end]
    normalization_value = float(np.median(normalization_region_data))
    
    if normalization_value <= 0:
        raise ValueError("Flat field normalization region has zero or negative median")
    
    normalized_data = combined.data / normalization_value
    normalized = CCDData(normalized_data, unit=combined.unit, header=combined.header.copy())
    
    # Identify bad pixels (very low or zero flat response)
    bad_pixel_threshold = 0.2  # pixels below 20% of normalized value
    bad_pixel_mask = normalized_data < bad_pixel_threshold
    bad_pixel_fraction = float(np.sum(bad_pixel_mask) / bad_pixel_mask.size)
    
    # Validate bad pixel fraction
    if bad_pixel_fraction > max_bad_pixel_fraction:
        raise ValueError(
            f"Bad pixel fraction too high: {bad_pixel_fraction:.3f} "
            f"(limit: {max_bad_pixel_fraction})"
        )
    
    # Add normalization metadata
    normalized.header['FLATNORM'] = (normalization_value, 'Flat normalization value')
    normalized.header['NORMREG'] = (str(normalization_region), 'Normalization region')
    normalized.header['BADPXFRC'] = (bad_pixel_fraction, 'Bad pixel fraction')
    
    # Build provenance
    provenance = {
        'n_frames': len(flat_frames),
        'input_files': [str(ff.file_path) for ff in flat_frames],
        'sigma_clip': sigma,
        'normalization_value': normalization_value,
        'normalization_region': normalization_region,
        'combine_method': 'sigma_clipped_median',
    }
    
    # Create MasterFlat object
    master_flat = MasterFlat(
        data=normalized,
        n_combined=len(flat_frames),
        normalization_region=normalization_region,
        bad_pixel_fraction=bad_pixel_fraction,
        provenance=provenance
    )
    
    # Validate master flat
    master_flat.validate()
    
    return master_flat
