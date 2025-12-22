"""
Bias frame processing.

Creates master bias from multiple bias frames with validation.
Per tasks.md T025 and data-model.md §7.
"""

from typing import List
from pathlib import Path
import numpy as np
from astropy.nddata import CCDData

from ..models import BiasFrame, MasterBias
from .combine import sigma_clipped_median_combine


def create_master_bias(bias_frames: List[BiasFrame], 
                       sigma: float = 3.0,
                       max_level_variation: float = 10.0) -> MasterBias:
    """
    Create master bias from multiple bias frames.
    
    Per data-model.md §7: Median-combines bias frames, validates bias level
    variation <10 ADU, computes bias statistics.
    
    Parameters
    ----------
    bias_frames : list of BiasFrame
        Bias frames to combine (≥5 recommended per config)
    sigma : float, optional
        Sigma clipping threshold (default: 3.0)
    max_level_variation : float, optional
        Maximum acceptable bias level variation in ADU (default: 10.0)
        
    Returns
    -------
    MasterBias
        Combined master bias with statistics
        
    Raises
    ------
    ValueError
        If too few frames or bias level variation too high
    """
    if len(bias_frames) < 3:
        raise ValueError(f"Need at least 3 bias frames, got {len(bias_frames)}")
    
    # Extract CCDData from BiasFrame objects
    ccd_frames = [bf.data for bf in bias_frames]
    
    # Combine using sigma-clipped median
    combined = sigma_clipped_median_combine(ccd_frames, sigma=sigma)
    
    # Compute bias statistics
    bias_level = float(np.median(combined.data))
    bias_stdev = float(np.std(combined.data))
    
    # Validate bias level consistency across input frames
    frame_medians = [float(np.median(bf.data.data)) for bf in bias_frames]
    level_variation = max(frame_medians) - min(frame_medians)
    
    if level_variation > max_level_variation:
        raise ValueError(
            f"Bias level variation too high: {level_variation:.2f} ADU "
            f"(limit: {max_level_variation} ADU). Check bias frame consistency."
        )
    
    # Build provenance
    provenance = {
        'n_frames': len(bias_frames),
        'input_files': [str(bf.file_path) for bf in bias_frames],
        'sigma_clip': sigma,
        'level_variation': level_variation,
        'combine_method': 'sigma_clipped_median',
    }
    
    # Create MasterBias object
    master_bias = MasterBias(
        data=combined,
        n_combined=len(bias_frames),
        bias_level=bias_level,
        bias_stdev=bias_stdev,
        provenance=provenance
    )
    
    # Validate master bias
    master_bias.validate()
    
    return master_bias
