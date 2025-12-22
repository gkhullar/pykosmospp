"""
Frame combination utilities using sigma-clipped statistics.

Per research.md §7: Median combine with sigma-clipping for bias and flat frames.
"""

from typing import List
import numpy as np
from astropy.nddata import CCDData
from astropy.stats import sigma_clipped_stats
import astropy.units as u

from ..models import MasterBias, MasterFlat, BiasFrame, FlatFrame


def sigma_clipped_median_combine(frames: List[CCDData], 
                                 sigma: float = 3.0,
                                 maxiters: int = 5) -> CCDData:
    """
    Combine frames using sigma-clipped median.
    
    Per research.md §7: Median combining is more robust to outliers than mean
    for bias and flat frames. Sigma-clipping removes cosmic rays and bad pixels.
    
    Parameters
    ----------
    frames : list of CCDData
        Frames to combine (must all have same shape)
    sigma : float, optional
        Sigma threshold for clipping (default: 3.0)
    maxiters : int, optional
        Maximum clipping iterations (default: 5)
        
    Returns
    -------
    CCDData
        Combined frame with median value per pixel
        
    Raises
    ------
    ValueError
        If frames have inconsistent shapes or units
    """
    if not frames:
        raise ValueError("Cannot combine empty frame list")
    
    # Validate all frames have same shape
    shape = frames[0].data.shape
    for i, frame in enumerate(frames[1:], start=1):
        if frame.data.shape != shape:
            raise ValueError(f"Frame {i} has shape {frame.data.shape}, expected {shape}")
    
    # Stack data arrays
    data_stack = np.array([frame.data for frame in frames])
    
    # Sigma-clipped median along frame axis
    # astropy.stats.sigma_clipped_stats returns (mean, median, stddev)
    _, median, _ = sigma_clipped_stats(data_stack, axis=0, sigma=sigma, maxiters=maxiters)
    
    # Create combined CCDData with first frame's header as template
    combined = CCDData(median, unit=frames[0].unit, header=frames[0].header.copy())
    
    # Add combination metadata
    combined.header['NCOMBINE'] = (len(frames), 'Number of frames combined')
    combined.header['COMBMETH'] = ('median', 'Combination method')
    combined.header['SIGCLIP'] = (sigma, 'Sigma clipping threshold')
    combined.header['CLIPITER'] = (maxiters, 'Max clipping iterations')
    
    return combined


def create_master_bias(bias_frames: List[BiasFrame], method: str = 'median') -> MasterBias:
    """
    Create master bias from multiple bias frames.
    
    Parameters
    ----------
    bias_frames : list of BiasFrame
        Bias frames to combine
    method : str
        Combination method ('median' or 'mean')
    
    Returns
    -------
    MasterBias
        Master bias frame
    """
    # Convert to CCDData
    ccd_list = [CCDData(frame.data, unit=u.electron) for frame in bias_frames]
    
    # Combine
    combined = sigma_clipped_median_combine(ccd_list)
    
    # Create MasterBias
    master_bias = MasterBias(
        data=combined.data,
        source_frames=bias_frames,
        combination_method=method
    )
    
    return master_bias


def create_master_flat(flat_frames: List[FlatFrame], master_bias: MasterBias, method: str = 'median') -> MasterFlat:
    """
    Create master flat from multiple flat frames.
    
    Parameters
    ----------
    flat_frames : list of FlatFrame
        Flat frames to combine
    master_bias : MasterBias
        Master bias for subtraction
    method : str
        Combination method ('median' or 'mean')
    
    Returns
    -------
    MasterFlat
        Master flat frame
    """
    # Bias-subtract each flat
    corrected_flats = []
    for flat in flat_frames:
        corrected = flat.data - master_bias.data
        corrected_flats.append(CCDData(corrected, unit=u.electron))
    
    # Combine
    combined = sigma_clipped_median_combine(corrected_flats)
    
    # Normalize to median
    median_value = np.median(combined.data)
    normalized = combined.data / median_value
    
    # Create MasterFlat
    master_flat = MasterFlat(
        data=normalized,
        source_frames=flat_frames,
        combination_method=method
    )
    
    return master_flat
