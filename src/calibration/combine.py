"""
Frame combination utilities using sigma-clipped statistics.

Per research.md §7: Median combine with sigma-clipping for bias and flat frames.
"""

from typing import List, Optional, Tuple
import numpy as np
from astropy.nddata import CCDData
from astropy.stats import sigma_clipped_stats
import astropy.units as u
import logging

from ..models import MasterBias, MasterFlat, BiasFrame, FlatFrame

logger = logging.getLogger(__name__)


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


def create_master_bias(bias_frames: List[BiasFrame], 
                       method: str = 'median',
                       clip_edges: bool = False,
                       edge_detection_params: Optional[dict] = None) -> MasterBias:
    """
    Create master bias from multiple bias frames.
    
    Parameters
    ----------
    bias_frames : list of BiasFrame
        Bias frames to combine
    method : str
        Combination method ('median' or 'mean')
    clip_edges : bool, optional
        If True, automatically detect and clip vignetted edges (default: False)
    edge_detection_params : dict, optional
        Parameters for edge detection: {'threshold': 0.1, 'edge_buffer': 10}
    
    Returns
    -------
    MasterBias
        Master bias frame (with edges clipped if clip_edges=True)
        
    Notes
    -----
    Edge clipping removes vignetted regions at detector edges where
    flat field response drops to near zero. This should be applied
    consistently to bias, flat, and all science frames.
    """
    # Convert to CCDData
    ccd_list = [CCDData(frame.data, unit=u.electron) for frame in bias_frames]
    
    # Combine
    combined = sigma_clipped_median_combine(ccd_list)
    
    # Clip edges if requested (but bias alone isn't great for edge detection)
    # Edge clipping is typically better determined from flat field
    # This option exists for consistency if edges already determined elsewhere
    if clip_edges:
        logger.warning("Edge clipping from bias frames - consider using flat field "
                      "to determine edges instead for better accuracy")
    
    # Calculate statistics
    bias_level = float(np.median(combined.data))
    bias_stdev = float(np.std(combined.data))
    
    # Create MasterBias with correct fields
    master_bias = MasterBias(
        data=combined,
        n_combined=len(bias_frames),
        bias_level=bias_level,
        bias_stdev=bias_stdev,
        provenance={'method': method, 'n_frames': len(bias_frames)}
    )
    
    return master_bias


def create_master_flat(flat_frames: List[FlatFrame], 
                       master_bias: MasterBias, 
                       method: str = 'median',
                       clip_edges: bool = True,
                       edge_detection_params: Optional[dict] = None) -> Tuple[MasterFlat, Optional[Tuple[int, int]]]:
    """
    Create master flat from multiple flat frames with automatic edge clipping.
    
    Parameters
    ----------
    flat_frames : list of FlatFrame
        Flat frames to combine
    master_bias : MasterBias
        Master bias for subtraction
    method : str
        Combination method ('median' or 'mean')
    clip_edges : bool, optional
        If True, automatically detect and clip vignetted edges (default: True)
    edge_detection_params : dict, optional
        Parameters for edge detection: {'threshold': 0.1, 'edge_buffer': 10}
        
    Returns
    -------
    master_flat : MasterFlat
        Master flat frame (with edges clipped if clip_edges=True)
    edge_bounds : tuple of (int, int) or None
        (x_min, x_max) edge boundaries if edges were clipped, else None
        
    Notes
    -----
    Edge clipping removes vignetted regions at detector edges where
    flat field response drops to near zero. The returned edge_bounds
    should be applied to all subsequent frames (arcs, science) to
    maintain alignment.
    
    Examples
    --------
    >>> master_flat, edges = create_master_flat(flats, master_bias, clip_edges=True)
    >>> if edges:
    ...     x_min, x_max = edges
    ...     # Apply same clipping to science frames
    ...     clipped_science = science_data[:, x_min:x_max]
    """
    # Bias-subtract each flat
    corrected_flats = []
    for flat in flat_frames:
        corrected = flat.data - master_bias.data.data
        corrected_flats.append(CCDData(corrected, unit=u.electron))
    
    # Combine
    combined = sigma_clipped_median_combine(corrected_flats)
    
    # Normalize to median in central region
    ny, nx = combined.data.shape
    y_start, y_end = ny // 4, 3 * ny // 4
    x_start, x_end = nx // 4, 3 * nx // 4
    central_region = combined.data[y_start:y_end, x_start:x_end]
    median_value = np.median(central_region)
    normalized_data = combined.data / median_value
    normalized = CCDData(normalized_data, unit=u.dimensionless_unscaled, 
                        header=combined.header.copy())
    
    # Detect and clip vignetted edges
    edge_bounds = None
    if clip_edges:
        from ..quality.detector_artifacts import detect_vignetted_edges, clip_detector_edges
        
        # Set default parameters
        params = edge_detection_params or {}
        threshold = params.get('threshold', 0.1)
        edge_buffer = params.get('edge_buffer', 10)
        
        # Detect edges from normalized flat (best for edge detection)
        x_min, x_max = detect_vignetted_edges(
            normalized_data,
            threshold=threshold,
            edge_buffer=edge_buffer
        )
        
        # Clip the normalized flat
        normalized = clip_detector_edges(normalized, x_min, x_max)
        edge_bounds = (x_min, x_max)
        
        logger.info(f"Master flat edges clipped: X=[{x_min}:{x_max}], "
                   f"shape now {normalized.data.shape}")
    
    # Calculate bad pixel fraction (pixels far from 1.0)
    bad_mask = (normalized.data < 0.5) | (normalized.data > 1.5)
    bad_pixel_fraction = float(np.sum(bad_mask) / bad_mask.size)
    
    # Update normalization region for clipped data
    if edge_bounds:
        x_min, x_max = edge_bounds
        nx_clipped = normalized.data.shape[1]
        x_start_norm = max(0, nx_clipped // 4)
        x_end_norm = min(nx_clipped, 3 * nx_clipped // 4)
        normalization_region = (y_start, y_end, x_start_norm, x_end_norm)
    else:
        normalization_region = (y_start, y_end, x_start, x_end)
    
    # Create MasterFlat with correct fields
    master_flat = MasterFlat(
        data=normalized,
        n_combined=len(flat_frames),
        normalization_region=normalization_region,
        bad_pixel_fraction=bad_pixel_fraction,
        provenance={'method': method, 'n_frames': len(flat_frames), 
                   'edges_clipped': clip_edges,
                   'edge_bounds': edge_bounds}
    )
    
    return master_flat, edge_bounds
