"""
Sky background estimation for spectroscopy.

Per tasks.md T042 and research.md: Estimate sky from spatial regions away
from traces using robust median with sigma-clipping.
"""

from typing import List, Optional
import numpy as np
from astropy.stats import sigma_clipped_stats


def estimate_sky_background(data_2d: np.ndarray,
                            traces: List,  # List[Trace]
                            sky_buffer: int = 30,
                            sigma_clip: float = 3.0,
                            mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Estimate sky background from regions away from traces.
    
    Per research.md: Uses median of spatial regions outside trace apertures,
    with sigma-clipping to reject cosmic rays and outliers. Broadcasts
    median sky to full 2D frame.
    
    Algorithm:
    1. Identify sky regions (>sky_buffer pixels from any trace)
    2. For each spectral pixel, compute sigma-clipped median of sky
    3. Broadcast sky spectrum to 2D
    
    Parameters
    ----------
    data_2d : np.ndarray
        2D spectral data (spatial x spectral)
    traces : List[Trace]
        List of detected traces
    sky_buffer : int, optional
        Buffer pixels from trace edges (default: 30)
    sigma_clip : float, optional
        Sigma clipping threshold (default: 3.0)
    mask : np.ndarray, optional
        Bad pixel mask (True = bad)
        
    Returns
    -------
    np.ndarray
        Sky background 2D array (same shape as input)
    """
    ny, nx = data_2d.shape
    
    if mask is None:
        mask = np.zeros_like(data_2d, dtype=bool)
    
    # Create trace mask (regions containing traces)
    trace_mask = np.zeros((ny, nx), dtype=bool)
    
    for trace in traces:
        for x_idx, y_center in enumerate(trace.spatial_positions):
            y_center = int(y_center)
            y_start = max(0, y_center - sky_buffer)
            y_end = min(ny, y_center + sky_buffer + 1)
            trace_mask[y_start:y_end, x_idx] = True
    
    # Sky regions are where trace_mask is False
    sky_mask = trace_mask | mask
    
    # Estimate sky for each spectral pixel
    sky_spectrum = np.zeros(nx)
    
    for x_idx in range(nx):
        sky_column = data_2d[:, x_idx]
        column_mask = sky_mask[:, x_idx]
        
        # Select sky pixels
        sky_pixels = sky_column[~column_mask]
        
        if len(sky_pixels) < 5:
            # Not enough sky pixels, use median of entire column
            sky_spectrum[x_idx] = np.median(sky_column[~mask[:, x_idx]])
        else:
            # Sigma-clipped median
            _, median, _ = sigma_clipped_stats(
                sky_pixels,
                sigma=sigma_clip,
                maxiters=3
            )
            sky_spectrum[x_idx] = median
    
    # Broadcast to 2D
    sky_2d = np.tile(sky_spectrum, (ny, 1))
    
    return sky_2d
