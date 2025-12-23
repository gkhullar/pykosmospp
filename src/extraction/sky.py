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
        2D spectral data with shape (ny_spectral, nx_spatial)
        Array indexing: data_2d[y, x] where Y=spectral/wavelength, X=spatial
    traces : List[Trace]
        List of detected traces
    sky_buffer : int, optional
        Buffer pixels from trace edges in spatial direction (default: 30)
    sigma_clip : float, optional
        Sigma clipping threshold (default: 3.0)
    mask : np.ndarray, optional
        Bad pixel mask (True = bad)
        
    Returns
    -------
    np.ndarray
        Sky background 2D array (same shape as input)
    """
    ny_spectral, nx_spatial = data_2d.shape
    
    if mask is None:
        mask = np.zeros_like(data_2d, dtype=bool)
    
    # Create trace mask (regions containing traces in spatial/X direction)
    trace_mask = np.zeros((ny_spectral, nx_spatial), dtype=bool)
    
    for trace in traces:
        for y_idx, x_center in enumerate(trace.spatial_positions):
            x_center = int(x_center)
            x_start = max(0, x_center - sky_buffer)
            x_end = min(nx_spatial, x_center + sky_buffer + 1)
            trace_mask[y_idx, x_start:x_end] = True
    
    # Sky regions are where trace_mask is False
    sky_mask = trace_mask | mask
    
    # Estimate sky for each spectral/Y pixel
    sky_spectrum = np.zeros(ny_spectral)
    
    for y_idx in range(ny_spectral):
        sky_row = data_2d[y_idx, :]  # Extract row at fixed Y
        row_mask = sky_mask[y_idx, :]
        
        # Select sky pixels (in spatial/X direction)
        sky_pixels = sky_row[~row_mask]
        
        if len(sky_pixels) < 5:
            # Not enough sky pixels, use median of entire row
            sky_spectrum[y_idx] = np.median(sky_row[~mask[y_idx, :]])
        else:
            # Sigma-clipped median
            _, median, _ = sigma_clipped_stats(
                sky_pixels,
                sigma=sigma_clip,
                maxiters=3
            )
            sky_spectrum[y_idx] = median
    
    # Broadcast to 2D (repeat spectrum across spatial/X direction)
    sky_2d = np.tile(sky_spectrum[:, np.newaxis], (1, nx_spatial))
    
    return sky_2d
