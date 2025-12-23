"""
Arc line identification from arc lamp spectra.

Per tasks.md T032 and research.md §5: Detect emission lines using peak-finding.
Constitution §VI: Consult notebooks/ for arc line detection examples.
"""

from typing import List, Tuple
import numpy as np
from scipy import signal
from astropy.nddata import CCDData


def detect_arc_lines(arc_data: CCDData, 
                     detection_threshold: float = 5.0,
                     min_separation: int = 5,
                     prominence_factor: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect arc emission lines in 1D arc spectrum.
    
    Per research.md §5: scipy.signal.find_peaks with detection threshold.
    Returns pixel positions and intensities of detected lines.
    
    Parameters
    ----------
    arc_data : CCDData
        Arc lamp 2D frame (will be collapsed to 1D)
    detection_threshold : float, optional
        Detection threshold in sigma above continuum (default: 5.0)
    min_separation : int, optional
        Minimum separation between peaks in pixels (default: 5)
    prominence_factor : float, optional
        Prominence as fraction of peak height (default: 0.5)
        
    Returns
    -------
    pixel_positions : np.ndarray
        Pixel positions of detected lines
    intensities : np.ndarray
        Intensities (heights) of detected lines
    """
    # Collapse 2D arc to 1D spectrum (median along spatial/X axis)
    # Arc frame: X=spatial (horizontal), Y=spectral/wavelength (vertical)
    # Emission lines run vertically, uniformly illuminated horizontally
    spectrum_1d = np.median(arc_data.data, axis=1)  # Collapse along spatial axis
    
    # Estimate continuum level (robust median with sigma-clipping)
    continuum = np.median(spectrum_1d)
    
    # Estimate noise (robust standard deviation)
    residuals = spectrum_1d - continuum
    noise = 1.4826 * np.median(np.abs(residuals))  # MAD estimator
    
    # Detection threshold in absolute units
    threshold = continuum + detection_threshold * noise
    
    # Find peaks using scipy.signal.find_peaks
    peak_indices, properties = signal.find_peaks(
        spectrum_1d,
        height=threshold,
        distance=min_separation,
        prominence=prominence_factor * (spectrum_1d.max() - continuum)
    )
    
    # Extract peak intensities
    intensities = spectrum_1d[peak_indices]
    
    # Refine peak positions using centroid around peak
    refined_positions = []
    for idx in peak_indices:
        # Use 5-pixel window around peak for centroid
        window_start = max(0, idx - 2)
        window_end = min(len(spectrum_1d), idx + 3)
        
        window = spectrum_1d[window_start:window_end]
        window_pixels = np.arange(window_start, window_end)
        
        # Weighted centroid
        if np.sum(window) > 0:
            centroid = np.sum(window_pixels * window) / np.sum(window)
            refined_positions.append(centroid)
        else:
            refined_positions.append(float(idx))
    
    return np.array(refined_positions), intensities
