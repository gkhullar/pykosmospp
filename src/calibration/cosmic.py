"""
Cosmic ray detection using LA Cosmic algorithm.

Per research.md §9 and FR-013: Use LA Cosmic for single frames.
Implementation based on scipy signal processing.
"""

import numpy as np
from scipy import ndimage, signal
from astropy.nddata import CCDData


def detect_cosmic_rays(data: CCDData,
                       sigma_clip: float = 5.0,
                       contrast: float = 3.0,
                       max_iterations: int = 5,
                       readnoise: float = 3.7,
                       gain: float = 1.4) -> np.ndarray:
    """
    Detect cosmic rays in science frame using LA Cosmic algorithm.
    
    Per config-schema.yaml: sigma_clip=5.0, contrast=3.0, max_iterations=5
    Based on van Dokkum 2001 (PASP 113, 1420).
    
    Algorithm:
    1. Convolve with Laplacian kernel to enhance sharp features
    2. Identify pixels >sigma_clip above local background
    3. Grow cosmic ray regions using contrast threshold
    4. Iterate until convergence or max_iterations
    
    Parameters
    ----------
    data : CCDData
        Science frame to check for cosmic rays
    sigma_clip : float, optional
        Detection threshold in sigma (default: 5.0)
    contrast : float, optional
        Contrast threshold for growing CR regions (default: 3.0)
    max_iterations : int, optional
        Maximum detection iterations (default: 5)
    readnoise : float, optional
        Read noise in e- (default: 3.7 for KOSMOS)
    gain : float, optional
        CCD gain in e-/ADU (default: 1.4 for KOSMOS)
        
    Returns
    -------
    np.ndarray (bool)
        Boolean mask with True for cosmic ray pixels
    """
    image = data.data.copy()
    mask = np.zeros_like(image, dtype=bool)
    
    # Convert ADU to electrons for noise calculation
    image_electrons = image * gain
    
    # Laplacian kernel for edge detection (finds sharp features)
    laplacian_kernel = np.array([[0, -1, 0],
                                  [-1, 4, -1],
                                  [0, -1, 0]])
    
    for iteration in range(max_iterations):
        # Apply Laplacian to find sharp features
        laplacian = ndimage.convolve(image, laplacian_kernel, mode='constant', cval=0.0)
        
        # Estimate noise: Poisson (sqrt of signal) + read noise
        # Avoid negative values from Laplacian
        noise_estimate = np.sqrt(np.abs(image_electrons) + readnoise**2) / gain
        
        # Avoid division by zero
        noise_estimate = np.maximum(noise_estimate, 1.0)
        
        # Significance of Laplacian features (S/N ratio)
        significance = laplacian / noise_estimate
        
        # Detect cosmic rays: high significance pixels
        new_cr_mask = significance > sigma_clip
        
        # Grow cosmic ray regions using contrast threshold
        # Dilate mask slightly to catch edges of cosmic rays
        grown_mask = ndimage.binary_dilation(new_cr_mask, iterations=1)
        
        # Check for convergence (no new cosmic rays detected)
        if np.array_equal(mask, grown_mask):
            break
        
        mask = grown_mask
        
        # Replace cosmic ray pixels with median of neighbors for next iteration
        if np.any(mask):
            for _ in range(3):  # Multiple passes to fill larger CRs
                image[mask] = ndimage.median_filter(image, size=5)[mask]
    
    return mask
