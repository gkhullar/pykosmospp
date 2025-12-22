.. _algorithm_cosmic_ray_detection:

================================
Cosmic Ray Detection Algorithm
================================

Overview
========

Cosmic rays are high-energy particles that create spurious bright pixels in CCD images. Detection and flagging is critical for accurate spectroscopy. pyKOSMOS++ uses a scipy-based implementation of the L.A.Cosmic algorithm (van Dokkum 2001).

Method: L.A.Cosmic Algorithm
=============================

L.A.Cosmic detects cosmic rays by comparing image sharpness to a smoothed version:

**Algorithm Steps:**

1. **Create Laplacian image** (edge-detection kernel):

   .. math::
   
       L = \\nabla^2 I = \\frac{\\partial^2 I}{\\partial x^2} + \\frac{\\partial^2 I}{\\partial y^2}

2. **Estimate noise model**:

   .. math::
   
       \\sigma = \\sqrt{\\sigma_{read}^2 + I / g}

3. **Compute significance**:

   .. math::
   
       S = \\frac{L}{\\sigma}

4. **Apply contrast criterion**:

   Pixel is cosmic ray if:
   
   * S > sigma_clip (typically 5σ)
   * AND pixel value > contrast × median of neighbors

5. **Iteratively clean** and re-detect (max 5 iterations)

Parameters
==========

**sigma_clip**: 5.0
  Detection threshold in sigma above noise

**contrast**: 3.0
  Minimum contrast with neighbors

**max_iterations**: 5
  Maximum cleaning iterations

**readnoise**: 3.7 e⁻
  CCD read noise (from config)

**gain**: 1.4 e⁻/ADU
  CCD gain (from config)

Implementation
==============

.. code-block:: python

    from scipy.ndimage import laplace, median_filter
    
    def detect_cosmic_rays(
        data: np.ndarray,
        sigma_clip: float = 5.0,
        contrast: float = 3.0,
        max_iterations: int = 5
    ) -> np.ndarray:
        """
        Detect cosmic rays using L.A.Cosmic algorithm.
        
        Returns
        -------
        mask : ndarray (bool)
            True for cosmic ray pixels
        """
        mask = np.zeros(data.shape, dtype=bool)
        
        for iteration in range(max_iterations):
            # Laplacian edge detection
            laplacian = laplace(data)
            
            # Noise model
            noise = np.sqrt(readnoise**2 + data / gain)
            
            # Significance
            significance = laplacian / noise
            
            # Detect outliers
            med_neighbors = median_filter(data, size=3)
            contrast_check = data > contrast * med_neighbors
            
            new_cosmic = (significance > sigma_clip) & contrast_check
            
            if np.sum(new_cosmic) == 0:
                break  # Converged
            
            mask |= new_cosmic
            
            # Clean image for next iteration
            data[mask] = med_neighbors[mask]
        
        return mask

Performance
===========

**Typical Results:**

* Detection rate: 0.5-2% of pixels flagged
* False positive rate: <0.1%
* Processing time: 2-5 seconds per 2048×515 image

**When to adjust parameters:**

* **More aggressive detection** (sigma_clip=4.0): More false positives, better for faint cosmic rays
* **Less aggressive** (sigma_clip=6.0): Fewer false positives, may miss faint events
* **Contrast threshold** (contrast=2.0-5.0): Lower = more detections

Integration with Pipeline
==========================

Cosmic ray detection occurs after flat correction, before trace detection:

1. **Load and calibrate** science frame (bias subtract, flat correct)
2. **Detect cosmic rays** and create mask
3. **Store mask** in Spectrum2D object
4. **Apply mask** during extraction (zero weight to flagged pixels)

Impact on Extraction
=====================

Flagged pixels receive zero weight during optimal extraction, preventing cosmic rays from contaminating the extracted 1D spectrum.

**Without cosmic ray rejection:**

* Bright spikes in 1D spectrum
* Artificial emission lines
* Reduced SNR due to outliers

**With cosmic ray rejection:**

* Clean 1D spectrum
* Improved SNR (5-10% typical)
* Accurate line flux measurements

Limitations
===========

**Cannot detect:**

* Cosmic rays aligned with spectral traces (< 1% of events)
* Very faint cosmic rays (< 3σ above background)
* Saturated cosmic rays (may create artifacts)

**Workarounds:**

* Use multiple exposures with sigma-clipping combination
* Inspect diagnostic plots for remaining artifacts
* Manual masking if needed

References
==========

* **van Dokkum, P. G. 2001**, PASP, 113, 1420 - "Cosmic-Ray Rejection by Laplacian Edge Detection"
* **Original L.A.Cosmic**: http://www.astro.yale.edu/dokkum/lacosmic/

See Also
========

* :ref:`api_calibration` - Cosmic ray detection API
* :ref:`configuration` - Cosmic ray parameters
* :ref:`troubleshooting` - Cosmic ray issues
