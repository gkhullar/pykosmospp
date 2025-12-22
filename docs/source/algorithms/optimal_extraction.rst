.. _algorithm_optimal_extraction:

===============================
Optimal Extraction Algorithm
===============================

Overview
========

Optimal extraction (Horne 1986) maximizes signal-to-noise ratio by weighting spatial pixels according to the spatial profile and inverse variance. This outperforms simple aperture summation, especially for faint sources.

Method: Horne 1986 Algorithm
=============================

The optimal extraction algorithm computes extracted flux as:

.. math::

    f_\\lambda = \\frac{\\sum_y P(y) \\cdot D(y, \\lambda) / V(y, \\lambda)}{\\sum_y P(y)^2 / V(y, \\lambda)}

where:

* :math:`D(y, \\lambda)` = 2D sky-subtracted data
* :math:`P(y)` = normalized spatial profile
* :math:`V(y, \\lambda)` = variance (read noise + Poisson)
* :math:`y` = spatial coordinate
* :math:`\\lambda` = wavelength coordinate

Algorithm Steps
===============

1. **Fit Spatial Profile**
   
   Fit Gaussian (or Moffat) to spatial cross-section:
   
   .. math::
   
       P(y) = A \\exp\\left(-\\frac{(y - y_0)^2}{2\\sigma^2}\\right)

2. **Compute Variance**
   
   Propagate noise from read noise and Poisson statistics:
   
   .. math::
   
       V(y, \\lambda) = (\\text{readnoise})^2 + \\text{gain} \\cdot D(y, \\lambda)

3. **Weighted Extraction**
   
   Apply profile weights normalized by variance

4. **Cosmic Ray Masking**
   
   Pixels flagged as cosmic rays get zero weight

Advantages Over Aperture Extraction
====================================

**Signal-to-Noise Improvement:**

* Typical gain: 10-30% for point sources
* Greater improvement for faint sources
* Down-weights noisy edge pixels

**Optimal Use of Data:**

* All pixels in aperture contribute
* Weights match actual spatial profile
* Robust to seeing variations

Comparison: Optimal vs Boxcar
==============================

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - Optimal Extraction
     - Boxcar (Aperture Sum)
   * - SNR
     - Higher (weighted by profile)
     - Lower (equal weights)
   * - Implementation
     - Complex (profile fitting)
     - Simple (sum pixels)
   * - Robustness
     - Sensitive to profile errors
     - Robust, less optimal
   * - Speed
     - Slower (~2×)
     - Faster
   * - Best for
     - Faint sources, precision
     - Bright sources, quick-look

When to Use Boxcar Instead
===========================

**Boxcar extraction preferred when:**

* Very bright sources (SNR > 50)
* Poor spatial profile fit (extended/irregular)
* Quick-look / exploratory analysis
* Profile varies significantly along spectrum

**Configure in YAML:**

.. code-block:: yaml

    extraction:
      method: boxcar  # or 'optimal'
      aperture_width: 10

Spatial Profile Fitting
========================

**Gaussian Profile:**

.. math::

    P(y) = A \\exp\\left(-\\frac{(y - y_0)^2}{2\\sigma^2}\\right)

* **Best for:** Point sources, good seeing
* **Parameters:** amplitude, center, width

**Moffat Profile:**

.. math::

    P(y) = A \\left[1 + \\left(\\frac{y - y_0}{\\alpha}\\right)^2\\right]^{-\\beta}

* **Best for:** Extended wings, poor seeing
* **Parameters:** amplitude, center, width, power

**Empirical Profile:**

* Use actual data profile without parametric fit
* Robust when profile shape unknown
* Fallback if Gaussian/Moffat fail

Variance Propagation
====================

Total variance combines read noise and Poisson noise:

.. math::

    V(y, \\lambda) = \\sigma_{read}^2 + \\frac{D(y, \\lambda)}{g}

where:

* :math:`\\sigma_{read}` = read noise (electrons)
* :math:`g` = gain (e⁻/ADU)
* :math:`D` = data (ADU)

**Uncertainty in extracted flux:**

.. math::

    \\sigma_f = \\sqrt{\\sum_y \\frac{P(y)^2}{V(y, \\lambda)}}

Parameters
==========

**aperture_width**: 10 pixels
  Width of extraction region around trace center

**profile_type**: 'Gaussian'
  Spatial profile model ('Gaussian', 'Moffat', 'empirical')

**method**: 'optimal'
  Extraction method ('optimal', 'boxcar')

Performance
===========

**Typical Metrics:**

* SNR improvement: 15-25% over boxcar
* Extraction time: 1-3 seconds per trace
* Profile fit χ²: <2.0 for good data

**Failure Modes:**

1. **Poor profile fit** (χ² > 10):
   
   * Use empirical profile
   * Or fall back to boxcar

2. **Negative flux values**:
   
   * Sky over-subtracted
   * Increase sky_buffer parameter

References
==========

* **Horne, K. 1986**, PASP, 98, 609 - "An Optimal Extraction Algorithm for CCD Spectroscopy"
* **Marsh, T. R. 1989**, PASP, 101, 1032 - "The Extraction of Highly Distorted Spectra"

See Also
========

* :ref:`api_extraction` - API reference
* :ref:`configuration` - Extraction parameters
* :ref:`algorithm_trace_detection` - Trace detection
