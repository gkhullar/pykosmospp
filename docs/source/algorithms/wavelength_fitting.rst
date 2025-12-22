.. _algorithm_wavelength_fitting:

==============================
Wavelength Fitting Algorithm
==============================

Overview
========

Wavelength calibration establishes the mapping between CCD pixel positions and wavelengths using arc lamp emission lines. pyKOSMOS++ uses Chebyshev polynomial fitting with Bayesian Information Criterion (BIC) order selection.

Method
======

1. **Detect arc lines** in 1D collapsed arc spectrum
2. **Match to catalog** wavelengths (He-Ne-Ar, Ar, Kr, Ne)
3. **Fit Chebyshev polynomial** with iterative sigma-clipping
4. **Select optimal order** using BIC to balance fit quality vs overfitting
5. **Validate** RMS residual < threshold (default 0.1Å)

Chebyshev Polynomials
=====================

Chebyshev polynomials are preferred over standard polynomials for numerical stability:

.. math::

    \\lambda(x) = \\sum_{i=0}^{n} c_i T_i(x_{norm})

where :math:`T_i` are Chebyshev polynomials and :math:`x_{norm}` is pixel position normalized to [-1, 1].

**Advantages:**

* Numerically stable for high orders (up to n=10)
* Orthogonal basis reduces coefficient correlations
* Better conditioned than power series

BIC Order Selection
===================

Bayesian Information Criterion balances fit quality with model complexity:

.. math::

    BIC = n \\ln(\\sigma^2) + k \\ln(n)

where:

* n = number of arc lines
* σ² = variance of residuals
* k = number of parameters (polynomial order + 1)

**Algorithm:**

1. Fit polynomials for orders 3-7 (configurable)
2. Compute BIC for each order
3. Select order with minimum BIC
4. Validate RMS < 0.1Å threshold

Iterative Sigma-Clipping
=========================

Removes outlier lines from fit:

1. Fit polynomial to all matched lines
2. Compute residuals: :math:`r_i = \\lambda_i - f(x_i)`
3. Calculate RMS: :math:`\\sigma = \\sqrt{\\sum r_i^2 / n}`
4. Reject lines with :math:`|r_i| > 3\\sigma`
5. Repeat until convergence or max iterations (default 5)

**Typical results:**

* 80-95% of lines retained
* RMS improves by 20-40% after clipping

Parameters
==========

**order_range**: (3, 7)
  Range of polynomial orders to test

**sigma_clip**: 3.0
  Threshold for outlier rejection (sigma)

**max_iterations**: 5
  Maximum sigma-clipping iterations

**rms_threshold**: 0.1
  Maximum acceptable RMS in Angstroms

Performance
===========

**Typical Metrics:**

* Lines detected: 40-60 for He-Ne-Ar
* Lines used in fit: 35-55 (after clipping)
* RMS residual: 0.05-0.15Å
* Optimal order: 5-7 for KOSMOS

References
==========

* **Chebyshev polynomials**: Press et al. "Numerical Recipes" Ch. 5
* **BIC**: Schwarz, G. 1978, Annals of Statistics, 6, 461
* **pyKOSMOS**: Davenport et al. 2023, DOI:10.5281/zenodo.10152905

See Also
========

* :ref:`api_wavelength` - API reference
* :ref:`configuration` - Wavelength parameters
* :ref:`troubleshooting` - Wavelength calibration issues
