"""
Wavelength solution fitting with robust methods.

Per tasks.md T034 and research.md §5: Chebyshev polynomial with iterative
sigma-clipping and BIC order selection.
Constitution §VI: Consult notebooks/ for wavelength fitting workflows.
"""

from typing import Tuple, Optional
import numpy as np
from numpy.polynomial import chebyshev
from scipy.interpolate import PchipInterpolator, UnivariateSpline
from scipy.optimize import minimize

from ..models import WavelengthSolution, ArcFrame


def fit_wavelength_solution(pixels: np.ndarray,
                            wavelengths: np.ndarray,
                            arc_frame: Optional[ArcFrame] = None,
                            poly_type: str = 'chebyshev',
                            order: Optional[int] = None,
                            sigma_clip: float = 3.0,
                            max_iterations: int = 5,
                            min_order: int = 3,
                            max_order: int = 7,
                            use_bic: bool = True,
                            order_range: Optional[Tuple[int, int]] = None,
                            strict_rms: bool = True,
                            calibration_method: str = 'line_matching',
                            template_used: str = None,
                            dtw_parameters: dict = None,
                            monotonic: bool = False,
                            spline_order: int = 3,
                            spline_smoothing: Optional[float] = None) -> WavelengthSolution:
    """
    Fit wavelength solution with robust fitting and BIC order selection.
    
    Per research.md §5: Chebyshev polynomial with iterative sigma-clipping.
    BIC (Bayesian Information Criterion) selects optimal polynomial order.
    Validates RMS <0.1 Å per FR-008 implementation target.
    
    Parameters
    ----------
    pixels : np.ndarray
        Pixel positions of matched arc lines
    wavelengths : np.ndarray
        Catalog wavelengths of matched lines (Angstroms)
    arc_frame : ArcFrame, optional
        Source arc frame (if None, creates minimal solution without frame reference)
    poly_type : str, optional
        Polynomial type (default: 'chebyshev')
    order : int, optional
        Polynomial order. If None, uses BIC selection.
    sigma_clip : float, optional
        Sigma clipping threshold (default: 3.0)
    max_iterations : int, optional
        Maximum clipping iterations (default: 5)
    min_order : int, optional
        Minimum order for BIC selection (default: 3)
    max_order : int, optional
        Maximum order for BIC selection (default: 7)
    use_bic : bool, optional
        Use BIC for order selection (default: True)
    order_range : tuple of (int, int), optional
        Alternative way to specify (min_order, max_order). Overrides individual parameters.
    strict_rms : bool, optional
        If True (default), raises error if RMS > 0.2 Å. Set False for testing.
    calibration_method : str, optional
        Method used: 'line_matching' (default) or 'dtw' (Constitution Principle III)
    template_used : str, optional
        Name of arc template file used (for DTW method, provenance tracking)
    dtw_parameters : dict, optional
        DTW parameters used (e.g., peak_threshold, step_pattern) for provenance
    monotonic : bool, optional
        If True, use monotonic smoothing spline instead of polynomial.
        Uses UnivariateSpline with automatic smoothing factor.
        Ensures wavelength is strictly monotonic (checks derivative).
        Recommended for extrapolation beyond calibrated region.
    spline_order : int, optional
        Spline order (degree) for monotonic spline (default: 3 for cubic).
        Can use 3 (cubic) or 5 (quintic) for smooth fits.
    spline_smoothing : float, optional
        Smoothing factor for spline. If None, auto-determined as n_points * 0.5.
        Higher values = smoother fit with larger residuals.
        Lower values = closer fit to data points.
        
    Returns
    -------
    WavelengthSolution
        Fitted wavelength solution
        
    Raises
    ------
    ValueError
        If fit fails or RMS exceeds threshold
    """
    # Support order_range parameter for backwards compatibility
    if order_range is not None:
        min_order, max_order = order_range
    
    if len(pixels) < 10:
        raise ValueError(f"Need at least 10 matched lines, got {len(pixels)}")
    
    # Determine polynomial order (not used for monotonic spline)
    if order is None and use_bic and not monotonic:
        order = _select_order_by_bic(pixels, wavelengths, min_order, max_order, poly_type)
    elif order is None:
        order = 5 if not monotonic else 3  # Lower default for spline
    
    # For monotonic spline, use UnivariateSpline with automatic data extent detection
    if monotonic:
        poly_type = 'spline'  # Override poly_type for monotonic smoothing spline
        order = spline_order  # Use spline_order parameter
    
    # Normalize pixels for Chebyshev
    pix_min, pix_max = pixels.min(), pixels.max()
    pix_norm = 2.0 * (pixels - pix_min) / (pix_max - pix_min) - 1.0
    
    # Detect safe extrapolation range (where arc data actually exists)
    # Add safety margin: only extrapolate ~200-500 pixels beyond last calibration point
    safe_extrapolation_margin = 300  # pixels beyond last calibrated point
    pix_extrapolate_max = pix_max + safe_extrapolation_margin
    
    # Iterative sigma-clipped fitting
    mask = np.ones(len(pixels), dtype=bool)
    
    for iteration in range(max_iterations):
        # Fit polynomial or spline to unmasked points
        if monotonic:
            # Use UnivariateSpline with smoothing for proper fit (not interpolation)
            pix_masked = pixels[mask]
            wave_masked = wavelengths[mask]
            n_points = len(pix_masked)
            
            # Determine smoothing factor
            if spline_smoothing is None:
                # Auto-determine: balance between fit quality and smoothness
                # Higher smoothing = smoother curve, more likely to be monotonic
                # Start with s ~ 1.0 * n_points for significant smoothing
                s = 1.0 * n_points
            else:
                s = spline_smoothing
            
            # Fit smoothing spline with specified order (3=cubic, 5=quintic)
            try:
                spline = UnivariateSpline(pix_masked, wave_masked, 
                                         s=s, k=min(order, n_points-1), ext=0)
                
                # Check monotonicity by evaluating derivative
                # For KOSMOS Red: wavelength decreases with pixel (dλ/dpix < 0)
                test_pixels = np.linspace(pix_masked.min(), pix_masked.max(), 100)
                derivatives = spline.derivative()(test_pixels)
                
                is_monotonic = np.all(derivatives < 0)  # Should be decreasing
                
                if not is_monotonic:
                    # Not monotonic - reduce smoothing and try again
                    s_reduced = s * 0.1
                    spline = UnivariateSpline(pix_masked, wave_masked,
                                            s=s_reduced, k=min(order, n_points-1), ext=0)
                    derivatives = spline.derivative()(test_pixels)
                    is_monotonic = np.all(derivatives < 0)
                    
                    if not is_monotonic:
                        # Still not monotonic - use PCHIP as guaranteed monotonic fallback
                        print(f"  Warning: Spline not monotonic even with reduced smoothing, using PCHIP")
                        spline = PchipInterpolator(pix_masked, wave_masked, extrapolate=True)
                        s = 0  # PCHIP is interpolating (no smoothing)
                
                fitted_waves = spline(pixels)
                coeffs = spline  # Store spline object
                
            except Exception as e:
                # Fallback to PCHIP if spline fitting fails
                print(f"  Warning: Spline fitting failed ({e}), using PCHIP")
                spline = PchipInterpolator(pix_masked, wave_masked, extrapolate=True)
                fitted_waves = spline(pixels)
                coeffs = spline
                s = 0
                
        elif poly_type == 'chebyshev':
            coeffs = chebyshev.chebfit(pix_norm[mask], wavelengths[mask], order)
            fitted_waves = chebyshev.chebval(pix_norm, coeffs)
        else:
            coeffs = np.polyfit(pixels[mask], wavelengths[mask], order)
            fitted_waves = np.polyval(coeffs, pixels)
        
        # Calculate residuals
        residuals = wavelengths - fitted_waves
        
        # Sigma clipping
        rms = np.sqrt(np.mean(residuals[mask]**2))
        outliers = np.abs(residuals) > sigma_clip * rms
        
        # Update mask
        new_mask = mask & ~outliers
        
        # Check for convergence
        if np.array_equal(mask, new_mask):
            break
        
        mask = new_mask
        
        # Require at least 80% of lines retained
        if np.sum(mask) / len(mask) < 0.8:
            raise ValueError(
                f"Too many outliers rejected: {np.sum(~mask)}/{len(mask)} "
                f"({100*np.sum(~mask)/len(mask):.1f}%). Check line identifications."
            )
    
    # Final fit statistics
    final_residuals = residuals[mask]
    rms_residual = float(np.sqrt(np.mean(final_residuals**2)))
    n_lines_used = int(np.sum(mask))
    
    # Validate RMS threshold (implementation target: <0.1 Å)
    if rms_residual > 0.1:
        # Warning but not fatal (acceptance criterion is <0.2 Å)
        import warnings
        warnings.warn(
            f"Wavelength RMS {rms_residual:.3f} Å exceeds implementation target 0.1 Å "
            f"(but within acceptance criterion 0.2 Å)"
        )
    
    if strict_rms and rms_residual > 0.2:
        raise ValueError(
            f"Wavelength RMS {rms_residual:.3f} Å exceeds acceptance criterion 0.2 Å"
        )
    
    # Create wavelength solution
    wavelength_range = (float(wavelengths.min()), float(wavelengths.max()))
    pixel_range = (float(pix_min), float(pix_max))
    
    # Store safe extrapolation limit for spline
    if monotonic:
        pixel_range_extrap = (float(pix_min), float(pix_extrapolate_max))
    else:
        pixel_range_extrap = pixel_range
    
    solution = WavelengthSolution(
        coefficients=coeffs,
        order=order,
        arc_frame=arc_frame,
        n_lines_identified=n_lines_used,
        rms_residual=rms_residual,
        wavelength_range=wavelength_range,
        poly_type=poly_type,
        pixel_range=pixel_range_extrap,  # Use extrapolation limit
        calibration_method=calibration_method,
        template_used=template_used,
        dtw_parameters=dtw_parameters
    )
    
    return solution


def _select_order_by_bic(pixels: np.ndarray, 
                         wavelengths: np.ndarray,
                         min_order: int,
                         max_order: int,
                         poly_type: str) -> int:
    """
    Select polynomial order using Bayesian Information Criterion.
    
    BIC = n*ln(RSS/n) + k*ln(n)
    where n = number of data points, k = number of parameters, RSS = residual sum of squares
    
    Lower BIC indicates better model (balances fit quality vs complexity).
    
    Parameters
    ----------
    pixels : np.ndarray
        Pixel positions
    wavelengths : np.ndarray
        Wavelengths
    min_order : int
        Minimum order to test
    max_order : int
        Maximum order to test
    poly_type : str
        Polynomial type
        
    Returns
    -------
    int
        Optimal polynomial order
    """
    n_points = len(pixels)
    pix_norm = 2.0 * (pixels - pixels.min()) / (pixels.max() - pixels.min()) - 1.0
    
    bic_values = []
    orders = range(min_order, max_order + 1)
    
    for order in orders:
        # Fit polynomial
        if poly_type == 'chebyshev':
            coeffs = chebyshev.chebfit(pix_norm, wavelengths, order)
            fitted = chebyshev.chebval(pix_norm, coeffs)
        else:
            coeffs = np.polyfit(pixels, wavelengths, order)
            fitted = np.polyval(coeffs, pixels)
        
        # Calculate RSS
        residuals = wavelengths - fitted
        rss = np.sum(residuals**2)
        
        # Calculate BIC
        n_params = order + 1
        bic = n_points * np.log(rss / n_points) + n_params * np.log(n_points)
        bic_values.append(bic)
    
    # Select order with minimum BIC
    best_idx = np.argmin(bic_values)
    best_order = orders[best_idx]
    
    return best_order
