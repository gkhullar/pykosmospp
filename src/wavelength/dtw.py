"""
Dynamic Time Warping (DTW) for wavelength calibration.

Implements automatic wavelength identification by aligning observed arc spectra
with pre-calibrated reference templates using DTW. This approach is more robust
than discrete line matching and doesn't require initial dispersion guess.

Based on pyKOSMOS identify_dtw() implementation by James R. A. Davenport.
"""

from typing import Tuple, Optional
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

try:
    from dtw import dtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False


def identify_dtw(arc_spectrum: np.ndarray,
                template_wavelengths: np.ndarray,
                template_flux: np.ndarray,
                step_pattern: str = 'asymmetric',
                window_type: str = 'none',
                open_begin: bool = True,
                open_end: bool = True,
                upsample: bool = False,
                upsample_factor: int = 5,
                peak_spline: bool = False,
                min_peak_separation: int = 5,
                peak_threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify arc line wavelengths using Dynamic Time Warping.
    
    DTW aligns the observed arc spectrum with a pre-calibrated reference template,
    providing a pixel-to-wavelength mapping without requiring line identification
    or initial dispersion guess. This is more robust than traditional line matching.
    
    Algorithm:
    1. Normalize observed and template spectra
    2. Optionally upsample for better alignment
    3. Perform DTW alignment
    4. Extract peaks from observed spectrum
    5. Map pixel positions to wavelengths via DTW alignment
    
    Parameters
    ----------
    arc_spectrum : np.ndarray
        Observed arc spectrum (1D flux array)
    template_wavelengths : np.ndarray
        Reference template wavelengths (Angstroms)
    template_flux : np.ndarray
        Reference template flux values
    step_pattern : str, optional
        DTW step pattern (default: 'asymmetric')
        Options: 'asymmetric', 'symmetric1', 'symmetric2'
    window_type : str, optional
        DTW windowing constraint (default: 'none')
        Options: 'none', 'sakoechiba', 'itakura'
    open_begin : bool, optional
        Allow alignment to start at any point in template (default: True)
    open_end : bool, optional
        Allow alignment to end at any point in template (default: True)
    upsample : bool, optional
        Upsample spectra before DTW for better alignment (default: False)
    upsample_factor : int, optional
        Upsampling factor (default: 5)
    peak_spline : bool, optional
        Use spline interpolation for peak positions (default: False)
    min_peak_separation : int, optional
        Minimum separation between peaks in pixels (default: 5)
    peak_threshold : float, optional
        Relative threshold for peak detection (0-1) (default: 0.3)
        
    Returns
    -------
    pixel_positions : np.ndarray
        Pixel positions of detected arc lines
    wavelengths : np.ndarray
        Corresponding wavelengths from template (Angstroms)
        
    Raises
    ------
    ImportError
        If dtw-python package not installed
    ValueError
        If inputs invalid or DTW alignment fails
        
    Notes
    -----
    - Requires dtw-python package: pip install dtw-python
    - Template should match lamp type and grating setting
    - Works best when template and observation have similar spectral coverage
    - Returns peaks only (not full spectrum mapping) for use with polynomial fitting
    
    Examples
    --------
    >>> from pykosmospp.wavelength.match import load_arc_template
    >>> template_waves, template_flux = load_arc_template('Ar', '1.18-ctr', 'Blue')
    >>> arc_1d = np.median(arc_frame.data.data, axis=1)  # Collapse to 1D
    >>> pixels, wavelengths = identify_dtw(arc_1d, template_waves, template_flux)
    >>> print(f"Identified {len(pixels)} arc lines")
    """
    if not DTW_AVAILABLE:
        raise ImportError(
            "DTW wavelength calibration requires dtw-python package.\n"
            "Install with: pip install dtw-python"
        )
    
    # Validate inputs
    if len(arc_spectrum) < 100:
        raise ValueError(f"Arc spectrum too short: {len(arc_spectrum)} pixels (need ≥100)")
    
    if len(template_wavelengths) != len(template_flux):
        raise ValueError(
            f"Template wavelength and flux arrays must have same length: "
            f"{len(template_wavelengths)} vs {len(template_flux)}"
        )
    
    # Normalize spectra for DTW
    arc_norm = _normalize_spectrum(arc_spectrum)
    template_norm = _normalize_spectrum(template_flux)
    
    # Optional upsampling for better alignment
    if upsample:
        arc_pixels_orig = np.arange(len(arc_spectrum))
        arc_pixels_up = np.linspace(0, len(arc_spectrum) - 1, len(arc_spectrum) * upsample_factor)
        arc_interp = interp1d(arc_pixels_orig, arc_norm, kind='cubic', fill_value='extrapolate')
        arc_norm = arc_interp(arc_pixels_up)
        
        template_idx_orig = np.arange(len(template_flux))
        template_idx_up = np.linspace(0, len(template_flux) - 1, len(template_flux) * upsample_factor)
        template_interp = interp1d(template_idx_orig, template_norm, kind='cubic', fill_value='extrapolate')
        template_norm = template_interp(template_idx_up)
        template_waves_interp = interp1d(template_idx_orig, template_wavelengths, kind='cubic', fill_value='extrapolate')
        template_wavelengths = template_waves_interp(template_idx_up)
    
    # Perform DTW alignment
    try:
        alignment = dtw(
            arc_norm,
            template_norm,
            step_pattern=step_pattern,
            window_type=window_type,
            open_begin=open_begin,
            open_end=open_end,
            keep_internals=True
        )
    except Exception as e:
        raise ValueError(f"DTW alignment failed: {e}")
    
    # Extract alignment mapping: arc pixel -> template index
    arc_indices = alignment.index1  # Indices in arc spectrum
    template_indices = alignment.index2  # Corresponding indices in template
    
    # Create mapping function: arc pixel -> wavelength
    if upsample:
        # Map back to original pixel scale
        arc_indices_orig = arc_indices / upsample_factor
        pixel_to_wave = interp1d(
            arc_indices_orig,
            template_wavelengths[template_indices],
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
    else:
        pixel_to_wave = interp1d(
            arc_indices,
            template_wavelengths[template_indices],
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
    
    # Detect peaks in observed arc spectrum
    if peak_spline:
        # Use spline-based peak detection for sub-pixel accuracy
        pixel_positions = _find_peaks_spline(arc_spectrum, min_peak_separation, peak_threshold)
    else:
        # Use scipy peak detection
        peaks, properties = find_peaks(
            arc_spectrum,
            height=np.max(arc_spectrum) * peak_threshold,
            distance=min_peak_separation
        )
        pixel_positions = peaks.astype(float)
    
    # Map pixel positions to wavelengths using DTW alignment
    wavelengths = pixel_to_wave(pixel_positions)
    
    # Filter out invalid wavelengths (outside template range)
    valid = (wavelengths >= template_wavelengths.min()) & (wavelengths <= template_wavelengths.max())
    pixel_positions = pixel_positions[valid]
    wavelengths = wavelengths[valid]
    
    if len(pixel_positions) < 10:
        raise ValueError(
            f"Too few arc lines identified: {len(pixel_positions)} (need ≥10).\n"
            f"Try adjusting peak_threshold or check template selection."
        )
    
    return pixel_positions, wavelengths


def _normalize_spectrum(flux: np.ndarray) -> np.ndarray:
    """
    Normalize spectrum for DTW alignment.
    
    Normalization: (flux - median) / max(abs(flux - median))
    
    This removes DC offset and scales amplitude to [-1, 1] range,
    making DTW distance metric more robust.
    
    Parameters
    ----------
    flux : np.ndarray
        Input flux array
        
    Returns
    -------
    np.ndarray
        Normalized flux
    """
    flux_centered = flux - np.median(flux)
    flux_max = np.max(np.abs(flux_centered))
    
    if flux_max == 0:
        return flux_centered
    
    return flux_centered / flux_max


def _find_peaks_spline(spectrum: np.ndarray, 
                       min_separation: int = 5,
                       threshold: float = 0.3) -> np.ndarray:
    """
    Find peaks using spline interpolation for sub-pixel accuracy.
    
    Algorithm:
    1. Detect integer pixel peaks
    2. Fit cubic spline around each peak
    3. Find spline maximum for sub-pixel position
    
    Parameters
    ----------
    spectrum : np.ndarray
        Input spectrum
    min_separation : int
        Minimum peak separation in pixels
    threshold : float
        Relative height threshold (0-1)
        
    Returns
    -------
    np.ndarray
        Sub-pixel peak positions
    """
    from scipy.interpolate import UnivariateSpline
    
    # First find integer peaks
    peaks, _ = find_peaks(
        spectrum,
        height=np.max(spectrum) * threshold,
        distance=min_separation
    )
    
    # Refine each peak with spline
    refined_peaks = []
    
    for peak in peaks:
        # Get local region around peak (±3 pixels)
        left = max(0, peak - 3)
        right = min(len(spectrum), peak + 4)
        
        if right - left < 4:
            # Not enough points for spline, use integer position
            refined_peaks.append(float(peak))
            continue
        
        x_local = np.arange(left, right)
        y_local = spectrum[left:right]
        
        try:
            # Fit cubic spline
            spline = UnivariateSpline(x_local, y_local, k=3, s=0)
            
            # Find maximum in range
            x_fine = np.linspace(left, right - 1, 100)
            y_fine = spline(x_fine)
            max_idx = np.argmax(y_fine)
            peak_refined = x_fine[max_idx]
            
            refined_peaks.append(peak_refined)
        except:
            # Spline fit failed, use integer position
            refined_peaks.append(float(peak))
    
    return np.array(refined_peaks)
