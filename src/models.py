"""
Data models for spectroscopic reduction pipeline.

Per data-model.md: RawFrame hierarchy, calibration classes,
observation sets, wavelength solutions, extracted spectra.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
import astropy.units as u

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple
import numpy as np
from astropy.nddata import CCDData
from astropy.io import fits
import astropy.units as u


# =============================================================================
# Raw Frame Hierarchy (Abstract Base + Subclasses)
# =============================================================================

class RawFrame(ABC):
    """
    Abstract base class for raw FITS files from telescope.
    
    Per data-model.md §1: Represents single FITS file with metadata
    (observation type, target, exposure time, instrument configuration).
    """
    
    def __init__(self, file_path: Path):
        """
        Initialize RawFrame from FITS file.
        
        Parameters
        ----------
        file_path : Path
            Path to FITS file
        """
        self.file_path = Path(file_path)
        self.data: Optional[CCDData] = None
        self.header: Optional[fits.Header] = None
        self.observation_date: Optional[datetime] = None
        self.exposure_time: float = 0.0
        self.gain: float = 1.4  # e-/ADU (KOSMOS default)
        self.readnoise: float = 3.7  # e- (KOSMOS default)
        self.saturate: float = 58982.0  # ADU (KOSMOS default)
        
    @classmethod
    def from_fits(cls, file_path: Path, gain: float = 1.4, 
                  readnoise: float = 3.7, saturate: float = 58982.0):
        """
        Load RawFrame from FITS file with detector parameters.
        
        Parameters
        ----------
        file_path : Path
            Path to FITS file
        gain : float
            CCD gain in e-/ADU
        readnoise : float
            Read noise in e-
        saturate : float
            Saturation level in ADU
            
        Returns
        -------
        RawFrame subclass instance
        """
        frame = cls(file_path)
        frame.gain = gain
        frame.readnoise = readnoise
        frame.saturate = saturate
        
        # Load FITS data as CCDData
        frame.data = CCDData.read(file_path, unit='adu')
        frame.header = frame.data.header
        
        # Extract metadata
        frame.validate_header()
        frame.exposure_time = frame.header.get('EXPTIME', 0.0)
        
        # Parse observation date
        date_obs = frame.header.get('DATE-OBS', None)
        if date_obs:
            frame.observation_date = datetime.fromisoformat(date_obs)
            
        return frame
    
    def validate_header(self) -> bool:
        """
        Validate required FITS header keywords are present.
        
        Returns
        -------
        bool
            True if header valid
            
        Raises
        ------
        ValueError
            If required keywords missing
        """
        required = ['IMAGETYP', 'EXPTIME']
        missing = [kw for kw in required if kw not in self.header]
        
        if missing:
            raise ValueError(f"Missing required header keywords: {missing}")
            
        return True
    
    def detect_saturation(self) -> tuple[bool, float]:
        """
        Detect saturated pixels in frame.
        
        Returns
        -------
        saturated : bool
            True if any pixels saturated
        fraction : float
            Fraction of pixels above saturation threshold
        """
        if self.data is None:
            return False, 0.0
            
        saturated_mask = self.data.data >= self.saturate
        fraction = np.sum(saturated_mask) / saturated_mask.size
        
        return fraction > 0, fraction


class BiasFrame(RawFrame):
    """
    Bias calibration frame.
    
    Per data-model.md §2: Captures detector readout bias pattern.
    """
    
    def __init__(self, file_path: Path):
        super().__init__(file_path)
        self.bias_level: Optional[float] = None
        
    def validate_header(self) -> bool:
        """Validate bias frame has zero exposure time."""
        super().validate_header()
        
        image_type = self.header.get('IMAGETYP', '').lower()
        # Accept various bias frame identifiers
        valid_types = ['bias', 'zero', 'bias frame', 'zero frame']
        if not any(vtype in image_type for vtype in valid_types):
            # If IMAGETYP is missing or doesn't match, check OBJECT field
            object_name = self.header.get('OBJECT', '').lower()
            if not any(vtype in object_name for vtype in valid_types):
                raise ValueError(
                    f"Frame {self.file_path} is not a bias frame. "
                    f"IMAGETYP='{self.header.get('IMAGETYP')}', OBJECT='{self.header.get('OBJECT')}'"
                )
            
        # Bias frames should have zero or very short exposure
        if self.exposure_time > 0.1:
            raise ValueError(f"Bias frame has non-zero exposure time: {self.exposure_time}s")
            
        return True


class FlatFrame(RawFrame):
    """
    Flat field calibration frame.
    
    Per data-model.md §3: Captures pixel-to-pixel sensitivity and illumination.
    """
    
    def __init__(self, file_path: Path):
        super().__init__(file_path)
        self.lamp_type: Optional[str] = None
        self.saturation_fraction: float = 0.0
        
    def validate_header(self) -> bool:
        """Validate flat frame header."""
        super().validate_header()
        
        image_type = self.header.get('IMAGETYP', '').lower()
        # Accept various flat frame identifiers
        valid_types = ['flat', 'flat field', 'flatfield', 'dome flat', 'sky flat']
        if not any(vtype in image_type for vtype in valid_types):
            # If IMAGETYP is missing or doesn't match, check OBJECT field
            object_name = self.header.get('OBJECT', '').lower()
            if not any(vtype in object_name for vtype in valid_types):
                raise ValueError(
                    f"Frame {self.file_path} is not a flat frame. "
                    f"IMAGETYP='{self.header.get('IMAGETYP')}', OBJECT='{self.header.get('OBJECT')}'"
                )
            
        # Extract lamp type if available
        self.lamp_type = self.header.get('LAMPTYPE', 'unknown')
        
        return True


class ArcFrame(RawFrame):
    """
    Arc lamp calibration frame for wavelength calibration.
    
    Per data-model.md §4: Contains emission line spectrum.
    """
    
    def __init__(self, file_path: Path):
        super().__init__(file_path)
        self.lamp_type: Optional[str] = None
        self.linelist_file: Optional[Path] = None
        
    def validate_header(self) -> bool:
        """Validate arc frame header and detect lamp type."""
        super().validate_header()
        
        image_type = self.header.get('IMAGETYP', '').lower()
        # Accept various arc frame identifiers
        valid_types = ['arc', 'comp', 'comparison', 'arc lamp', 'wavelength', 
                      'henear', 'argon', 'krypton', 'thar', 'cuar', 'xenon']
        if not any(vtype in image_type for vtype in valid_types):
            # If IMAGETYP is missing or doesn't match, check OBJECT field
            object_name = self.header.get('OBJECT', '').lower()
            if not any(vtype in object_name for vtype in valid_types):
                raise ValueError(
                    f"Frame {self.file_path} is not an arc frame. "
                    f"IMAGETYP='{self.header.get('IMAGETYP')}', OBJECT='{self.header.get('OBJECT')}'"
                )
            
        # Detect arc lamp type from filename or header (per research.md §8)
        self._detect_lamp_type_from_filename()
        
        if self.lamp_type is None:
            self.lamp_type = self.header.get('LAMPTYPE', 'henear')
            
        return True
    
    def _detect_lamp_type_from_filename(self):
        """
        Detect arc lamp type from filename patterns.
        
        Per research.md §8: Filename-based arc lamp identification
        """
        filename_lower = self.file_path.name.lower()
        
        lamp_patterns = {
            'henear': ['henear', 'he-ne-ar', 'hene'],
            'apohenear': ['apohenear', 'apo'],
            'henearhres': ['henearhres', 'hires', 'highres'],
            'argon': ['argon', 'ar'],
            'krypton': ['krypton', 'kr'],
            'thar': ['thar', 'th-ar'],
            'cuar': ['cuar', 'cu-ar'],
            'xenon': ['xenon', 'xe'],
            'fear': ['fear', 'fe-ar'],
        }
        
        for lamp_type, patterns in lamp_patterns.items():
            if any(pattern in filename_lower for pattern in patterns):
                self.lamp_type = lamp_type
                break


class ScienceFrame(RawFrame):
    """
    Science observation frame (target spectrum).
    
    Per data-model.md §5: 2D spectral image of astronomical target.
    """
    
    def __init__(self, file_path: Path):
        super().__init__(file_path)
        self.target_name: Optional[str] = None
        self.ra: Optional[float] = None  # degrees
        self.dec: Optional[float] = None  # degrees
        self.airmass: Optional[float] = None
        self.nod_position: Optional[str] = None  # 'A' or 'B' for nod pairs
        
    def validate_header(self) -> bool:
        """Validate science frame header and extract metadata."""
        super().validate_header()
        
        image_type = self.header.get('IMAGETYP', '').lower()
        # Accept various science frame identifiers
        valid_types = ['object', 'science', 'target', 'light']
        if not any(vtype in image_type for vtype in valid_types):
            # If IMAGETYP doesn't match, be lenient - if it's not bias/flat/arc, assume science
            # This handles observatories with non-standard IMAGETYP values
            if not any(exclude in image_type for exclude in ['bias', 'zero', 'flat', 'arc', 'comp', 'dark']):
                # Assume it's a science frame if it doesn't match calibration types
                pass
            else:
                raise ValueError(
                    f"Frame {self.file_path} is not a science frame. "
                    f"IMAGETYP='{self.header.get('IMAGETYP')}'"
                )
            
        # Extract target metadata
        self.target_name = self.header.get('OBJECT', 'unknown')
        self.ra = self.header.get('RA', None)
        self.dec = self.header.get('DEC', None)
        self.airmass = self.header.get('AIRMASS', None)
        self.nod_position = self.header.get('NODPOS', None)
        
        return True


# =============================================================================
# Calibration Products
# =============================================================================

@dataclass
class MasterBias:
    """
    Combined master bias frame.
    
    Per data-model.md §7: Median-combined bias with provenance.
    """
    data: CCDData
    n_combined: int
    bias_level: float  # ADU
    bias_stdev: float  # ADU
    provenance: Dict[str, any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """
        Validate master bias quality.
        
        Per data-model.md §7: bias_stdev <10 ADU
        """
        if self.bias_stdev > 10.0:
            raise ValueError(f"Bias stdev too high: {self.bias_stdev:.2f} ADU (limit: 10 ADU)")
        return True


@dataclass
class MasterFlat:
    """
    Combined master flat field frame.
    
    Per data-model.md §8: Median-combined flat normalized to unity.
    """
    data: CCDData
    n_combined: int
    normalization_region: tuple  # (spatial_start, spatial_end, spectral_start, spectral_end)
    bad_pixel_fraction: float
    provenance: Dict[str, any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """
        Validate master flat quality.
        
        Per data-model.md §8: bad_pixel_fraction <0.05
        """
        if self.bad_pixel_fraction > 0.05:
            raise ValueError(f"Bad pixel fraction too high: {self.bad_pixel_fraction:.3f} (limit: 0.05)")
        return True


@dataclass
class CalibrationSet:
    """
    Complete set of calibrations for science frame reduction.
    
    Per data-model.md §6: Master bias, flat, bad pixel mask.
    """
    master_bias: MasterBias
    master_flat: MasterFlat
    bad_pixel_mask: Optional[np.ndarray] = None
    
    def apply_to_frame(self, science_frame: ScienceFrame, 
                       propagate_uncertainty: bool = True) -> CCDData:
        """
        Apply calibrations to science frame with proper uncertainty propagation.
        
        Per T112: Propagates read noise + Poisson noise through bias subtraction
        and flat fielding per FR-014.
        
        Uncertainty Propagation:
        1. Raw frame: σ² = (readnoise)² + (data × gain) [Poisson + read noise]
        2. Bias subtraction: σ²_calib = σ²_science + σ²_bias
        3. Flat fielding: σ²_final = σ²_calib / flat² + (calib × σ_flat / flat²)²
        
        Parameters
        ----------
        science_frame : ScienceFrame
            Raw science frame
        propagate_uncertainty : bool, optional
            Whether to compute and propagate uncertainties (default: True)
            
        Returns
        -------
        CCDData
            Calibrated science data with uncertainty
        """
        from astropy.nddata import StdDevUncertainty
        
        # Compute initial uncertainty for science frame if not present
        if propagate_uncertainty and science_frame.data.uncertainty is None:
            # σ² = readnoise² + (data × gain) [Poisson variance in e-]
            # Convert to ADU: σ = sqrt(readnoise² + data × gain) / gain
            variance = (science_frame.readnoise**2 + 
                       np.maximum(science_frame.data.data, 0) * science_frame.gain) / science_frame.gain**2
            science_frame.data.uncertainty = StdDevUncertainty(np.sqrt(variance))
        
        # Compute uncertainty for master bias if not present
        if propagate_uncertainty and self.master_bias.data.uncertainty is None:
            # Bias uncertainty from read noise (no Poisson component)
            # Reduced by sqrt(N) for N combined frames
            n_bias = len(self.master_bias.source_frames)
            bias_variance = (science_frame.readnoise**2 / n_bias) / science_frame.gain**2
            self.master_bias.data.uncertainty = StdDevUncertainty(
                np.full_like(self.master_bias.data.data, np.sqrt(bias_variance))
            )
        
        # Bias subtraction (astropy propagates uncertainties automatically)
        calibrated = science_frame.data.subtract(self.master_bias.data)
        
        # Compute uncertainty for master flat if not present
        if propagate_uncertainty and self.master_flat.data.uncertainty is None:
            # Flat uncertainty from Poisson statistics
            # Flats are typically high S/N, so Poisson dominates
            n_flat = len(self.master_flat.source_frames)
            flat_variance = (self.master_flat.data.data * science_frame.gain) / (science_frame.gain**2 * n_flat)
            self.master_flat.data.uncertainty = StdDevUncertainty(
                np.sqrt(np.maximum(flat_variance, 0))
            )
        
        # Flat fielding (astropy propagates uncertainties automatically)
        # σ²_final = σ²_calib / flat² + (calib × σ_flat / flat²)²
        calibrated = calibrated.divide(self.master_flat.data)
        
        # Apply bad pixel mask if available
        if self.bad_pixel_mask is not None:
            calibrated.mask = self.bad_pixel_mask
            
        return calibrated
    
    def validate(self) -> bool:
        """Validate all calibration components."""
        self.master_bias.validate()
        self.master_flat.validate()
        return True


# =============================================================================
# Placeholder classes (will be implemented in later phases)
# =============================================================================

class Spectrum2D:
    """
    2D spectrum with calibration-applied data and detected traces.
    
    Per data-model.md §9: Contains calibrated 2D data (bias-subtracted,
    flat-fielded), variance map, mask, cosmic ray flags, and list of
    detected/selected traces.
    """
    
    def __init__(self, data: np.ndarray, variance: np.ndarray,
                 source_frame: 'ScienceFrame',
                 mask: Optional[np.ndarray] = None,
                 cosmic_ray_mask: Optional[np.ndarray] = None):
        """
        Initialize 2D spectrum.
        
        Parameters
        ----------
        data : np.ndarray
            Calibrated 2D data (spatial x spectral)
        variance : np.ndarray
            Variance map (same shape as data)
        source_frame : ScienceFrame
            Source science frame
        mask : np.ndarray, optional
            Bad pixel mask (True = bad)
        cosmic_ray_mask : np.ndarray, optional
            Cosmic ray mask (True = cosmic ray)
        """
        self.data = data
        self.variance = variance
        self.source_frame = source_frame
        self.mask = mask if mask is not None else np.zeros_like(data, dtype=bool)
        self.cosmic_ray_mask = cosmic_ray_mask if cosmic_ray_mask is not None else np.zeros_like(data, dtype=bool)
        self.traces: List['Trace'] = []
        
    def detect_traces(self, min_snr: float = 3.0, **kwargs) -> List['Trace']:
        """
        Detect spectral traces using cross-correlation.
        
        Parameters
        ----------
        min_snr : float
            Minimum SNR for trace detection
        **kwargs
            Additional arguments for trace detection
            
        Returns
        -------
        List[Trace]
            Detected traces
        """
        from .extraction.trace import detect_traces_cross_correlation
        
        self.traces = detect_traces_cross_correlation(
            self.data,
            self.variance,
            min_snr=min_snr,
            mask=self.mask | self.cosmic_ray_mask,
            **kwargs
        )
        return self.traces
    
    def subtract_sky(self, sky_buffer: int = 30) -> np.ndarray:
        """
        Estimate and subtract sky background.
        
        Parameters
        ----------
        sky_buffer : int
            Buffer pixels from trace edges
            
        Returns
        -------
        np.ndarray
            Sky-subtracted 2D data
        """
        from .extraction.sky import estimate_sky_background
        
        sky = estimate_sky_background(
            self.data,
            self.traces,
            sky_buffer=sky_buffer,
            mask=self.mask | self.cosmic_ray_mask
        )
        
        # Subtract sky from data
        self.data = self.data - sky
        return self.data
    
    def extract_spectrum(self, trace: 'Trace', method: str = 'optimal') -> 'Spectrum1D':
        """
        Extract 1D spectrum from trace.
        
        Parameters
        ----------
        trace : Trace
            Trace to extract
        method : str
            Extraction method ('optimal' or 'aperture')
            
        Returns
        -------
        Spectrum1D
            Extracted 1D spectrum
        """
        if method == 'optimal':
            from .extraction.extract import extract_optimal
            return extract_optimal(self.data, self.variance, trace)
        else:
            raise ValueError(f"Unknown extraction method: {method}")


class Trace:
    """
    Spectral trace with position, profile, and wavelength solution.
    
    Per data-model.md §10: Spatial position as function of spectral pixel,
    fitted spatial profile, wavelength solution, SNR estimate.
    """
    
    def __init__(self, trace_id: int,
                 spatial_positions: np.ndarray,
                 spectral_pixels: np.ndarray,
                 snr_estimate: float,
                 spatial_profile: Optional['SpatialProfile'] = None,
                 wavelength_solution: Optional['WavelengthSolution'] = None,
                 user_selected: bool = False):
        """
        Initialize trace.
        
        Parameters
        ----------
        trace_id : int
            Unique trace identifier
        spatial_positions : np.ndarray
            Spatial (Y) position at each spectral pixel
        spectral_pixels : np.ndarray
            Spectral (X) pixel array
        snr_estimate : float
            Estimated median SNR
        spatial_profile : SpatialProfile, optional
            Fitted spatial profile
        wavelength_solution : WavelengthSolution, optional
            Wavelength calibration
        user_selected : bool
            Whether user manually selected this trace
        """
        self.trace_id = trace_id
        self.spatial_positions = spatial_positions
        self.spectral_pixels = spectral_pixels
        self.snr_estimate = snr_estimate
        self.spatial_profile = spatial_profile
        self.wavelength_solution = wavelength_solution
        self.user_selected = user_selected
        
    def fit_profile(self, data_2d: np.ndarray, variance_2d: np.ndarray,
                   aperture_width: int = 10) -> 'SpatialProfile':
        """
        Fit spatial profile to trace.
        
        Parameters
        ----------
        data_2d : np.ndarray
            2D spectral data
        variance_2d : np.ndarray
            2D variance map
        aperture_width : int
            Width for profile extraction
            
        Returns
        -------
        SpatialProfile
            Fitted profile
        """
        from .extraction.profile import fit_spatial_profile
        
        self.spatial_profile = fit_spatial_profile(
            data_2d,
            variance_2d,
            self,
            aperture_width=aperture_width
        )
        return self.spatial_profile
    
    def apply_wavelength_solution(self, wavelength_solution: 'WavelengthSolution'):
        """
        Apply wavelength calibration to this trace.
        
        Parameters
        ----------
        wavelength_solution : WavelengthSolution
            Wavelength solution to apply
        """
        self.wavelength_solution = wavelength_solution
    
    def extract_optimal(self, data_2d: np.ndarray, variance_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract optimal 1D spectrum from trace.
        
        Parameters
        ----------
        data_2d : np.ndarray
            2D spectral data
        variance_2d : np.ndarray
            2D variance map
            
        Returns
        -------
        flux : np.ndarray
            Extracted flux
        variance : np.ndarray
            Extracted variance
        """
        from .extraction.extract import extract_optimal
        return extract_optimal(data_2d, variance_2d, self)


class SpatialProfile:
    """
    Fitted spatial profile (cross-dispersion direction).
    
    Per data-model.md §11: Profile type (Gaussian, Moffat, empirical),
    parameters (center, width, amplitude), fit quality (chi-squared).
    """
    
    def __init__(self, profile_type: str,
                 center: float,
                 width: float,
                 amplitude: float,
                 profile_function: Callable[[np.ndarray], np.ndarray],
                 chi_squared: float):
        """
        Initialize spatial profile.
        
        Parameters
        ----------
        profile_type : str
            Profile type ('gaussian', 'moffat', 'empirical')
        center : float
            Profile center position (pixels)
        width : float
            Profile width (FWHM in pixels)
        amplitude : float
            Profile amplitude (peak value)
        profile_function : callable
            Function to evaluate profile at positions
        chi_squared : float
            Chi-squared of fit
        """
        self.profile_type = profile_type
        self.center = center
        self.width = width
        self.amplitude = amplitude
        self.profile_function = profile_function
        self.chi_squared = chi_squared
        
    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        """
        Evaluate profile at given positions.
        
        Parameters
        ----------
        positions : np.ndarray
            Spatial positions
            
        Returns
        -------
        np.ndarray
            Profile values
        """
        return self.profile_function(positions)


class WavelengthSolution:
    """
    Wavelength calibration solution mapping pixel to wavelength.
    
    Per data-model.md §12: Polynomial coefficients, arc line identifications,
    RMS residual, wavelength range.
    """
    
    def __init__(self, coefficients: np.ndarray, order: int, arc_frame: 'ArcFrame',
                 n_lines_identified: int, rms_residual: float, 
                 wavelength_range: tuple, poly_type: str = 'chebyshev',
                 pixel_range: tuple = None,
                 calibration_method: str = 'line_matching',
                 template_used: str = None,
                 dtw_parameters: dict = None):
        """
        Initialize wavelength solution.
        
        Parameters
        ----------
        coefficients : np.ndarray
            Polynomial coefficients
        order : int
            Polynomial order
        arc_frame : ArcFrame
            Source arc frame
        n_lines_identified : int
            Number of arc lines identified
        rms_residual : float
            RMS residual of fit in Angstroms
        wavelength_range : tuple
            (min_wavelength, max_wavelength) in Angstroms
        poly_type : str
            Polynomial type ('chebyshev', 'legendre', 'polynomial')
        pixel_range : tuple, optional
            (min_pixel, max_pixel) used for normalization. If None, uses (0, 4095)
        calibration_method : str
            Method used for calibration: 'line_matching' or 'dtw'
        template_used : str, optional
            Name of arc template file used (for DTW method)
        dtw_parameters : dict, optional
            DTW parameters used (e.g., peak_threshold, step_pattern)
        """
        self.coefficients = coefficients
        self.order = order
        self.arc_frame = arc_frame
        self.n_lines_identified = n_lines_identified
        self.rms_residual = rms_residual
        self.wavelength_range = wavelength_range
        self.poly_type = poly_type
        self.pixel_range = pixel_range if pixel_range is not None else (0, 4095)
        
        # Provenance tracking (Constitution Principle III)
        self.calibration_method = calibration_method
        self.template_used = template_used
        self.dtw_parameters = dtw_parameters or {}
        
        # Timestamp
        from datetime import datetime
        self.timestamp = datetime.utcnow().isoformat()
        
    def wavelength(self, pixels: np.ndarray) -> np.ndarray:
        """
        Evaluate wavelength at pixel positions.
        
        Parameters
        ----------
        pixels : np.ndarray
            Pixel positions
            
        Returns
        -------
        np.ndarray
            Wavelengths in Angstroms
        """
        if self.poly_type == 'chebyshev':
            from numpy.polynomial import chebyshev
            # Normalize pixels to [-1, 1] using same range as fitting
            pix_min, pix_max = self.pixel_range
            pix_norm = 2.0 * (pixels - pix_min) / (pix_max - pix_min) - 1.0
            return chebyshev.chebval(pix_norm, self.coefficients)
        else:
            # Standard polynomial
            return np.polyval(self.coefficients, pixels)
    
    def inverse(self, wavelengths: np.ndarray) -> np.ndarray:
        """
        Approximate inverse: wavelength to pixel (via interpolation).
        
        Parameters
        ----------
        wavelengths : np.ndarray
            Wavelengths in Angstroms
            
        Returns
        -------
        np.ndarray
            Pixel positions
        """
        # Generate dense sampling of wavelength solution
        pix_min, pix_max = self.pixel_range
        pixels = np.linspace(pix_min, pix_max, 1000)
        waves = self.wavelength(pixels)
        
        # Interpolate inverse
        return np.interp(wavelengths, waves, pixels)
    
    def validate(self) -> bool:
        """
        Validate wavelength solution quality.
        
        Returns
        -------
        bool
            True if valid
            
        Raises
        ------
        ValueError
            If RMS too high or too few lines
        """
        if self.rms_residual > 0.2:  # Acceptance criterion per SC-003
            raise ValueError(f"Wavelength RMS too high: {self.rms_residual:.3f} Å (limit: 0.2 Å)")
        if self.n_lines_identified < 10:
            raise ValueError(f"Too few arc lines: {self.n_lines_identified} (need ≥10)")
        return True


class Spectrum1D:
    """Placeholder for 1D spectrum class (data-model.md §13)"""
    pass


class QualityMetrics:
    """
    Quality metrics for reduced spectra.
    
    Per data-model.md §14: SNR, wavelength RMS, sky residuals,
    cosmic ray fraction, overall quality grade.
    """
    
    def __init__(self):
        """Initialize quality metrics."""
        self.median_snr: Optional[float] = None
        self.wavelength_rms: Optional[float] = None
        self.sky_residual_rms: Optional[float] = None
        self.cosmic_ray_fraction: float = 0.0
        self.saturation_flag: bool = False
        self.ab_subtraction_quality: Optional[float] = None
        self.overall_grade: str = 'Unknown'
        
    def compute(self, spectrum_1d, spectrum_2d: Optional['Spectrum2D'] = None):
        """
        Compute all quality metrics.
        
        Parameters
        ----------
        spectrum_1d : Spectrum1D
            Extracted 1D spectrum
        spectrum_2d : Spectrum2D, optional
            Source 2D spectrum
        """
        from .quality.metrics import compute_quality_metrics
        metrics = compute_quality_metrics(spectrum_1d, spectrum_2d)
        
        self.median_snr = metrics['median_snr']
        self.wavelength_rms = metrics.get('wavelength_rms')
        self.sky_residual_rms = metrics.get('sky_residual_rms')
        self.cosmic_ray_fraction = metrics.get('cosmic_ray_fraction', 0.0)
        self.saturation_flag = metrics.get('saturation_flag', False)
        self.overall_grade = metrics['overall_grade']
        
    def generate_report(self) -> str:
        """
        Generate formatted quality report.
        
        Returns
        -------
        str
            Formatted report string
        """
        report = []
        report.append("Quality Assessment Report")
        report.append("=" * 50)
        report.append(f"Overall Grade: {self.overall_grade}")
        report.append("")
        report.append("Metrics:")
        
        if self.median_snr is not None:
            report.append(f"  Median SNR: {self.median_snr:.1f}")
        if self.wavelength_rms is not None:
            report.append(f"  Wavelength RMS: {self.wavelength_rms:.3f} Å")
        if self.sky_residual_rms is not None:
            report.append(f"  Sky Residual RMS: {self.sky_residual_rms:.1f} e-")
        
        report.append(f"  Cosmic Ray Fraction: {self.cosmic_ray_fraction:.3f}")
        report.append(f"  Saturation: {'Yes' if self.saturation_flag else 'No'}")
        
        return "\n".join(report)


class PipelineConfig:
    """Placeholder for pipeline config class (data-model.md §15)"""
    pass


@dataclass
class ObservationSet:
    """
    Collection of frames for a single observation sequence.
    
    Per data-model.md §16: Groups bias, flat, arc, and science frames
    with methods for validation and AB pair grouping.
    """
    observation_date: datetime
    target_name: str
    bias_frames: List[BiasFrame] = field(default_factory=list)
    flat_frames: List[FlatFrame] = field(default_factory=list)
    arc_frames: List[ArcFrame] = field(default_factory=list)
    science_frames: List[ScienceFrame] = field(default_factory=list)
    calibration_set: Optional[CalibrationSet] = None
    
    @classmethod
    def from_directory(cls, input_dir: Path, config: dict) -> 'ObservationSet':
        """
        Create ObservationSet by discovering FITS files in directory.
        
        Parameters
        ----------
        input_dir : Path
            Directory with arcs/, flats/, biases/, science/ subdirectories
        config : dict
            Pipeline configuration
            
        Returns
        -------
        ObservationSet
            Populated observation set
        """
        from .io.organize import discover_fits_files
        
        files_by_type = discover_fits_files(input_dir)
        
        # Load frames
        gain = config.get('detector', {}).get('gain', 1.4)
        readnoise = config.get('detector', {}).get('readnoise', 3.7)
        saturate = config.get('detector', {}).get('saturate', 58982)
        
        bias_frames = [BiasFrame.from_fits(f, gain, readnoise, saturate) 
                      for f in files_by_type['bias']]
        flat_frames = [FlatFrame.from_fits(f, gain, readnoise, saturate)
                      for f in files_by_type['flat']]
        arc_frames = [ArcFrame.from_fits(f, gain, readnoise, saturate)
                     for f in files_by_type['arc']]
        science_frames = [ScienceFrame.from_fits(f, gain, readnoise, saturate)
                         for f in files_by_type['science']]
        
        # Extract observation metadata
        obs_date = science_frames[0].observation_date if science_frames else datetime.now()
        target = science_frames[0].target_name if science_frames else 'unknown'
        
        return cls(
            observation_date=obs_date,
            target_name=target,
            bias_frames=bias_frames,
            flat_frames=flat_frames,
            arc_frames=arc_frames,
            science_frames=science_frames
        )
    
    def group_ab_pairs(self, max_time_diff: float = 600.0) -> List[Tuple[ScienceFrame, ScienceFrame]]:
        """
        Group science frames into AB nod pairs.
        
        Per data-model.md §16: Matches by nod_position='A'/'B' or by
        observation time proximity (<10 minutes).
        
        Parameters
        ----------
        max_time_diff : float, optional
            Maximum time difference in seconds (default: 600 = 10 minutes)
            
        Returns
        -------
        list of tuples
            List of (A_frame, B_frame) pairs
        """
        pairs = []
        
        # First try matching by nod_position header
        a_frames = [f for f in self.science_frames if f.nod_position == 'A']
        b_frames = [f for f in self.science_frames if f.nod_position == 'B']
        
        if a_frames and b_frames:
            # Match A and B frames by timestamp proximity
            for a_frame in a_frames:
                for b_frame in b_frames:
                    if a_frame.observation_date and b_frame.observation_date:
                        time_diff = abs((b_frame.observation_date - 
                                       a_frame.observation_date).total_seconds())
                        if time_diff < max_time_diff:
                            pairs.append((a_frame, b_frame))
                            b_frames.remove(b_frame)
                            break
        
        return pairs
    
    def validate_completeness(self) -> bool:
        """
        Validate observation set has required calibrations.
        
        Returns
        -------
        bool
            True if complete
            
        Raises
        ------
        ValueError
            If required frames missing
        """
        if len(self.bias_frames) < 3:
            raise ValueError(f"Insufficient bias frames: {len(self.bias_frames)} (need ≥3)")
        if len(self.flat_frames) < 2:
            raise ValueError(f"Insufficient flat frames: {len(self.flat_frames)} (need ≥2)")
        if len(self.arc_frames) < 1:
            raise ValueError(f"No arc frames found (need ≥1)")
        if len(self.science_frames) < 1:
            raise ValueError(f"No science frames found")
            
        return True


@dataclass
class ReducedData:
    """
    Container for fully reduced data products.
    
    Per data-model.md §17: Contains source frame, 2D spectrum,
    extracted 1D spectra, diagnostic plots, processing log.
    """
    source_frame: ScienceFrame
    spectrum_2d: Spectrum2D
    spectra_1d: List = field(default_factory=list)  # List[Spectrum1D]
    diagnostic_plots: Dict[str, Path] = field(default_factory=dict)
    processing_log: List[str] = field(default_factory=list)
    reduction_timestamp: datetime = field(default_factory=datetime.now)
    quality_metrics: Optional[QualityMetrics] = None
    
    def save_to_disk(self, output_dir: Path):
        """
        Save all reduced data products to disk.
        
        Parameters
        ----------
        output_dir : Path
            Output directory
        """
        from .io.fits import write_fits_with_provenance
        from specutils import Spectrum1D
        from astropy.nddata import CCDData
        import astropy.units as u
        
        # Create subdirectories
        (output_dir / 'reduced_2d').mkdir(exist_ok=True, parents=True)
        (output_dir / 'spectra_1d').mkdir(exist_ok=True, parents=True)
        (output_dir / 'quality_reports').mkdir(exist_ok=True, parents=True)
        
        # Save 2D spectrum
        base_name = self.source_frame.file_path.stem
        spec2d_path = output_dir / 'reduced_2d' / f"{base_name}_2d.fits"
        
        ccd_2d = CCDData(self.spectrum_2d.data, unit=u.electron)
        ccd_2d.uncertainty = self.spectrum_2d.variance
        write_fits_with_provenance(
            ccd_2d,
            spec2d_path,
            {'OBJECT': self.source_frame.header.get('OBJECT', 'Unknown')}
        )
        
        # Save 1D spectra
        for i, spec1d in enumerate(self.spectra_1d):
            spec1d_path = output_dir / 'spectra_1d' / f"{base_name}_trace{i}_1d.fits"
            spec1d.write(spec1d_path, format='wcs1d-fits', overwrite=True)
        
        # Save quality report
        if self.quality_metrics is not None:
            report_path = output_dir / 'quality_reports' / f"{base_name}_quality.txt"
            with open(report_path, 'w') as f:
                f.write(self.quality_metrics.generate_report())
    
    def generate_summary_report(self) -> str:
        """
        Generate summary report of reduction.
        
        Returns
        -------
        str
            Summary report
        """
        report = []
        report.append("Reduction Summary")
        report.append("=" * 60)
        report.append(f"Source: {self.source_frame.file_path.name}")
        report.append(f"Timestamp: {self.reduction_timestamp}")
        report.append(f"Number of extracted spectra: {len(self.spectra_1d)}")
        report.append("")
        
        if self.quality_metrics:
            report.append(self.quality_metrics.generate_report())
        
        report.append("")
        report.append("Processing Steps:")
        for log_entry in self.processing_log:
            report.append(f"  - {log_entry}")
        
        return "\n".join(report)


class InteractiveSelection:
    """Placeholder for interactive selection class (data-model.md §18)"""
    pass


class ProcessingLog:
    """Placeholder for processing log class (data-model.md §19)"""
    pass
