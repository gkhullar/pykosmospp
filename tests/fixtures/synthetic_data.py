"""
Synthetic FITS file generation for testing.

Per T113: Generates synthetic KOSMOS FITS files with realistic properties
for testing the reduction pipeline without requiring real APO data.

Supports:
- Bias frames (flat pedestal + read noise)
- Flat frames (illumination pattern + Poisson noise)
- Arc frames (HeNeAr emission lines + background)
- Science frames (1-2 galaxy traces + sky background + cosmic rays)

All frames include proper FITS headers matching KOSMOS instrument.
"""

from pathlib import Path
from typing import Optional, List, Tuple, Dict
import numpy as np
from astropy.io import fits
from astropy.time import Time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# KOSMOS detector parameters (from research.md and config)
KOSMOS_SHAPE = (4096, 2148)  # (nx, ny) - spectral × spatial
KOSMOS_GAIN = 1.4  # e-/ADU
KOSMOS_READNOISE = 3.7  # e-
KOSMOS_SATURATE = 58982.0  # ADU
KOSMOS_DISPERSION = 0.93  # Å/pixel (approximate)


def create_kosmos_header(frame_type: str, exposure_time: float = 0.0,
                        object_name: str = "TEST", lamp: str = "None",
                        airmass: float = 1.0) -> fits.Header:
    """
    Create realistic KOSMOS FITS header.
    
    Parameters
    ----------
    frame_type : str
        'bias', 'flat', 'arc', or 'object'
    exposure_time : float
        Exposure time in seconds
    object_name : str
        Target object name
    lamp : str
        Lamp name for calibrations ('HeNeAr', 'Krypton', 'Flat', 'None')
    airmass : float
        Airmass for science frames
        
    Returns
    -------
    fits.Header
        FITS header with KOSMOS keywords
    """
    header = fits.Header()
    
    # Basic FITS keywords
    header['SIMPLE'] = (True, 'Standard FITS format')
    header['BITPIX'] = (-32, 'Bits per pixel')
    header['NAXIS'] = (2, 'Number of axes')
    header['NAXIS1'] = (KOSMOS_SHAPE[0], 'Spectral axis length')
    header['NAXIS2'] = (KOSMOS_SHAPE[1], 'Spatial axis length')
    header['EXTEND'] = (True, 'FITS extensions may exist')
    
    # Observation metadata
    header['OBSTYPE'] = (frame_type.upper(), 'Observation type')
    header['OBJECT'] = (object_name, 'Object name')
    header['EXPTIME'] = (exposure_time, 'Exposure time (seconds)')
    header['AIRMASS'] = (airmass, 'Airmass')
    
    # Timestamp
    now = Time.now()
    header['DATE-OBS'] = (now.isot, 'UTC observation start')
    header['MJD-OBS'] = (now.mjd, 'Modified Julian Date')
    
    # Instrument configuration
    header['TELESCOP'] = ('APO 3.5m', 'Telescope')
    header['INSTRUME'] = ('KOSMOS', 'Instrument')
    header['DETECTOR'] = ('e2v 231-C6', 'CCD detector')
    header['GAIN'] = (KOSMOS_GAIN, 'CCD gain (e-/ADU)')
    header['RDNOISE'] = (KOSMOS_READNOISE, 'Read noise (e-)')
    header['SATURATE'] = (KOSMOS_SATURATE, 'Saturation level (ADU)')
    
    # Spectroscopic configuration
    header['DISPAXIS'] = (1, 'Dispersion axis (1=horizontal)')
    header['DISPER'] = (KOSMOS_DISPERSION, 'Dispersion (Angstrom/pixel)')
    header['CENWAVE'] = (5500.0, 'Central wavelength (Angstrom)')
    header['LAMPNAME'] = (lamp, 'Calibration lamp name')
    
    # Data processing flags
    header['BIASSEC'] = ('[1:50,*]', 'Bias section (if present)')
    header['TRIMSEC'] = ('[51:4096,*]', 'Trim section')
    
    return header


def generate_bias_frame(output_path: Path, seed: Optional[int] = None) -> None:
    """
    Generate synthetic bias frame.
    
    Bias frames have:
    - Flat pedestal (~500 ADU)
    - Read noise (Gaussian, σ=readnoise/gain)
    - No Poisson component (zero photons)
    
    Parameters
    ----------
    output_path : Path
        Output FITS file path
    seed : int, optional
        Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Pedestal level (typical bias level)
    pedestal = 500.0  # ADU
    
    # Read noise (Gaussian)
    noise_adu = KOSMOS_READNOISE / KOSMOS_GAIN
    data = pedestal + np.random.normal(0, noise_adu, KOSMOS_SHAPE)
    
    # Create FITS
    header = create_kosmos_header('bias', exposure_time=0.0)
    hdu = fits.PrimaryHDU(data=data.astype(np.float32), header=header)
    hdu.writeto(output_path, overwrite=True)
    
    logger.info(f"Generated bias frame: {output_path}")


def generate_flat_frame(output_path: Path, mean_counts: float = 30000.0,
                       illumination_pattern: bool = True,
                       seed: Optional[int] = None) -> None:
    """
    Generate synthetic flat field frame.
    
    Flat frames have:
    - High signal level (~30k ADU, well below saturation)
    - Poisson noise (dominant at high counts)
    - Read noise (negligible compared to Poisson)
    - Optional illumination pattern (spatial variation)
    
    Parameters
    ----------
    output_path : Path
        Output FITS file path
    mean_counts : float
        Mean count level in ADU
    illumination_pattern : bool
        Add realistic illumination gradient
    seed : int, optional
        Random seed
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Start with uniform illumination
    data = np.full(KOSMOS_SHAPE, mean_counts, dtype=np.float32)
    
    # Add illumination pattern (vignetting, slit function)
    if illumination_pattern:
        # Spatial profile (Gaussian slit function)
        y = np.arange(KOSMOS_SHAPE[1])
        spatial_profile = np.exp(-((y - KOSMOS_SHAPE[1]/2) / (KOSMOS_SHAPE[1]/4))**2)
        data *= spatial_profile[np.newaxis, :]
        
        # Spectral gradient (wavelength-dependent efficiency)
        x = np.arange(KOSMOS_SHAPE[0])
        spectral_efficiency = 1.0 - 0.2 * (x / KOSMOS_SHAPE[0])
        data *= spectral_efficiency[:, np.newaxis]
    
    # Add Poisson noise (dominant)
    data_electrons = data * KOSMOS_GAIN
    noisy_electrons = np.random.poisson(data_electrons)
    data = noisy_electrons / KOSMOS_GAIN
    
    # Add read noise (small contribution)
    noise_adu = KOSMOS_READNOISE / KOSMOS_GAIN
    data += np.random.normal(0, noise_adu, KOSMOS_SHAPE)
    
    # Create FITS
    header = create_kosmos_header('flat', exposure_time=10.0, lamp='Flat')
    hdu = fits.PrimaryHDU(data=data.astype(np.float32), header=header)
    hdu.writeto(output_path, overwrite=True)
    
    logger.info(f"Generated flat frame: {output_path}")


def generate_arc_frame(output_path: Path, lamp: str = 'HeNeAr',
                      num_lines: int = 50, mean_background: float = 100.0,
                      seed: Optional[int] = None) -> None:
    """
    Generate synthetic arc lamp frame.
    
    Arc frames have:
    - Emission lines at known wavelengths
    - Gaussian spatial profile (matches slit)
    - Low continuum background
    - Poisson + read noise
    
    Parameters
    ----------
    output_path : Path
        Output FITS file path
    lamp : str
        Lamp type ('HeNeAr' or 'Krypton')
    num_lines : int
        Number of emission lines to generate
    mean_background : float
        Background level in ADU
    seed : int, optional
        Random seed
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Start with background
    data = np.full(KOSMOS_SHAPE, mean_background, dtype=np.float32)
    
    # Load line list for wavelengths (simplified)
    if lamp == 'HeNeAr':
        # Common HeNeAr lines in optical (Å)
        line_wavelengths = [
            4046.56, 4158.59, 4200.67, 4348.06, 4358.34, 4471.48,
            4545.05, 4678.16, 4764.87, 4879.86, 4965.08, 5015.68,
            5085.82, 5154.66, 5187.75, 5330.78, 5400.56, 5460.74,
            5558.70, 5656.66, 5769.60, 5852.49, 5881.90, 5944.83,
            6029.00, 6074.34, 6096.16, 6143.06, 6163.59, 6217.28,
            6266.50, 6304.79, 6334.43, 6382.99, 6402.25, 6506.53,
            6532.88, 6598.95, 6677.28, 6717.04, 6929.47, 7032.41,
            7067.22, 7173.94, 7245.17, 7272.94, 7383.98, 7503.87,
            7635.11, 7723.76
        ]
    else:  # Krypton
        line_wavelengths = [
            4273.97, 4318.55, 4319.58, 4376.12, 4463.69, 4502.35,
            4579.35, 5208.32, 5562.23, 5570.29, 5649.56, 6421.02,
            7224.10, 7587.41, 7601.54, 7685.25, 7694.54, 7854.82
        ]
    
    # Select subset
    selected_lines = np.random.choice(line_wavelengths, size=min(num_lines, len(line_wavelengths)), replace=False)
    
    # Convert wavelengths to pixel positions
    # Assume linear dispersion centered at 5500 Å
    central_wavelength = 5500.0
    central_pixel = KOSMOS_SHAPE[0] / 2
    
    # Add emission lines
    for wavelength in selected_lines:
        # Pixel position (with small nonlinearity)
        pixel_pos = central_pixel + (wavelength - central_wavelength) / KOSMOS_DISPERSION
        pixel_pos += 0.001 * (pixel_pos - central_pixel)**2 / KOSMOS_SHAPE[0]  # Slight distortion
        
        if 0 <= pixel_pos < KOSMOS_SHAPE[0]:
            # Line intensity (varies by line strength)
            intensity = np.random.uniform(500, 5000)  # ADU peak
            
            # Spectral width (instrument resolution)
            fwhm_spectral = 3.0  # pixels
            sigma_spectral = fwhm_spectral / 2.355
            
            # Spatial profile (slit)
            fwhm_spatial = 4.0  # pixels
            sigma_spatial = fwhm_spatial / 2.355
            
            # Generate 2D Gaussian line
            x = np.arange(KOSMOS_SHAPE[0])
            y = np.arange(KOSMOS_SHAPE[1])
            
            # Spectral Gaussian
            spectral_profile = np.exp(-((x - pixel_pos) / sigma_spectral)**2)
            
            # Spatial Gaussian (centered)
            spatial_profile = np.exp(-((y - KOSMOS_SHAPE[1]/2) / sigma_spatial)**2)
            
            # Add to data
            line_data = intensity * spectral_profile[:, np.newaxis] * spatial_profile[np.newaxis, :]
            data += line_data
    
    # Add Poisson noise
    data_electrons = data * KOSMOS_GAIN
    noisy_electrons = np.random.poisson(np.maximum(data_electrons, 0))
    data = noisy_electrons / KOSMOS_GAIN
    
    # Add read noise
    noise_adu = KOSMOS_READNOISE / KOSMOS_GAIN
    data += np.random.normal(0, noise_adu, KOSMOS_SHAPE)
    
    # Create FITS
    header = create_kosmos_header('arc', exposure_time=30.0, lamp=lamp)
    hdu = fits.PrimaryHDU(data=data.astype(np.float32), header=header)
    hdu.writeto(output_path, overwrite=True)
    
    logger.info(f"Generated arc frame: {output_path} ({lamp}, {num_lines} lines)")


def generate_science_frame(output_path: Path, num_traces: int = 1,
                          trace_snr: float = 10.0, sky_level: float = 500.0,
                          cosmic_ray_fraction: float = 0.001,
                          exposure_time: float = 600.0,
                          object_name: str = "Galaxy", airmass: float = 1.2,
                          seed: Optional[int] = None) -> None:
    """
    Generate synthetic science frame with galaxy spectrum.
    
    Science frames have:
    - 1-2 spectral traces (object continuum)
    - Sky background (continuum + emission lines)
    - Cosmic rays (random high-intensity pixels)
    - Poisson + read noise
    
    Parameters
    ----------
    output_path : Path
        Output FITS file path
    num_traces : int
        Number of spectral traces (1 or 2)
    trace_snr : float
        Target S/N ratio for trace
    sky_level : float
        Sky background level in ADU
    cosmic_ray_fraction : float
        Fraction of pixels with cosmic rays
    exposure_time : float
        Exposure time in seconds
    object_name : str
        Object name for header
    airmass : float
        Airmass
    seed : int, optional
        Random seed
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Start with sky background
    data = np.full(KOSMOS_SHAPE, sky_level, dtype=np.float32)
    
    # Add sky emission lines (OH airglow)
    sky_lines = [5577.34, 6300.30, 6363.78, 7316.28, 8399.17, 8430.17]
    for wavelength in sky_lines:
        # Convert to pixel
        central_wavelength = 5500.0
        central_pixel = KOSMOS_SHAPE[0] / 2
        pixel_pos = central_pixel + (wavelength - central_wavelength) / KOSMOS_DISPERSION
        
        if 0 <= pixel_pos < KOSMOS_SHAPE[0]:
            intensity = np.random.uniform(200, 800)  # Sky line strength
            fwhm_spectral = 3.0
            sigma_spectral = fwhm_spectral / 2.355
            
            x = np.arange(KOSMOS_SHAPE[0])
            y = np.arange(KOSMOS_SHAPE[1])
            
            spectral_profile = np.exp(-((x - pixel_pos) / sigma_spectral)**2)
            spatial_profile = np.ones(KOSMOS_SHAPE[1])  # Uniform across slit
            
            line_data = intensity * spectral_profile[:, np.newaxis] * spatial_profile[np.newaxis, :]
            data += line_data
    
    # Add spectral traces
    trace_positions = []
    if num_traces == 1:
        trace_positions = [KOSMOS_SHAPE[1] / 2]
    elif num_traces == 2:
        trace_positions = [KOSMOS_SHAPE[1] / 2 - 30, KOSMOS_SHAPE[1] / 2 + 30]
    
    for trace_y in trace_positions:
        # Galaxy continuum spectrum (simple power law + absorption features)
        x = np.arange(KOSMOS_SHAPE[0])
        wavelengths = 5500 + (x - KOSMOS_SHAPE[0]/2) * KOSMOS_DISPERSION
        
        # Continuum (decreasing toward blue)
        continuum = 1000 * (wavelengths / 5500)**(-1.5)
        
        # Add absorption lines (simplified galaxy spectrum)
        # Ca H&K, G-band, Mg b, Na D
        absorption_lines = [3933.66, 3968.47, 4304.4, 5175.4, 5894.0]
        for line_wave in absorption_lines:
            absorption_depth = 0.3
            absorption_width = 2.0 / KOSMOS_DISPERSION  # pixels
            line_profile = absorption_depth * np.exp(-((wavelengths - line_wave) / (absorption_width * KOSMOS_DISPERSION))**2)
            continuum *= (1 - line_profile)
        
        # Adjust to target SNR
        # SNR ≈ signal / sqrt(signal + sky + readnoise²)
        signal_electrons = trace_snr**2  # Simplified
        signal_adu = signal_electrons / KOSMOS_GAIN
        continuum = continuum / np.median(continuum) * signal_adu
        
        # Spatial profile (Gaussian PSF)
        fwhm_spatial = 4.0
        sigma_spatial = fwhm_spatial / 2.355
        y = np.arange(KOSMOS_SHAPE[1])
        spatial_profile = np.exp(-((y - trace_y) / sigma_spatial)**2)
        
        # Add trace to data
        trace_data = continuum[:, np.newaxis] * spatial_profile[np.newaxis, :]
        data += trace_data
    
    # Add cosmic rays
    num_cosmics = int(cosmic_ray_fraction * data.size)
    cosmic_x = np.random.randint(0, KOSMOS_SHAPE[0], size=num_cosmics)
    cosmic_y = np.random.randint(0, KOSMOS_SHAPE[1], size=num_cosmics)
    cosmic_intensity = np.random.uniform(5000, 50000, size=num_cosmics)
    data[cosmic_x, cosmic_y] += cosmic_intensity
    
    # Add Poisson noise
    data_electrons = data * KOSMOS_GAIN
    noisy_electrons = np.random.poisson(np.maximum(data_electrons, 0))
    data = noisy_electrons / KOSMOS_GAIN
    
    # Add read noise
    noise_adu = KOSMOS_READNOISE / KOSMOS_GAIN
    data += np.random.normal(0, noise_adu, KOSMOS_SHAPE)
    
    # Create FITS
    header = create_kosmos_header('object', exposure_time=exposure_time, 
                                 object_name=object_name, airmass=airmass)
    hdu = fits.PrimaryHDU(data=data.astype(np.float32), header=header)
    hdu.writeto(output_path, overwrite=True)
    
    logger.info(f"Generated science frame: {output_path} ({num_traces} trace(s), SNR~{trace_snr})")


def generate_test_dataset(output_dir: Path, num_bias: int = 10,
                         num_flat: int = 10, num_arc: int = 3,
                         num_science: int = 5, seed: Optional[int] = None) -> Dict[str, List[Path]]:
    """
    Generate complete test dataset.
    
    Per T113: Creates a full set of synthetic KOSMOS data for testing
    the entire reduction pipeline.
    
    Parameters
    ----------
    output_dir : Path
        Output directory for FITS files
    num_bias : int
        Number of bias frames
    num_flat : int
        Number of flat frames
    num_arc : int
        Number of arc frames
    num_science : int
        Number of science frames
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary with keys 'bias', 'flat', 'arc', 'science' mapping to lists of file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    files = {'bias': [], 'flat': [], 'arc': [], 'science': []}
    
    logger.info(f"Generating test dataset in {output_dir}")
    
    # Generate bias frames
    for i in range(num_bias):
        path = output_dir / f"bias_{i+1:03d}.fits"
        generate_bias_frame(path, seed=seed+i if seed else None)
        files['bias'].append(path)
    
    # Generate flat frames
    for i in range(num_flat):
        path = output_dir / f"flat_{i+1:03d}.fits"
        generate_flat_frame(path, seed=seed+100+i if seed else None)
        files['flat'].append(path)
    
    # Generate arc frames
    for i in range(num_arc):
        path = output_dir / f"arc_{i+1:03d}.fits"
        generate_arc_frame(path, seed=seed+200+i if seed else None)
        files['arc'].append(path)
    
    # Generate science frames (variety of SNR and configurations)
    for i in range(num_science):
        path = output_dir / f"science_{i+1:03d}.fits"
        # Vary parameters for diversity
        num_traces = 1 if i < num_science//2 else 2
        snr = 5.0 + i * 5.0  # Increasing SNR
        generate_science_frame(path, num_traces=num_traces, trace_snr=snr,
                             object_name=f"Galaxy{i+1}",
                             seed=seed+300+i if seed else None)
        files['science'].append(path)
    
    logger.info(f"Generated {sum(len(v) for v in files.values())} FITS files")
    
    return files
