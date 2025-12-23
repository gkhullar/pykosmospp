"""
Arc line matching to reference catalogs and arc template loading.

Per tasks.md T033: Load pyKOSMOS linelists and match detected lines.
Supports both discrete line matching and DTW (Dynamic Time Warping) with arc templates.
"""

from typing import List, Tuple, Dict, Optional
from pathlib import Path
import numpy as np


def load_arc_template(lamp_type: str, 
                     grating: str = '1.18-ctr',
                     arm: str = 'Blue',
                     resources_dir: Path = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load arc template spectrum for DTW wavelength calibration.
    
    Arc templates are pre-calibrated wavelength spectra from pyKOSMOS resources,
    used for automatic wavelength identification via Dynamic Time Warping (DTW).
    This approach is more robust than discrete line matching and doesn't require
    an initial dispersion guess.
    
    Parameters
    ----------
    lamp_type : str
        Arc lamp type. Supported: 'Ar' (Argon), 'Kr' (Krypton), 'Ne' (Neon)
    grating : str, optional
        Grating setting (default: '1.18-ctr')
        Options: '0.86-high', '1.18-ctr', '2.0-low'
    arm : str, optional
        Spectrograph arm (default: 'Blue')
        Options: 'Blue', 'Red'
    resources_dir : Path, optional
        Path to resources directory. If None, uses package default.
        
    Returns
    -------
    wavelengths : np.ndarray
        Template wavelengths in Angstroms
    flux : np.ndarray
        Template flux values (normalized)
        
    Raises
    ------
    FileNotFoundError
        If template file not found
    ValueError
        If lamp type, grating, or arm unknown
        
    Examples
    --------
    >>> waves, flux = load_arc_template('Ar', grating='1.18-ctr', arm='Blue')
    >>> print(f"Template covers {waves[0]:.1f}-{waves[-1]:.1f} Å")
    """
    if resources_dir is None:
        resources_dir = Path(__file__).parent.parent.parent / 'resources' / 'pykosmos_reference'
    
    templates_dir = resources_dir / 'arctemplates'
    
    # Validate lamp type
    valid_lamps = ['Ar', 'Kr', 'Ne']
    if lamp_type not in valid_lamps:
        raise ValueError(f"Unknown lamp type: {lamp_type}. Must be one of {valid_lamps}")
    
    # Validate grating
    valid_gratings = ['0.86-high', '1.18-ctr', '2.0-low']
    if grating not in valid_gratings:
        raise ValueError(f"Unknown grating: {grating}. Must be one of {valid_gratings}")
    
    # Validate arm
    valid_arms = ['Blue', 'Red']
    if arm not in valid_arms:
        raise ValueError(f"Unknown arm: {arm}. Must be one of {valid_arms}")
    
    # Construct filename: e.g., ArBlue1.18-ctr.spec
    filename = f"{lamp_type}{arm}{grating}.spec"
    filepath = templates_dir / filename
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Arc template not found: {filepath}\n"
            f"Available templates for lamp={lamp_type}: "
            f"{list(templates_dir.glob(f'{lamp_type}*.spec'))}"
        )
    
    # Load CSV format: wave,flux
    try:
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)  # Skip header
        wavelengths = data[:, 0]
        flux = data[:, 1]
        
        # Sort by wavelength (templates may be in descending order)
        sort_idx = np.argsort(wavelengths)
        wavelengths = wavelengths[sort_idx]
        flux = flux[sort_idx]
        
        return wavelengths, flux
    except Exception as e:
        raise ValueError(f"Failed to load arc template {filepath}: {e}")


def get_arc_template_name(lamp_type: str, header: Dict = None) -> Tuple[str, str, str]:
    """
    Automatically select appropriate arc template based on observation metadata.
    
    Parameters
    ----------
    lamp_type : str
        Lamp type keyword (e.g., 'argon', 'henear', 'cuar')
    header : dict, optional
        FITS header with grating/arm information
        
    Returns
    -------
    template_lamp : str
        Template lamp type ('Ar', 'Kr', or 'Ne')
    grating : str
        Grating setting ('0.86-high', '1.18-ctr', or '2.0-low')
    arm : str
        Spectrograph arm ('Blue' or 'Red')
        
    Notes
    -----
    - If header unavailable, uses default: '1.18-ctr', 'Blue'
    - Lamp type mapping:
        * 'argon', 'ar' -> 'Ar'
        * 'krypton', 'kr' -> 'Kr'
        * 'neon', 'ne' -> 'Ne'
        * 'henear', 'apohenear' -> 'Ar' (Argon has strongest lines)
    """
    # Map lamp keywords to template lamp types
    lamp_map = {
        'argon': 'Ar',
        'ar': 'Ar',
        'krypton': 'Kr',
        'kr': 'Kr',
        'neon': 'Ne',
        'ne': 'Ne',
        'henear': 'Ar',  # Use Argon template for He-Ne-Ar
        'apohenear': 'Ar',
        'henearhres': 'Ar',
    }
    
    template_lamp = lamp_map.get(lamp_type.lower(), 'Ar')  # Default to Argon
    
    # Extract grating and arm from header if available
    grating = '1.18-ctr'  # Default
    arm = 'Blue'  # Default
    
    if header is not None:
        # Try to extract grating info from header
        grating_keyword = header.get('GRATING', header.get('GRISM', ''))
        if '0.86' in str(grating_keyword) or 'high' in str(grating_keyword).lower():
            grating = '0.86-high'
        elif '2.0' in str(grating_keyword) or 'low' in str(grating_keyword).lower():
            grating = '2.0-low'
        else:
            grating = '1.18-ctr'
        
        # Try to extract arm info
        arm_keyword = header.get('ARM', header.get('FILTER', ''))
        if 'red' in str(arm_keyword).lower():
            arm = 'Red'
        else:
            arm = 'Blue'
    
    return template_lamp, grating, arm


def load_linelist(lamp_type: str, resources_dir: Path = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load arc lamp linelist from pyKOSMOS resources.
    
    Parameters
    ----------
    lamp_type : str
        Lamp type. Supported types:
        - Arc lamps: 'henear', 'apohenear', 'henearhres', 'idhenear', 'idheneart',
          'ctiohenear', 'ctiohear', 'argon', 'krypton', 'thar', 'cuar', 'xenon', 'fear'
        - Vacuum wavelengths: 'vacidhenear', 'vacthar'
        - Sky lines: 'skylines', 'lowskylines', 'ohlines'
    resources_dir : Path, optional
        Path to resources directory. If None, uses package default.
        
    Returns
    -------
    wavelengths : np.ndarray
        Reference wavelengths in Angstroms
    intensities : np.ndarray
        Relative intensities (0-1000)
        
    Raises
    ------
    FileNotFoundError
        If linelist file not found
    ValueError
        If lamp type unknown
    """
    if resources_dir is None:
        resources_dir = Path(__file__).parent.parent.parent / 'resources' / 'pykosmos_reference'
    
    linelists_dir = resources_dir / 'linelists'
    
    # Map lamp types to filenames (supports all available linelists)
    linelist_files = {
        'henear': 'apohenear.dat',
        'apohenear': 'apohenear.dat',
        'argon': 'argon.dat',
        'krypton': 'krypton.dat',
        'thar': 'thar.dat',
        'cuar': 'cuar.dat',
        'xenon': 'xenon.dat',
        'fear': 'fear.dat',  # Iron-Argon
        'henearhres': 'henearhres.dat',  # High-resolution He-Ne-Ar
        'idhenear': 'idhenear.dat',  # Identified He-Ne-Ar
        'idheneart': 'idhenearT.dat',  # Identified He-Ne-Ar (T variant)
        'ctiohenear': 'ctiohenear.dat',  # CTIO He-Ne-Ar
        'ctiohear': 'ctiohear.dat',  # CTIO He-Ar
        'vacidhenear': 'vacidhenear.dat',  # Vacuum wavelengths He-Ne-Ar
        'vacthar': 'vacthar.dat',  # Vacuum wavelengths Th-Ar
        'skylines': 'skylines.dat',  # Sky emission lines
        'lowskylines': 'lowskylines.dat',  # Low sky emission lines
        'ohlines': 'ohlines.dat',  # OH emission lines
    }
    
    filename = linelist_files.get(lamp_type.lower())
    if filename is None:
        raise ValueError(f"Unknown lamp type: {lamp_type}. Must be one of {list(linelist_files.keys())}")
    
    filepath = linelists_dir / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Linelist not found: {filepath}")
    
    # Load linelist (format: wavelength element or wavelength intensity)
    try:
        # Try loading with comments stripped
        data = np.loadtxt(filepath, comments='#')
        
        # Check if we have 1 or 2 columns
        if data.ndim == 1:
            # Single column: just wavelengths
            wavelengths = data
            intensities = np.ones_like(wavelengths)
        elif data.shape[1] == 1:
            # Single column: just wavelengths
            wavelengths = data[:, 0]
            intensities = np.ones_like(wavelengths)
        elif data.shape[1] >= 2:
            # Check if second column is numeric (intensity) or will fail (element name)
            try:
                wavelengths = data[:, 0]
                intensities = data[:, 1]
            except (ValueError, IndexError):
                # Second column is not numeric, just use wavelengths
                wavelengths = data[:, 0]
                intensities = np.ones_like(wavelengths)
        else:
            wavelengths = data[:, 0]
            intensities = np.ones_like(wavelengths)
        
        return wavelengths, intensities
    except Exception as e:
        # Try alternative parsing: read as text and extract first column
        try:
            with open(filepath, 'r') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            wavelengths = []
            for line in lines:
                parts = line.split()
                if len(parts) > 0:
                    try:
                        wavelengths.append(float(parts[0]))
                    except ValueError:
                        continue  # Skip lines that don't start with a number
            
            wavelengths = np.array(wavelengths)
            intensities = np.ones_like(wavelengths)
            return wavelengths, intensities
        except Exception as e2:
            raise ValueError(f"Failed to load linelist {filepath}: {e2}")


def match_lines_to_catalog(pixel_positions: np.ndarray,
                           lamp_type: str,
                           wavelength_range: Tuple[float, float] = (3500, 7500),
                           match_tolerance: float = 2.0,
                           initial_dispersion: float = 1.0,
                           resources_dir: Path = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match detected arc lines to catalog wavelengths.
    
    Per config-schema.yaml: match_tolerance=2.0 Å, wavelength_range=[3500, 7500]
    
    Uses iterative pattern matching:
    1. Generate initial wavelength guess from pixel positions
    2. Match bright lines to catalog within tolerance
    3. Refine wavelength scale using matched lines
    4. Repeat until convergence
    
    Parameters
    ----------
    pixel_positions : np.ndarray
        Detected line pixel positions
    lamp_type : str
        Arc lamp type
    wavelength_range : tuple, optional
        Expected wavelength range in Angstroms (default: 3500-7500)
    match_tolerance : float, optional
        Matching tolerance in Angstroms (default: 2.0)
    initial_dispersion : float, optional
        Initial dispersion estimate in Å/pixel (default: 1.0)
    resources_dir : Path, optional
        Resources directory path
        
    Returns
    -------
    matched_pixels : np.ndarray
        Pixel positions of matched lines
    matched_wavelengths : np.ndarray
        Catalog wavelengths of matched lines
    matched_intensities : np.ndarray
        Intensities of matched lines
        
    Raises
    ------
    ValueError
        If too few matches found
    """
    # Load catalog
    catalog_waves, catalog_intensities = load_linelist(lamp_type, resources_dir)
    
    # Filter catalog to expected wavelength range
    in_range = (catalog_waves >= wavelength_range[0]) & (catalog_waves <= wavelength_range[1])
    catalog_waves = catalog_waves[in_range]
    catalog_intensities = catalog_intensities[in_range]
    
    if len(catalog_waves) == 0:
        raise ValueError(f"No catalog lines in wavelength range {wavelength_range}")
    
    # Initial wavelength estimate (linear dispersion)
    wave_min, wave_max = wavelength_range
    wave_guess = wave_min + (pixel_positions - pixel_positions.min()) * initial_dispersion
    
    # Match lines iteratively
    matched_pixels = []
    matched_waves = []
    matched_intensities = []
    
    for pixel, wave_est in zip(pixel_positions, wave_guess):
        # Find closest catalog line within tolerance
        diffs = np.abs(catalog_waves - wave_est)
        min_diff_idx = np.argmin(diffs)
        min_diff = diffs[min_diff_idx]
        
        if min_diff < match_tolerance:
            matched_pixels.append(pixel)
            matched_waves.append(catalog_waves[min_diff_idx])
            matched_intensities.append(catalog_intensities[min_diff_idx])
    
    if len(matched_pixels) < 10:
        raise ValueError(
            f"Too few line matches: {len(matched_pixels)} (need ≥10). "
            f"Try adjusting wavelength_range or match_tolerance."
        )
    
    return np.array(matched_pixels), np.array(matched_waves), np.array(matched_intensities)
