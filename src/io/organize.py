"""
Output directory organization for KOSMOS pipeline.

Functions for creating organized output structure and discovering input files.
Per tasks.md T020, T029.
"""

from pathlib import Path
from typing import List, Dict, Tuple
from astropy.io import fits


def create_output_dirs(output_dir: Path, subdirs: List[str] = None) -> Dict[str, Path]:
    """
    Create organized output directory structure.
    
    Per FR-012: Product-type directories for each galaxy:
    - calibrations/ : Master bias, flat, arc wavelength solutions
    - reduced_2d/ : Calibrated 2D science frames
    - spectra_1d/ : Extracted wavelength-calibrated 1D spectra
    - wavelength_solutions/ : Wavelength solution FITS tables
    - plots/ : Diagnostic plots
    - logs/ : Processing logs and quality reports
    
    Parameters
    ----------
    output_dir : Path
        Base output directory
    subdirs : list, optional
        List of subdirectory names to create. If None, uses default set.
        
    Returns
    -------
    dict
        Dictionary mapping subdirectory name to full path
    """
    if subdirs is None:
        subdirs = ['calibrations', 'reduced_2d', 'spectra_1d', 
                  'wavelength_solutions', 'plots', 'logs']
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    for subdir in subdirs:
        subdir_path = output_dir / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)
        paths[subdir] = subdir_path
    
    return paths


def discover_fits_files(input_dir: Path) -> Dict[str, List[Path]]:
    """
    Scan input directories and classify FITS files by type.
    
    Per FR-002: Input structure with arcs/, flats/, biases/, science/
    subdirectories. Classifies files using FITS header IMAGETYP keyword.
    
    Parameters
    ----------
    input_dir : Path
        Base input directory containing subdirectories
        
    Returns
    -------
    dict
        Dictionary mapping file type to list of paths:
        {'bias': [...], 'flat': [...], 'arc': [...], 'science': [...]}
        
    Raises
    ------
    FileNotFoundError
        If input directory doesn't exist
    ValueError
        If no FITS files found
    """
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Define expected subdirectories and their mapping to frame types
    subdir_map = {
        'biases': 'bias',
        'flats': 'flat',
        'arcs': 'arc',
        'science': 'science',
    }
    
    files_by_type = {ftype: [] for ftype in subdir_map.values()}
    
    # Scan each subdirectory
    for subdir_name, frame_type in subdir_map.items():
        subdir_path = input_dir / subdir_name
        
        if not subdir_path.exists():
            continue
        
        # Find all FITS files
        fits_files = list(subdir_path.glob('*.fits')) + list(subdir_path.glob('*.fit'))
        
        for fits_file in fits_files:
            # Verify file type from FITS header
            try:
                header = fits.getheader(fits_file)
                image_type = header.get('IMAGETYP', '').lower()
                
                # Match image type to expected frame type
                if _matches_frame_type(image_type, frame_type):
                    files_by_type[frame_type].append(fits_file)
                else:
                    # File in wrong directory - classify by header
                    classified_type = _classify_by_imagetyp(image_type)
                    if classified_type:
                        files_by_type[classified_type].append(fits_file)
                        
            except Exception as e:
                # Skip files that can't be read as FITS
                print(f"Warning: Could not read {fits_file}: {e}")
                continue
    
    # Check if any files were found
    total_files = sum(len(files) for files in files_by_type.values())
    if total_files == 0:
        raise ValueError(f"No FITS files found in {input_dir}")
    
    return files_by_type


def _matches_frame_type(image_type: str, expected_type: str) -> bool:
    """
    Check if IMAGETYP matches expected frame type.
    
    Parameters
    ----------
    image_type : str
        IMAGETYP header value (lowercase)
    expected_type : str
        Expected frame type (bias, flat, arc, science)
        
    Returns
    -------
    bool
        True if types match
    """
    type_keywords = {
        'bias': ['bias', 'zero'],
        'flat': ['flat'],
        'arc': ['arc', 'comp', 'comparison'],
        'science': ['object', 'science'],
    }
    
    keywords = type_keywords.get(expected_type, [])
    return any(kw in image_type for kw in keywords)


def _classify_by_imagetyp(image_type: str) -> str:
    """
    Classify frame type from IMAGETYP keyword.
    
    Parameters
    ----------
    image_type : str
        IMAGETYP header value (lowercase)
        
    Returns
    -------
    str
        Frame type (bias, flat, arc, science) or None if unknown
    """
    if 'bias' in image_type or 'zero' in image_type:
        return 'bias'
    elif 'flat' in image_type:
        return 'flat'
    elif 'arc' in image_type or 'comp' in image_type:
        return 'arc'
    elif 'object' in image_type or 'science' in image_type:
        return 'science'
    else:
        return None
