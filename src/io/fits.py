"""
FITS file I/O utilities for KOSMOS pipeline.

Functions for reading FITS as CCDData, validating headers, and writing with provenance.
Per tasks.md T018.
"""

from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import astropy.units as u
from astropy.nddata import CCDData
from astropy.io import fits


def read_fits_as_ccddata(filepath: Path, unit: str = 'adu', 
                         gain: float = 1.4, readnoise: float = 3.7) -> CCDData:
    """
    Read FITS file as astropy CCDData object.
    
    Parameters
    ----------
    filepath : Path
        Path to FITS file
    unit : str, optional
        Data unit (default: 'adu')
    gain : float, optional
        CCD gain in e-/ADU (default: 1.4 for KOSMOS)
    readnoise : float, optional
        Read noise in e- (default: 3.7 for KOSMOS)
        
    Returns
    -------
    CCDData
        FITS data as CCDData with metadata
        
    Raises
    ------
    FileNotFoundError
        If FITS file does not exist
    ValueError
        If FITS file cannot be read
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"FITS file not found: {filepath}")
    
    try:
        # Read as CCDData with unit
        ccd = CCDData.read(filepath, unit=unit)
        
        # Add instrument metadata to header if not present
        if 'GAIN' not in ccd.header:
            ccd.header['GAIN'] = (gain, 'CCD gain [e-/ADU]')
        if 'RDNOISE' not in ccd.header:
            ccd.header['RDNOISE'] = (readnoise, 'Read noise [e-]')
            
        # Add provenance
        ccd.header['FILENAME'] = (filepath.name, 'Original filename')
        ccd.header['FILEPATH'] = (str(filepath), 'Original file path')
        
        return ccd
        
    except Exception as e:
        raise ValueError(f"Failed to read FITS file {filepath}: {e}")


def validate_fits_header(header: fits.Header, required_keywords: Optional[list] = None) -> bool:
    """
    Validate FITS header contains required keywords.
    
    Parameters
    ----------
    header : fits.Header
        FITS header to validate
    required_keywords : list, optional
        List of required keyword names. If None, uses default set.
        
    Returns
    -------
    bool
        True if all required keywords present
        
    Raises
    ------
    ValueError
        If required keywords missing
    """
    if required_keywords is None:
        # Default required keywords for KOSMOS
        required_keywords = ['IMAGETYP', 'EXPTIME', 'DATE-OBS']
    
    missing = [kw for kw in required_keywords if kw not in header]
    
    if missing:
        raise ValueError(f"Missing required FITS header keywords: {missing}")
    
    return True


def write_fits_with_provenance(ccd: CCDData, filepath: Path, 
                               processing_steps: Optional[Dict] = None,
                               overwrite: bool = False) -> None:
    """
    Write CCDData to FITS file with full provenance tracking.
    
    Per Constitution Principle III: Data Provenance & Reproducibility
    
    Parameters
    ----------
    ccd : CCDData
        Data to write
    filepath : Path
        Output FITS file path
    processing_steps : dict, optional
        Dictionary of processing steps applied (stage name -> parameters)
    overwrite : bool, optional
        Whether to overwrite existing file (default: False)
        
    Raises
    ------
    FileExistsError
        If file exists and overwrite=False
    """
    filepath = Path(filepath)
    
    if filepath.exists() and not overwrite:
        raise FileExistsError(f"Output file exists: {filepath}. Use overwrite=True to replace.")
    
    # Add provenance to header
    ccd.header['PIPEVER'] = ('0.1.0', 'Pipeline version')
    ccd.header['PROCDATE'] = (datetime.now().isoformat(), 'Processing date')
    
    if processing_steps:
        # Add processing history
        for i, (stage, params) in enumerate(processing_steps.items(), start=1):
            key = f'HISTORY{i:02d}'
            value = f"{stage}: {params}"
            ccd.header[key] = (value, 'Processing step')
    
    # Create output directory if needed
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Write FITS file
    ccd.write(filepath, overwrite=overwrite)
