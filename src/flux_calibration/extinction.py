"""
Atmospheric extinction correction.

Per tasks.md T107 and FR-009: Apply extinction correction using APO
extinction curve from pyKOSMOS apoextinct.dat.
"""

import numpy as np
from specutils import Spectrum1D
import astropy.units as u
from pathlib import Path
from typing import Optional
import warnings


def load_apo_extinction(extinction_file: Optional[Path] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Load APO atmospheric extinction curve.
    
    Parameters
    ----------
    extinction_file : Path, optional
        Path to extinction file. If None, uses apoextinct.dat from
        resources/pykosmos_reference/extinction/
        
    Returns
    -------
    wavelengths : np.ndarray
        Wavelengths in Angstroms
    extinction : np.ndarray
        Extinction in magnitudes per airmass
    """
    if extinction_file is None:
        # Default to pyKOSMOS reference data
        extinction_file = Path(__file__).parents[2] / 'resources' / 'pykosmos_reference' / 'extinction' / 'apoextinct.dat'
    
    if not extinction_file.exists():
        raise FileNotFoundError(
            f"Extinction file not found: {extinction_file}\n"
            f"Download from pyKOSMOS: https://github.com/jradavenport/pykosmos/tree/main/resources/extinction"
        )
    
    # Load extinction data
    # Format: wavelength(Å) extinction(mag/airmass)
    data = np.loadtxt(extinction_file)
    wavelengths = data[:, 0]
    extinction = data[:, 1]
    
    return wavelengths, extinction


def apply_extinction_correction(spectrum: Spectrum1D,
                                airmass: float,
                                extinction_file: Optional[Path] = None) -> Spectrum1D:
    """
    Apply atmospheric extinction correction to spectrum.
    
    Per tasks.md T107 and FR-009: Corrects for atmospheric absorption
    using airmass and APO extinction curve.
    
    Parameters
    ----------
    spectrum : Spectrum1D
        Input spectrum to correct
    airmass : float
        Airmass of observation (sec(zenith angle))
        Typical range: 1.0 (zenith) to 2.0 (30° from zenith)
    extinction_file : Path, optional
        Path to extinction curve file (default: APO curve from pyKOSMOS)
        
    Returns
    -------
    Spectrum1D
        Extinction-corrected spectrum
        
    Notes
    -----
    Extinction correction formula:
    
    .. math::
        F_{corr} = F_{obs} \\times 10^{0.4 \\times E(\\lambda) \\times X}
    
    where:
    - F_obs is observed flux
    - E(λ) is extinction in magnitudes per airmass
    - X is airmass
    
    Examples
    --------
    >>> # Correct for airmass 1.3 observation
    >>> corrected = apply_extinction_correction(spectrum, airmass=1.3)
    >>> print(f"Applied extinction for airmass {airmass}")
    
    >>> # Use custom extinction curve
    >>> corrected = apply_extinction_correction(
    ...     spectrum, 
    ...     airmass=1.5,
    ...     extinction_file=Path('my_extinction.dat')
    ... )
    """
    # Load extinction curve
    ext_wavelengths, ext_magnitudes = load_apo_extinction(extinction_file)
    
    # Interpolate extinction to spectrum wavelengths
    spec_wavelengths = spectrum.spectral_axis.to(u.Angstrom).value
    
    # Interpolate (linear) extinction curve to spectrum wavelengths
    extinction_at_spec = np.interp(
        spec_wavelengths,
        ext_wavelengths,
        ext_magnitudes,
        left=ext_magnitudes[0],  # Extrapolate with edge values
        right=ext_magnitudes[-1]
    )
    
    # Compute correction factor
    # Flux_corrected = Flux_observed * 10^(0.4 * extinction * airmass)
    correction_factor = 10**(0.4 * extinction_at_spec * airmass)
    
    # Apply correction
    corrected_flux = spectrum.flux * correction_factor
    
    # Propagate uncertainty
    if spectrum.uncertainty is not None:
        corrected_uncertainty = spectrum.uncertainty.array * correction_factor * spectrum.flux.unit
        from astropy.nddata import StdDevUncertainty
        corrected_uncertainty = StdDevUncertainty(corrected_uncertainty)
    else:
        corrected_uncertainty = None
    
    # Create corrected spectrum
    corrected_spectrum = Spectrum1D(
        spectral_axis=spectrum.spectral_axis,
        flux=corrected_flux,
        uncertainty=corrected_uncertainty
    )
    
    # Update metadata
    if hasattr(spectrum, 'meta'):
        corrected_spectrum.meta = spectrum.meta.copy()
    else:
        corrected_spectrum.meta = {}
    
    corrected_spectrum.meta['extinction_corrected'] = True
    corrected_spectrum.meta['airmass'] = airmass
    corrected_spectrum.meta['extinction_file'] = str(extinction_file) if extinction_file else 'apoextinct.dat'
    
    # Store mean correction factor for reference
    corrected_spectrum.meta['mean_extinction_correction'] = np.median(correction_factor)
    
    return corrected_spectrum


def compute_airmass(zenith_angle_degrees: float) -> float:
    """
    Compute airmass from zenith angle.
    
    Uses simple sec(z) approximation, valid for zenith angles < 70°.
    
    Parameters
    ----------
    zenith_angle_degrees : float
        Zenith angle in degrees (0° = zenith, 90° = horizon)
        
    Returns
    -------
    float
        Airmass
        
    Examples
    --------
    >>> airmass = compute_airmass(30)  # 30° from zenith
    >>> print(f"Airmass: {airmass:.2f}")
    Airmass: 1.15
    """
    if zenith_angle_degrees >= 70:
        warnings.warn(
            f"Zenith angle {zenith_angle_degrees}° is large. "
            f"Simple sec(z) approximation may be inaccurate."
        )
    
    zenith_angle_radians = np.radians(zenith_angle_degrees)
    airmass = 1.0 / np.cos(zenith_angle_radians)
    
    return airmass
