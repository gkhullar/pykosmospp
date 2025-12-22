# Reference Documentation

This directory contains reference documentation for spectroscopic reduction methodology.

## Files

### `massey_ccd_reductions.pdf`

**Title**: A User's Guide to CCD Reductions with IRAF  
**Authors**: Philip Massey and Margaret M. Hanson  
**Source**: <https://ned.ipac.caltech.edu/level5/Sept19/Massey/paper.pdf>  
**Downloaded**: 2025-12-22

**Description**: Comprehensive guide to CCD spectroscopy reduction methodology, including:

- **Chapter 1**: Introduction to CCD detectors and data characteristics
- **Chapter 2**: Bias frame processing and overscan correction
- **Chapter 3**: Flat field correction and illumination correction
- **Chapter 4**: Cosmic ray rejection techniques
- **Chapter 5**: Wavelength calibration with arc lamps
  - Arc line identification procedures
  - Wavelength solution fitting (polynomial, spline)
  - Dispersion function validation
- **Chapter 6**: Sky background subtraction methods
  - Longslit spectroscopy sky estimation
  - Spatial profile modeling
- **Chapter 7**: Spectral extraction
  - Aperture extraction
  - Optimal extraction (variance-weighted)
- **Chapter 8**: Flux calibration with standard stars
- **Chapter 9**: Telluric correction

**Relevance to Pipeline**:

- Defines standard spectroscopy reduction workflow (essential steps)
- Provides best practices for wavelength calibration (FR-007, FR-008)
- Describes sky subtraction methodology for longslit spectroscopy (FR-006)
- Explains optimal extraction techniques (FR-007)
- Referenced in specification as authoritative methodology source

**Key Concepts**:

- Bias estimation: Median combine multiple bias frames for stable bias pattern
- Wavelength calibration: Identify arc lines, fit dispersion relation with robust fitting
- Sky subtraction: Use spatial regions away from object trace for sky estimation
- Optimal extraction: Weight by spatial profile for improved S/N
- Error propagation: Track uncertainties through all reduction stages

## Usage

This PDF serves as the authoritative reference for spectroscopic reduction methodology. During planning and implementation:

1. **Planning Phase** (`/speckit.plan`): Reference Massey methodology when defining algorithm choices in `research.md`
2. **Implementation Phase** (`/speckit.implement`): Consult specific chapters for algorithm details and best practices
3. **Testing Phase**: Validate pipeline results against Massey standards for data quality

## Relation to Specification

- **FR-001**: Calibration frame processing (Chapters 2-3)
- **FR-003**: Bias subtraction methodology (Chapter 2)
- **FR-004**: Flat field correction (Chapter 3)
- **FR-006**: Sky background subtraction (Chapter 6)
- **FR-007**: Spectral extraction (Chapter 7)
- **FR-008**: Wavelength calibration (Chapter 5)
- **FR-009**: Flux calibration and extinction (Chapter 8)
- **FR-013**: Cosmic ray rejection (Chapter 4)

## Other References

- **pyKOSMOS**: <https://github.com/jradavenport/pykosmos> - Davenport, J. R. A. et al. (2023). DOI:10.5281/zenodo.10152905
  - Original longslit spectroscopy reduction package for APO-KOSMOS
  - Authors: James R. A. Davenport, Francisca Chabour Barra, Azalee Bostroem, Erin Howard (University of Washington)
- **PyDIS**: <https://github.com/StellarCartography/pydis> - Davenport, J. R. A. (2016). Predecessor to pyKOSMOS
- **specreduce**: <https://github.com/astropy/specreduce> - Astropy specreduce (inherited methods from PyDIS and pyKOSMOS)
- **PypeIt Documentation**: <https://pypeit.readthedocs.io/en/stable/> (modern Python implementation of these principles)
- **pyKOSMOS Resources**: `pykosmos_reference/` (reference data for wavelength/flux calibration from pyKOSMOS repository)
- **Specification**: `../specs/001-galaxy-spec-pipeline/spec.md`
