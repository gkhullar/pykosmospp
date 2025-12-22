# Changelog

All notable changes to the pyKOSMOS Spectral Reduction Pipeline will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-12-22

### Added - MVP Release

#### Core Pipeline (User Story 1)
- **Automated end-to-end reduction pipeline** from raw FITS files to wavelength-calibrated 1D spectra
- **PipelineRunner class** orchestrating 10-step workflow with comprehensive error handling
- **CLI tool** `kosmos-reduce` with argparse interface, subcommands, and exit codes

#### Calibration Module
- Master bias creation via sigma-clipped median combining
- Master flat creation with bias subtraction and normalization
- Calibration validation checking bias variation (<10 ADU) and flat quality
- Cosmic ray detection using L.A.Cosmic algorithm

#### Wavelength Calibration
- Arc lamp line detection with SNR threshold and centroid refinement
- Line catalog matching with initial dispersion guess
- Chebyshev polynomial fitting (order 3-7) with BIC model selection
- Iterative 3σ outlier rejection (max 5 iterations)
- RMS validation (<0.1Å target, <0.2Å acceptance per FR-008)
- Wavelength solution application to create Spectrum1D objects

#### Spectral Extraction
- Cross-correlation trace detection with emission line masking
- Spatial profile fitting (Gaussian/Moffat) with empirical fallback
- Sky background estimation from trace-free regions
- Optimal extraction (Horne 1986) with variance propagation
- Aperture extraction fallback

#### Quality Assessment
- **QualityMetrics class** tracking SNR, wavelength RMS, cosmic ray fraction, saturation
- **ReducedData class** containing complete data products with provenance
- Quality grading system (Excellent/Good/Fair/Poor) per FR-009
- Calibration validation per FR-010
- LaTeX-formatted diagnostic plots:
  - 2D spectrum with log-scale colormap
  - Wavelength solution residuals
  - Spatial profile fits
  - Sky background subtraction

#### Data Models
- **RawFrame hierarchy**: BiasFrame, FlatFrame, ArcFrame, ScienceFrame with FITS I/O
- **Calibration classes**: MasterBias, MasterFlat, CalibrationSet
- **Spectroscopic data**: Spectrum2D, Trace, SpatialProfile, WavelengthSolution
- **ObservationSet**: Frame discovery and organization
- **Configuration**: PipelineConfig with YAML loading

#### I/O and Utilities
- FITS file discovery and classification by IMAGETYP header
- Configuration loading from YAML with validation
- Provenance tracking in FITS headers
- Structured logging with verbosity control

### Testing
- 33 unit tests passing (wavelength, extraction, quality modules)
- Integration tests for pipeline workflow and CLI
- Synthetic data generation for testing
- Test coverage for error handling and edge cases

### Documentation
- Comprehensive README with usage examples
- CLI reference with options and exit codes
- Python API examples
- Output products specification
- Configuration file documentation

### Dependencies
- Python ≥3.10
- astropy ≥5.3 (FITS I/O, CCDData, sigma-clipped stats)
- specutils ≥1.10 (Spectrum1D)
- scipy ≥1.10 (signal processing, optimization)
- numpy ≥1.23 (array operations)
- matplotlib ≥3.6 (diagnostic plots)
- pyyaml ≥6.0 (configuration)

### Architecture
- Modular package structure: calibration/, extraction/, wavelength/, quality/, io/
- Constitution-compliant: Uses established astronomy packages (astropy, specutils)
- Physics-validated algorithms: Horne 1986 extraction, L.A.Cosmic cosmic ray detection
- Provenance tracking in all data products
- Comprehensive error handling (CriticalPipelineError, QualityWarning)

### Known Limitations
- Interactive trace viewer not yet implemented (planned for v0.2.0)
- Flux calibration not yet supported (planned for v0.3.0)
- Multi-object spectroscopy requires manual trace selection
- LaTeX plotting requires LaTeX installation
- Pipeline tested with synthetic data, real KOSMOS validation pending

## [Unreleased]

### Planned for v0.2.0 (User Story 2)
- Interactive trace viewer with matplotlib widgets
- Manual trace selection and validation
- Real-time profile visualization
- Enhanced trace detection algorithms

### Planned for v0.3.0 (User Story 3)
- Robust wavelength calibration with multiple arc lamps
- Automatic lamp type detection
- Template matching for poor SNR arcs
- Cross-correlation wavelength refinement

### Planned for v0.4.0 (User Story 4)
- Comprehensive quality validation framework
- Automated quality report generation
- Bad pixel masking and interpolation
- Sensitivity analysis and diagnostics

### Planned for v0.5.0 (User Story 5)
- Multi-trace batch processing
- Parallel reduction of observation sets
- A-B sky subtraction for nod-and-shuffle
- Automated output organization

### Planned for v1.0.0 (Polish Phase)
- Spectral binning utilities
- Flux calibration with standard stars
- Complete test suite with real data
- User documentation and tutorials
- Performance optimization

---

[0.1.0]: https://github.com/gkhullar/pykosmospp/releases/tag/v0.1.0
