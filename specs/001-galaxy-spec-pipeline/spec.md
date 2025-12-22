# Feature Specification: APO-KOSMOS Galaxy Spectroscopy Pipeline

**Feature Branch**: `001-galaxy-spec-pipeline`  
**Created**: 2025-12-22  
**Status**: Draft  
**Input**: Build a spectroscopic reduction pipeline for astronomical data collected from the KOSMOS spectrograph on the 3.5m Apache Point Observatory (APO) telescope to extract faint galaxy spectra with robust wavelength calibration

## Clarifications

### Session 2025-12-22

- Q: When multiple potential traces are detected (target galaxy, serendipitous sources, noise), how should the pipeline identify the primary science target? → A: Identify multiple unique traces and prompt user with interactive pop-up viewer to select all traces that look good
- Q: What configuration approach balances flexibility for different observing modes with simplicity for typical users? → A: YAML file with sensible KOSMOS defaults, requiring only target-specific overrides (trace position hints, SNR thresholds)
- Q: How should the pipeline organize input and output files to support both single-target and multi-target observations? → A: Input structure: directories split into arcs/, flats/, biases/, science/; Output structure: product-type organization (calibrations/, reduced_2d/, spectra_1d/, logs/) within each galaxy subdirectory
- Q: For longslit spectroscopy of faint galaxies, how should the pipeline estimate and subtract sky background? → A: Follow pyKOSMOS sky subtraction method as primary approach, with PypeIt framework patterns as fallback/alternative
- Q: What is the pipeline's error handling philosophy for balancing robustness with productivity? → A: Tiered approach - critical failures (corrupt FITS, wrong instrument configuration) halt immediately; quality issues (low SNR, missing arc lines) produce outputs with quality flags and warnings
- Q: What methodology should guide the spectroscopic reduction workflow? → A: Follow standard spectroscopy reduction methodology from Massey (NED Level 5) and PypeIt documentation for essential steps: bias estimation, wavelength calibration with arcs, object finding, object extraction, background subtraction, and AB difference imaging (if applicable); flux calibration and telluric correction are optional steps
- Q: What reference data should be used for wavelength calibration and extinction correction? → A: Utilize pyKOSMOS resources directory (<https://github.com/jradavenport/pykosmos/tree/main/pykosmos/resources>) for arc lamp linelists (linelists/), atmospheric extinction curves (extinction/), arc templates (arctemplates/), and standard star spectra (onedstds/); downloaded locally to resources/pykosmos_reference/ for offline use

## References

- **Massey et al.**: [A User's Guide to CCD Reductions with IRAF](https://ned.ipac.caltech.edu/level5/Sept19/Massey/paper.pdf) - Standard methodology for spectroscopic CCD reduction steps
- **PypeIt Documentation**: [https://pypeit.readthedocs.io/en/stable/](https://pypeit.readthedocs.io/en/stable/) - Modern Python spectroscopy pipeline architecture and reduction workflow
- **pyKOSMOS**: [https://github.com/jradavenport/pykosmos](https://github.com/jradavenport/pykosmos) - Existing KOSMOS reduction pipeline to improve upon
- **pyKOSMOS Resources**: [https://github.com/jradavenport/pykosmos/tree/main/pykosmos/resources](https://github.com/jradavenport/pykosmos/tree/main/pykosmos/resources) - Reference data for wavelength calibration and extinction correction:
  - `linelists/`: Arc lamp emission line catalogs (He-Ne-Ar, Ar, Kr, Th-Ar, Cu-Ar, etc.) for wavelength solution
  - `extinction/`: Observatory-specific atmospheric extinction curves (APO, CTIO, KPNO, ORM)
  - `arctemplates/`: Pre-extracted arc lamp spectra for different KOSMOS grating configurations
  - `onedstds/`: Spectrophotometric standard star templates for flux calibration
- **pyKOSMOS Notebooks**: [Interactive trace example](https://github.com/jradavenport/pykosmos-notebooks/blob/main/interactive_trace_example.ipynb) - Current trace extraction approach

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Push-Button Reduction (Priority: P1)

An astronomer has a night's worth of KOSMOS observations (science frames, arcs, flats, biases) and wants to produce wavelength-calibrated 1D spectra for faint galaxies with minimal manual intervention.

**Why this priority**: Core value proposition - automated pipeline that works end-to-end for faint extended sources, not just bright point sources. This is the fundamental improvement over existing pyKOSMOS.

**Independent Test**: Can be fully tested by providing a complete observation set (science + calibrations) and verifying that pipeline produces 1D spectra files with wavelength solutions and quality metrics without requiring user intervention at each stage.

**Acceptance Scenarios**:

1. **Given** a directory containing raw FITS files (science, arc, flat, bias frames), **When** user runs the pipeline with a single command specifying the target directory, **Then** pipeline automatically identifies file types, performs reduction stages in sequence, and outputs wavelength-calibrated 1D spectra for all detected traces
2. **Given** multiple science exposures of the same target, **When** pipeline processes the full sequence, **Then** intermediate products (bias-corrected, flat-fielded, extracted spectra) are validated at each stage before proceeding, with clear error messages if validation fails
3. **Given** pipeline execution completes, **When** user inspects output directory, **Then** they find organized data products including: reduced 2D frames, extracted 1D spectra, wavelength solution diagnostics, and a processing log documenting all stages

---

### User Story 2 - Faint Trace Identification (Priority: P2)

An astronomer observing low surface brightness galaxies needs the pipeline to reliably identify and extract faint traces that would be missed by simple peak-finding algorithms.

**Why this priority**: Critical for scientific use case (faint galaxies), but can be iterated after basic pipeline infrastructure exists. Distinguishes this pipeline from existing tools.

**Independent Test**: Can be tested independently with synthetic or real faint galaxy data by injecting known traces at various S/N levels and verifying extraction success rate vs. traditional peak-finding methods.

**Acceptance Scenarios**:

1. **Given** a 2D spectrum frame containing a faint extended trace (SNR < 5 per pixel), **When** pipeline performs trace identification, **Then** trace is detected using robust algorithms (e.g., matched filtering, adaptive thresholding, or spatial profile fitting) rather than simple peak detection
2. **Given** multiple traces in the same frame (e.g., serendipitous sources), **When** pipeline identifies traces, **Then** each trace is assigned a unique identifier with metadata (position, approximate flux, spatial extent) and user is prompted with interactive pop-up viewer displaying all detected traces to select which ones to process
3. **Given** a trace barely above detection threshold, **When** pipeline attempts extraction, **Then** system provides quality metrics (SNR estimate, spatial profile consistency) and allows user to set minimum quality thresholds for automated processing

---

### User Story 3 - Robust Wavelength Calibration (Priority: P2)

An astronomer needs accurate wavelength solutions for galaxy spectra to measure redshifts and emission line properties, even when arc lamp lines are faint or partially missing.

**Why this priority**: Wavelength accuracy is fundamental to scientific utility. Must work robustly for various observing conditions. Equal priority to trace extraction since both are core scientific requirements.

**Independent Test**: Can be tested by comparing pipeline wavelength solutions against known arc line positions and reference spectra, measuring RMS residuals across detector.

**Acceptance Scenarios**:

1. **Given** arc lamp exposures with standard emission lines, **When** pipeline performs wavelength calibration, **Then** identified arc lines are matched to reference catalog with RMS residuals < 0.1 Å across full spectral range
2. **Given** an arc frame with some missing or saturated lines, **When** pipeline fits wavelength solution, **Then** algorithm uses robust fitting (sigma-clipping outliers) and reports diagnostic metrics (number of lines used, polynomial order, residuals)
3. **Given** science spectrum and corresponding wavelength solution, **When** applying calibration, **Then** output 1D spectrum includes wavelength array, flux array, error array, and header metadata documenting calibration quality

---

### User Story 4 - Quality Assessment & Validation (Priority: P3)

An astronomer wants confidence that each reduction stage produces valid results before proceeding, with clear diagnostics when problems occur.

**Why this priority**: Important for scientific rigor and debugging, but can be enhanced after core functionality works. Essential for production use but not for initial MVP.

**Independent Test**: Can be tested by intentionally introducing problematic data (bad pixels, cosmic rays, incorrect file types) and verifying pipeline catches issues with informative error messages.

**Acceptance Scenarios**:

1. **Given** pipeline completes a reduction stage, **When** validation checks run, **Then** intermediate products are tested against physics-based criteria (e.g., bias level consistency, flat field normalization, flux conservation) with pass/fail status logged
2. **Given** a reduction stage fails validation, **When** pipeline detects the failure, **Then** processing halts with clear error message indicating: which stage failed, what validation criterion was violated, and suggested remediation (e.g., "Flat field shows < 50% illumination in blue region - check lamp setup")
3. **Given** pipeline successfully completes, **When** user reviews output, **Then** summary report includes: data quality metrics for each frame, S/N estimates for extracted spectra, wavelength calibration residuals, and flags for any marginal products

---

### User Story 5 - Multi-Trace Batch Processing (Priority: P3)

An astronomer with multi-object spectroscopy or multiple targets per night wants to process all traces efficiently with consistent calibrations.

**Why this priority**: Useful for productivity but not essential for initial science use case focused on single faint galaxies. Can be added once single-trace pipeline is robust.

**Independent Test**: Can be tested with multi-slit or multi-target observations by processing batch and verifying each trace gets correct calibrations without cross-contamination.

**Acceptance Scenarios**:

1. **Given** a 2D frame containing multiple traced objects, **When** pipeline processes the frame, **Then** each trace is extracted independently with separate wavelength calibrations and output files labeled by trace ID
2. **Given** a night's observations of different targets, **When** pipeline runs in batch mode, **Then** calibration frames (bias, flat, arc) are associated with correct science frames based on observation timestamps and instrument configuration
3. **Given** batch processing completes, **When** user inspects outputs, **Then** each target has its own subdirectory with reduced products, and a master log tracks which calibrations were applied to which science frames

---

### Edge Cases

- **Trace too faint for identification**: Pipeline should report SNR estimate below threshold and offer options: (a) proceed with user-specified trace position, (b) increase spatial binning to boost SNR, (c) skip frame with warning
- **Arc lamp lines saturated or missing**: Wavelength calibration should use robust subset of available lines and flag reduced wavelength coverage or degraded accuracy
- **Cosmic rays in science frame**: Pipeline should perform cosmic ray rejection (e.g., using LA Cosmic or sigma-clipping multiple exposures) before extraction
- **Flat field has bad columns**: Pipeline should identify and mask bad pixels/columns before normalizing science frames
- **Mismatched calibrations**: If arc/flat frames have different instrumental setup (grating angle, slit width) than science frames, pipeline should warn user before applying calibrations
- **Low SNR spectrum extraction**: If extracted 1D spectrum has SNR < user-defined threshold (e.g., SNR < 3 per resolution element), pipeline should flag spectrum as low quality but still output result with quality warning
- **Overlapping traces in multi-object mode**: Pipeline should detect spatial overlap and warn about potential contamination or attempt deblending if algorithms support it

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST ingest and classify raw FITS files from APO-KOSMOS spectrograph (longslit spectra, arc lamp exposures, flat field exposures, bias frames) organized in input directories by type (arcs/, flats/, biases/, science/), using FITS header keywords (IMAGETYP, OBJECT, EXPTIME) for automatic classification and directory validation, following standard spectroscopy reduction methodology (Massey NED Level 5, PypeIt documentation)
- **FR-003**: System MUST perform **bias estimation** (essential step) using median-combined bias frames, preserving FITS header provenance (validation criteria per FR-010)
- **FR-004**: System MUST normalize flat field frames and apply flat fielding to science and arc frames, masking bad pixels
- **FR-005**: System MUST identify spectral traces in 2D science frames (**object finding** essential step) using cross-correlation with Gaussian kernel templates (width=expected_fwhm from config, default 4.0 pixels) as primary method, excluding emission line regions via spatial gradient threshold (per research.md §1 decision); alternative robust algorithms (adaptive thresholding, matched filtering) listed for research context only; System MUST display detected traces in interactive pop-up viewer for user selection
- **FR-006**: System MUST extract 1D spectra from 2D frames (**object extraction** essential step) using optimal extraction (Horne 1986) or aperture extraction with sky **background subtraction** (essential step) following pyKOSMOS methodology (median of spatial regions flanking trace ± sky_buffer pixels, default 30px per config-schema.yaml, sigma-clipped with sigma=3.0), with PypeIt framework patterns as alternative; System MUST support **AB difference imaging** (optional step) when BOTH conditions met: (1) config parameter `ab_subtraction.enabled=True` AND (2) FITS headers contain nod_position metadata ('A'/'B') or observation timestamps enable pairing
- **FR-007**: System MUST identify arc emission lines in calibration frames (**wavelength calibration with arcs** essential step) using peak-finding with configurable detection threshold; System MUST use arc lamp linelists from pyKOSMOS resources (e.g., apohenear.dat, argon.dat, krypton.dat) for line identification
- **FR-008**: System MUST fit polynomial wavelength solutions to arc line identifications with sigma-clipped robust fitting, reporting RMS residuals (implementation target <0.1 Å for typical data quality; acceptance criterion <0.2 Å per SC-003 allows degraded conditions while maintaining scientific validity)
- **FR-009**: System MUST apply wavelength calibration to extracted 1D science spectra, producing wavelength-flux-error arrays; System MAY optionally apply **flux calibration** (optional step) using standard star observations and pyKOSMOS onedstds templates if provided, and **telluric correction** (optional step) if requested in configuration; System MAY use pyKOSMOS extinction curves (apoextinct.dat for APO observations) for atmospheric extinction correction
- **FR-010**: System MUST validate intermediate products at each stage using physics-based checks with explicit thresholds per config-schema.yaml (e.g., bias level 300-500 ADU, bias stdev <10 ADU, flat median counts 10k-50k ADU for well-exposed frames, saturation fraction <0.01, flux conservation after extraction within 5%)
- **FR-011**: System MUST log all processing steps with timestamps, including input files, parameters used, and validation results
- **FR-012**: System MUST output organized data products in product-type directories for each galaxy: calibrations/ (combined bias/flat/arc), reduced_2d/ (science frames after calibration), spectra_1d/ (extracted wavelength-calibrated spectra), logs/ (processing records and quality metrics)
- **FR-013**: System MUST handle cosmic ray rejection in science frames using LA Cosmic algorithm for single frames or median-combining for ≥3 exposures of same field
- **FR-014**: System MUST propagate uncertainties through all reduction stages, outputting error arrays for 1D spectra based on Poisson statistics and read noise
- **FR-015**: System MUST support batch processing of multiple science frames with shared calibration frames (one set of bias/flat/arc per observation sequence)
- **FR-016**: System MUST provide configurable parameters via YAML file with sensible KOSMOS defaults; required parameters: detector.gain, detector.readnoise, detector.saturate; optional user overrides: trace_position_hint, extraction.aperture_width, wavelength.polynomial_order_range, min_snr_threshold, sky_buffer_pixels (see contracts/config-schema.yaml for complete parameter specification)
- **FR-017**: System MUST generate quality metrics for each output spectrum including: S/N estimate per resolution element, wavelength calibration RMS, spatial trace profile consistency
- **FR-018**: System MUST implement tiered error handling: halt immediately for critical failures (corrupt FITS files, wrong instrument configuration, missing required calibrations); for quality issues (low SNR, partial arc line coverage, cosmic ray contamination) produce outputs with quality flags, warning messages, and degraded-quality indicators in logs

### Key Entities

- **RawFrame**: Represents a single FITS file from telescope with metadata (observation type, target name, exposure time, timestamp, instrument configuration)
- **BiasFrame**: Calibration frame capturing detector readout bias pattern, median-combined from multiple exposures
- **FlatFrame**: Calibration frame capturing pixel-to-pixel sensitivity variations and illumination pattern, normalized to unity
- **ArcFrame**: Calibration frame containing emission line spectrum for wavelength calibration with identified line positions
- **ScienceFrame**: 2D spectral image of astronomical target after bias subtraction and flat fielding
- **Trace**: Spatial profile of a spectrum on the detector with attributes (center position vs wavelength, width, flux profile, trace ID)
- **Spectrum1D**: Extracted one-dimensional spectrum with arrays (wavelength, flux, error) and metadata (extraction method, aperture, S/N, calibration quality)
- **WavelengthSolution**: Mapping from detector pixel to wavelength with polynomial coefficients, arc line identifications, RMS residual, and valid wavelength range
- **ProcessingLog**: Record of all reduction stages applied to data with timestamps, input/output files, parameters, validation results, and quality flags

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Astronomer can process a typical night's KOSMOS observations (10 science frames + calibrations) from raw data to wavelength-calibrated 1D spectra in under 30 minutes of compute time on standard workstation
- **SC-002**: Pipeline successfully extracts 1D spectra from faint galaxy observations with continuum flux SNR ≥ 3 per spatial pixel (spatially averaged across extraction aperture in continuum regions, excluding emission lines), where existing pyKOSMOS trace identification fails (tested on benchmark dataset of 20 faint galaxy exposures)
- **SC-003**: Wavelength calibration achieves RMS residuals < 0.2 Å for arc lamp fits across full spectral range (4000-7000 Å typical for KOSMOS), validated against known arc line catalogs (acceptance criterion; implementation targets <0.1 Å per config-schema.yaml)
- **SC-004**: Extracted 1D spectra preserve flux to within 5% compared to aperture photometry on 2D frames, demonstrating proper extraction and calibration
- **SC-005**: Pipeline validation checks catch 95% of problematic data in testing (corrupt FITS files, wrong instrument configuration, saturated calibrations) with informative error messages before producing invalid output
- **SC-006**: Astronomer following documentation can install pipeline, configure for their data, and successfully reduce their first dataset within 2 hours (including reading docs and troubleshooting)
- **SC-007**: Batch processing mode successfully handles multi-trace observations with correct association of calibrations to science frames, verified on multi-slit or multi-target datasets

## Assumptions

- Raw FITS files follow standard APO-KOSMOS header conventions with IMAGETYP, OBJECT, EXPTIME, DATE-OBS keywords
- Input data organized in separate directories by type: arcs/, flats/, biases/, science/
- Typical observing sequence includes sufficient calibration frames (≥5 bias frames for median combine, ≥3 flat frames, ≥2 arc frames per grating setup)
- Astronomer has basic Python environment and can install standard astronomy packages (astropy, specutils, scipy, pyyaml)
- Target spectra are longslit observations (not multi-fiber IFU data which would require different trace extraction)
- Wavelength coverage and spectral resolution are within typical KOSMOS specifications (~5Å resolution, 4000-10000Å range depending on grating)
- Computing resources available: modern workstation with ≥8GB RAM and multi-core CPU for parallel processing
- User can interact with pop-up viewer for trace selection when multiple traces detected
- Pipeline follows standard spectroscopic reduction workflow per Massey and PypeIt: **essential steps** (bias estimation, wavelength calibration with arcs, object finding, object extraction, background subtraction) executed by default; **optional steps** (flux calibration, telluric correction, AB differencing) executed only when configured or data provided
- AB difference imaging requires user specification of nod/dither pattern in YAML configuration (observing mode specific)
- pyKOSMOS reference resources (linelists, extinction curves, arc templates, standard star spectra) are available locally in resources/pykosmos_reference/ and used for wavelength calibration and optional flux/extinction corrections

## Out of Scope

- Real-time reduction at telescope (pipeline is post-observation analysis tool)
- Advanced sky subtraction for crowded fields beyond pyKOSMOS/PypeIt standard methods (basic sky background subtraction included)
- Fully automated trace selection without user confirmation (interactive viewer required for quality control)

**Note**: Flux calibration using spectrophotometric standard stars and telluric absorption correction are **optional steps within scope** (not out of scope), executed only when standard star observations are provided or explicitly requested in configuration. These are not automated/required by default for faint galaxy extraction.

- Database storage of reduced products (pipeline writes files to disk; archival system is separate concern)
- Fully non-interactive headless operation (interactive matplotlib trace viewer is required for quality control; batch mode auto-selects all traces but still requires display capability)
