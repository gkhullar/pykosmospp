# Implementation Plan: APO-KOSMOS Galaxy Spectroscopy Pipeline

**Branch**: `001-galaxy-spec-pipeline` | **Date**: 2025-12-22 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-galaxy-spec-pipeline/spec.md`

## Summary

Build a spectroscopic reduction pipeline for APO-KOSMOS longslit observations that extracts faint galaxy spectra with robust wavelength calibration. The pipeline improves upon existing pyKOSMOS by implementing interactive trace selection, iterative AB subtraction, customizable spatial/spectral binning, and comprehensive quality validation at each reduction stage. Technical approach follows Massey/PypeIt methodology using physics-validated packages (astropy, specutils, scipy) with matplotlib-based interactive visualizations for trace selection and quality assessment.

## Technical Context

**Language/Version**: Python 3.10+ (required for astropy ≥5.3, specutils ≥1.10 compatibility)
**Primary Dependencies**:

- astropy ≥5.3 (FITS I/O, CCDData, units, coordinates)
- specutils ≥1.10 (Spectrum1D, spectral manipulation, line fitting)
- scipy ≥1.10 (signal processing, optimization, interpolation)
- numpy ≥1.23 (array operations, statistical functions)
- matplotlib ≥3.6 (LaTeX-styled plots, interactive trace viewer)
- pyyaml ≥6.0 (YAML configuration parsing)

**Storage**: File-based (FITS standard); Input: arcs/, flats/, biases/, science/; Output: calibrations/, reduced_2d/, spectra_1d/, logs/ per galaxy subdirectory
**Testing**: pytest with astropy-pytest-plugins (fixture support for FITS data); unit tests for each reduction module; integration tests with synthetic and real KOSMOS data
**Target Platform**: Linux/macOS workstation (≥8GB RAM, multi-core CPU for parallel processing)
**Project Type**: Single Python package with CLI entry point
**Performance Goals**:

- Process typical night's observations (10 science frames + calibrations) in <30 minutes
- Interactive trace viewer responds to user input in <1 second
- Handle 2048×2048 pixel FITS frames with <500MB memory per frame

**Constraints**:

- Must preserve FITS header provenance through all stages (FR-003, FR-011)
- Wavelength calibration RMS <0.2 Å acceptance criterion (SC-003), <0.1 Å implementation target for typical data quality (FR-008)
- Tiered error handling: critical failures halt; quality issues produce flagged outputs (FR-018)
- Interactive matplotlib viewer required for trace selection (FR-005)

**Scale/Scope**:

- ~2000-3000 LOC for core pipeline modules
- Support for 3-4 KOSMOS grating configurations (Blue/Red, varying resolutions)
- Process 10-50 science frames per batch run
- Handle 1-10 traces per 2D spectrum

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
| --------- | ------ | ----- |
| **I. Physics-Validated Packages** | ✅ PASS | Uses astropy (FITS, CCDData), specutils (Spectrum1D), scipy (signal processing), numpy, matplotlib per constitution mandate |
| **II. Spectroscopy Standards Alignment** | ✅ PASS | Follows pyKOSMOS/PypeIt workflow patterns; uses standard terminology (arc lamps, trace extraction, wavelength solution); references pyKOSMOS notebooks for patterns |
| **III. Data Provenance & Reproducibility** | ✅ PASS | FR-011 requires logging all steps with timestamps; FR-012 specifies organized output structure; FR-016 enables YAML parameter preservation |
| **IV. Modular Pipeline Architecture** | ✅ PASS | Specification defines independent stages (calibration, extraction, wavelength, quality); each operates on standard data structures (CCDData, Spectrum1D) |
| **V. Scientific Validation** | ✅ PASS | FR-010 mandates physics-based validation checks (bias level, flat normalization, flux conservation); FR-017 requires quality metrics per output |
| **VI. Learning Resources** | ✅ PASS | References Massey CCD reductions guide (resources/massey_ccd_reductions.pdf), pyKOSMOS notebooks (interactive_trace_example.ipynb), PypeIt documentation, and local pyKOSMOS resources |

**Decision**: All constitution gates pass. Proceed to Phase 0 research.

## Project Structure

### Documentation (this feature)

```text
specs/001-galaxy-spec-pipeline/
├── plan.md              # This file
├── research.md          # Phase 0: Algorithm research
├── data-model.md        # Phase 1: Entity definitions
├── quickstart.md        # Phase 1: Test scenarios
├── contracts/           # Phase 1: CLI and config schemas
│   ├── cli-spec.yaml
│   └── config-schema.yaml
├── checklists/
│   ├── requirements.md  # Already complete
│   └── clarification-coverage.md  # Already complete
└── spec.md              # Feature specification
```

### Source Code (repository root)

```text
pykosmos_spec_ai/
├── src/
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── bias.py          # Bias frame combination and subtraction
│   │   ├── flat.py          # Flat field normalization and application
│   │   ├── cosmic.py        # Cosmic ray rejection (LA Cosmic)
│   │   └── combine.py       # Frame combining utilities (median, sigma-clip)
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── trace.py         # Trace identification and modeling
│   │   ├── interactive.py   # Matplotlib interactive trace viewer
│   │   ├── profile.py       # Spatial profile fitting (Gaussian, Moffat)
│   │   ├── extract.py       # Optimal and aperture extraction
│   │   ├── sky.py           # Sky background estimation and subtraction
│   │   └── ab_subtract.py   # AB difference imaging (iterative solver)
│   ├── wavelength/
│   │   ├── __init__.py
│   │   ├── identify.py      # Arc line identification (peak-finding)
│   │   ├── match.py         # Line matching to reference linelists
│   │   ├── fit.py           # Wavelength solution fitting (polynomial, robust)
│   │   └── apply.py         # Apply wavelength calibration to spectra
│   ├── flux_calibration/    # Optional module
│   │   ├── __init__.py
│   │   ├── sensitivity.py   # Sensitivity function from standard stars
│   │   ├── extinction.py    # Atmospheric extinction correction
│   │   └── telluric.py      # Telluric absorption correction
│   ├── io/
│   │   ├── __init__.py
│   │   ├── fits.py          # FITS file reading/writing with provenance
│   │   ├── config.py        # YAML configuration parsing
│   │   └── organize.py      # Output directory structure management
│   ├── quality/
│   │   ├── __init__.py
│   │   ├── validate.py      # Physics-based validation checks
│   │   ├── metrics.py       # SNR, RMS, quality flag computation
│   │   └── plots.py         # LaTeX-styled matplotlib diagnostic plots
│   ├── models.py            # Data model classes (entities)
│   ├── pipeline.py          # Main pipeline orchestration
│   └── cli.py               # Command-line interface entry point
├── tests/
│   ├── unit/
│   │   ├── test_calibration.py
│   │   ├── test_extraction.py
│   │   ├── test_wavelength.py
│   │   └── test_quality.py
│   ├── integration/
│   │   ├── test_pipeline_e2e.py
│   │   └── test_interactive_viewer.py
│   └── fixtures/
│       ├── synthetic_data.py  # Generate synthetic KOSMOS frames
│       └── sample_data/        # Small real KOSMOS dataset
├── resources/
│   ├── pykosmos_reference/     # Downloaded pyKOSMOS resources
│   │   ├── linelists/
│   │   ├── extinction/
│   │   ├── arctemplates/
│   │   └── onedstds/
│   ├── massey_ccd_reductions.pdf
│   └── README.md
├── config/
│   └── kosmos_defaults.yaml   # Default KOSMOS configuration
├── pyproject.toml             # Package metadata and dependencies
├── README.md
└── .specify/                  # Specify workflow files
```

**Structure Decision**: Single Python package (Option 1) appropriate for scientific pipeline. Modular separation by reduction stage (calibration, extraction, wavelength, flux_calibration, quality) enables independent testing and aligns with Constitution Principle IV. CLI entry point in `src/cli.py` with pipeline orchestration in `src/pipeline.py`.

## Complexity Tracking

**No violations requiring justification.** Constitution Check passes all gates.

## Phase 0: Research & Algorithm Selection

See [research.md](./research.md) for detailed algorithm research and technology decisions.

**Key Research Areas**:

1. **Faint Trace Detection**: Algorithm selection for low surface brightness galaxies (cross-correlation vs. matched filter vs. adaptive thresholding)
2. **AB Subtraction**: Iterative/robust methods for nod-dither pattern subtraction with misalignment handling
3. **Interactive Visualization**: Matplotlib patterns from pyKOSMOS notebooks for trace selection UI
4. **Spatial Profile Modeling**: Gaussian vs. Moffat vs. empirical profiles for optimal extraction
5. **Wavelength Solution Fitting**: Polynomial order selection, robust fitting (sigma-clipping), residual diagnostics
6. **Customizable Binning**: Spatial and spectral binning strategies for SNR improvement without resolution loss

## Phase 1: Design

See design artifacts:

- [data-model.md](./data-model.md) - Entity definitions and relationships
- [contracts/](./contracts/) - CLI and configuration schemas
- [quickstart.md](./quickstart.md) - Test scenarios and usage examples

## Phase 2: Task Breakdown

Generated by `/speckit.tasks` command (not part of `/speckit.plan` output).

See [tasks.md](./tasks.md) after tasks command execution.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
