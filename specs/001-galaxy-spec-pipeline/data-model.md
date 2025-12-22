# Phase 1: Data Model

**Feature**: 001-galaxy-spec-pipeline  
**Date**: 2025-12-22  
**Purpose**: Define all entities, attributes, relationships, and validation rules for KOSMOS pipeline

## Overview

This document defines the data model for the KOSMOS galaxy spectroscopy pipeline, derived from functional requirements in spec.md and design decisions in research.md. All entities use astropy-compatible types for scientific data handling.

## Core Entity Relationships

```
ObservationSet
├── RawFrame (abstract)
│   ├── BiasFrame
│   ├── FlatFrame
│   ├── ArcFrame
│   └── ScienceFrame
├── CalibrationSet
│   ├── MasterBias
│   ├── MasterFlat
│   └── BadPixelMask
└── ReducedData
    ├── Spectrum2D
    │   └── Trace (1 to many)
    │       ├── SpatialProfile
    │       └── WavelengthSolution
    └── Spectrum1D (extracted)
        └── QualityMetrics
```

---

## 1. RawFrame (Abstract Base)

**Description**: Base class for all raw FITS frames from KOSMOS detector.

**Attributes**:

| Attribute | Type | Units | Description | Validation |
| --------- | ---- | ----- | ----------- | ---------- |
| `file_path` | `pathlib.Path` | - | Absolute path to FITS file | Must exist, `.fits` extension |
| `data` | `astropy.nddata.CCDData` | `u.adu` | 2D pixel array with uncertainty | Shape (2048, 4096) for KOSMOS |
| `header` | `astropy.io.fits.Header` | - | FITS header metadata | Required keys: OBJECT, EXPTIME, DATE-OBS |
| `observation_date` | `astropy.time.Time` | - | UTC observation timestamp | Parsed from DATE-OBS header |
| `exposure_time` | `astropy.units.Quantity` | `u.second` | Exposure duration | > 0 seconds |
| `gain` | `float` | `u.electron / u.adu` | CCD gain | Default 1.4 e⁻/ADU (KOSMOS) |
| `readnoise` | `float` | `u.electron` | Read noise | Default 3.7 e⁻ (KOSMOS) |
| `saturate` | `float` | `u.adu` | Saturation level | Default 58982 ADU (90% of 16-bit) |

**Methods**:

- `from_fits(file_path: Path) -> RawFrame`: Factory method to load FITS file
- `validate_header() -> bool`: Check required header keys present
- `detect_saturation() -> Tuple[np.ndarray, float]`: Return saturation mask and fraction

**Relationships**:

- Belongs to one `ObservationSet`
- Used by `CalibrationSet` (for biases, flats, arcs)

---

## 2. BiasFrame

**Inherits**: `RawFrame`

**Description**: Bias exposure (0 second exposure) for electronic offset correction.

**Additional Attributes**:

| Attribute | Type | Units | Description | Validation |
| --------- | ---- | ----- | ----------- | ---------- |
| `bias_level` | `float` | `u.adu` | Median pixel value | Typical range 300-500 ADU |

**Additional Validation**:

- `exposure_time` must be 0.0 seconds
- `header['OBJECT']` must contain 'bias' (case-insensitive)

**Usage**: Combined into `MasterBias` via sigma-clipped median (research.md §7)

---

## 3. FlatFrame

**Inherits**: `RawFrame`

**Description**: Flat field exposure for pixel-to-pixel sensitivity correction.

**Additional Attributes**:

| Attribute | Type | Units | Description | Validation |
| --------- | ---- | ----- | ----------- | ---------- |
| `lamp_type` | `str` | - | Flat lamp type (quartz, tungsten) | Optional |
| `saturation_fraction` | `float` | - | Fraction of saturated pixels | < 0.01 recommended |

**Additional Validation**:

- `exposure_time` > 0 seconds
- `header['OBJECT']` must contain 'flat' (case-insensitive)
- Median counts should be 10,000-50,000 ADU (well-exposed range)

**Usage**: Combined into `MasterFlat` via median combine with normalization (Massey Ch. 3)

---

## 4. ArcFrame

**Inherits**: `RawFrame`

**Description**: Arc lamp exposure for wavelength calibration.

**Additional Attributes**:

| Attribute | Type | Units | Description | Validation |
| --------- | ---- | ----- | ----------- | ---------- |
| `lamp_type` | `str` | - | Arc lamp (HeNeAr, Argon, Krypton, ThAr, CuAr) | Must be in LAMP_KEYWORD_MAP (research.md §8) |
| `linelist_file` | `pathlib.Path` | - | Path to reference linelist | Must exist in resources/pykosmos_reference/linelists/ |

**Additional Validation**:

- `exposure_time` > 0 seconds
- `lamp_type` automatically detected from filename or header (research.md §8)
- At least 20 unsaturated arc lines required for wavelength fit

**Usage**: Identifies arc lines to fit `WavelengthSolution` (research.md §5)

---

## 5. ScienceFrame

**Inherits**: `RawFrame`

**Description**: Science target exposure (galaxy spectrum).

**Additional Attributes**:

| Attribute | Type | Units | Description | Validation |
| --------- | ---- | ----- | ----------- | ---------- |
| `target_name` | `str` | - | Galaxy/object name | Non-empty |
| `ra` | `astropy.coordinates.Angle` | `u.degree` | Right ascension | 0-360° |
| `dec` | `astropy.coordinates.Angle` | `u.degree` | Declination | -90 to +90° |
| `airmass` | `float` | - | Airmass at observation | > 1.0 |
| `nod_position` | `str` | - | Nod position (A, B, or None) | Optional for AB subtraction |

**Additional Validation**:

- `exposure_time` > 0 seconds
- If `nod_position` is 'A' or 'B', must have matching pair for AB subtraction

**Usage**: Primary data for spectral extraction (FR-004)

---

## 6. CalibrationSet

**Description**: Container for combined calibration frames.

**Attributes**:

| Attribute | Type | Units | Description | Validation |
| --------- | ---- | ----- | ----------- | ---------- |
| `master_bias` | `MasterBias` | - | Combined bias frame | Required |
| `master_flat` | `MasterFlat` | - | Combined flat field | Required |
| `bad_pixel_mask` | `np.ndarray (bool)` | - | Mask of bad/hot/dead pixels | Shape matches detector |
| `creation_date` | `astropy.time.Time` | - | When calibrations generated | - |

**Methods**:

- `apply_to_frame(raw_frame: RawFrame) -> CCDData`: Bias-subtract, flat-correct, mask bad pixels
- `validate() -> bool`: Check all calibrations present and compatible

**Relationships**:

- Created from multiple `BiasFrame` and `FlatFrame` objects
- Applied to each `ScienceFrame` and `ArcFrame`

---

## 7. MasterBias

**Description**: Combined bias frame from multiple bias exposures.

**Attributes**:

| Attribute | Type | Units | Description | Validation |
| --------- | ---- | ----- | ----------- | ---------- |
| `data` | `astropy.nddata.CCDData` | `u.adu` | Combined bias data | Shape (2048, 4096) |
| `n_combined` | `int` | - | Number of bias frames combined | ≥ 5 recommended |
| `bias_level` | `float` | `u.adu` | Median pixel value | - |
| `bias_stdev` | `float` | `u.adu` | Standard deviation across frames | < 10 ADU recommended |
| `provenance` | `List[Path]` | - | List of input bias files | Non-empty |

**Creation Method**: Sigma-clipped median combine (research.md §7)

---

## 8. MasterFlat

**Description**: Combined, normalized flat field frame.

**Attributes**:

| Attribute | Type | Units | Description | Validation |
| --------- | ---- | ----- | ----------- | ---------- |
| `data` | `astropy.nddata.CCDData` | - | Normalized flat field (median=1.0) | Shape (2048, 4096) |
| `n_combined` | `int` | - | Number of flat frames combined | ≥ 3 recommended |
| `normalization_region` | `Tuple[slice, slice]` | - | Region used for normalization | Central 50% of detector |
| `bad_pixel_fraction` | `float` | - | Fraction of pixels < 0.5 or > 1.5 | < 0.05 acceptable |
| `provenance` | `List[Path]` | - | List of input flat files | Non-empty |

**Creation Method**: Median combine bias-subtracted flats, normalize by median (Massey Ch. 3)

---

## 9. Spectrum2D

**Description**: Calibrated 2D spectrum (science frame after bias/flat/cosmic ray correction).

**Attributes**:

| Attribute | Type | Units | Description | Validation |
| --------- | ---- | ----- | ----------- | ---------- |
| `data` | `astropy.nddata.CCDData` | `u.electron` | Calibrated 2D pixel array | Shape (2048, 4096) |
| `variance` | `np.ndarray` | `u.electron**2` | Variance array | Non-negative |
| `mask` | `np.ndarray (bool)` | - | Bad pixel mask | Shape matches data |
| `source_frame` | `ScienceFrame` | - | Original science frame | Required |
| `traces` | `List[Trace]` | - | Detected spectral traces | 0-10 traces typical |
| `cosmic_ray_mask` | `np.ndarray (bool)` | - | Detected cosmic rays | - |

**Methods**:

- `detect_traces(min_snr: float = 3.0) -> List[Trace]`: Find traces via cross-correlation (research.md §1)
- `subtract_sky(trace: Trace, spatial_buffer: int = 30) -> None`: Subtract median sky away from trace
- `extract_spectrum(trace: Trace, extraction_width: int = 10) -> Spectrum1D`: Extract 1D spectrum

**Relationships**:

- Created from one `ScienceFrame` after calibration
- Contains multiple `Trace` objects (1-10 typical)

---

## 10. Trace

**Description**: Single spectral trace in 2D spectrum (e.g., one galaxy or serendipitous source).

**Attributes**:

| Attribute | Type | Units | Description | Validation |
| --------- | ---- | ----- | ----------- | ---------- |
| `trace_id` | `int` | - | Unique identifier within Spectrum2D | > 0 |
| `spatial_positions` | `np.ndarray` | `u.pixel` | Trace center vs. spectral pixel | Length = n_spectral_pixels |
| `spectral_pixels` | `np.ndarray` | `u.pixel` | Spectral pixel coordinates | Length = n_spectral_pixels |
| `snr_estimate` | `float` | - | Signal-to-noise ratio estimate | > 0 |
| `spatial_profile` | `SpatialProfile` | - | Profile model for extraction | Required |
| `wavelength_solution` | `WavelengthSolution` | - | λ(pixel) mapping | Required for calibration |
| `user_selected` | `bool` | - | User confirmed in interactive viewer | Default False |

**Methods**:

- `fit_profile(continuum_mask: np.ndarray) -> SpatialProfile`: Fit Gaussian profile (research.md §4)
- `apply_wavelength_solution(wave_sol: WavelengthSolution) -> None`: Assign wavelengths
- `extract_optimal(profile: SpatialProfile, variance: np.ndarray) -> Spectrum1D`: Optimal extraction (Horne 1986)

**Relationships**:

- Belongs to one `Spectrum2D`
- Has one `SpatialProfile` and one `WavelengthSolution`
- Extracted into one `Spectrum1D`

---

## 11. SpatialProfile

**Description**: Spatial profile model for optimal spectral extraction.

**Attributes**:

| Attribute | Type | Units | Description | Validation |
| --------- | ---- | ----- | ----------- | ---------- |
| `profile_type` | `str` | - | Model type (Gaussian, Moffat, Empirical) | Must be in ['Gaussian', 'Moffat', 'Empirical'] |
| `center` | `float` | `u.pixel` | Profile center position | Within detector bounds |
| `width` | `float` | `u.pixel` | Profile FWHM (Gaussian σ * 2.355) | 1-20 pixels typical |
| `amplitude` | `float` | `u.electron` | Peak amplitude | > 0 |
| `profile_function` | `Callable[[np.ndarray], np.ndarray]` | - | Evaluates profile at positions | Returns non-negative values |
| `chi_squared` | `float` | - | Goodness-of-fit statistic | Lower is better |

**Creation Method**: Fit Gaussian to spatial collapse of continuum regions (research.md §4), excluding emission lines

**Validation**:

- If χ² > threshold (10.0), fall back to empirical profile (research.md §4)
- Profile must integrate to finite value

---

## 12. WavelengthSolution

**Description**: Polynomial fit mapping pixel position to wavelength.

**Attributes**:

| Attribute | Type | Units | Description | Validation |
| --------- | ---- | ----- | ----------- | ---------- |
| `coefficients` | `np.ndarray` | - | Chebyshev polynomial coefficients | Length = order + 1 |
| `order` | `int` | - | Polynomial order (3-7 typical) | 3 ≤ order ≤ 7 |
| `arc_frame` | `ArcFrame` | - | Source arc lamp exposure | Required |
| `n_lines_identified` | `int` | - | Number of arc lines used in fit | ≥ 20 recommended |
| `rms_residual` | `float` | `u.Angstrom` | RMS wavelength residual | < 0.1 Å (FR-008) |
| `wavelength_range` | `Tuple[float, float]` | `u.Angstrom` | (λ_min, λ_max) coverage | Depends on grating |

**Methods**:

- `wavelength(pixel: np.ndarray) -> astropy.units.Quantity`: Evaluate λ(pixel) using Chebyshev polynomial
- `inverse(wavelength: astropy.units.Quantity) -> np.ndarray`: Find pixel(λ) via root-finding
- `validate() -> bool`: Check RMS < 0.1 Å and residuals plotted

**Creation Method**: Chebyshev polynomial fit with iterative sigma-clipping (research.md §5)

**Relationships**:

- Fitted from one `ArcFrame`
- Applied to all `Trace` objects in same observation

---

## 13. Spectrum1D

**Description**: Extracted, wavelength-calibrated 1D spectrum.

**Inherits**: `specutils.Spectrum1D`

**Additional Attributes**:

| Attribute | Type | Units | Description | Validation |
| --------- | ---- | ----- | ----------- | ---------- |
| `spectral_axis` | `astropy.units.Quantity` | `u.Angstrom` | Wavelength array | Monotonically increasing |
| `flux` | `astropy.units.Quantity` | `u.electron / u.second` | Flux density | Non-negative |
| `uncertainty` | `astropy.nddata.StdDevUncertainty` | `u.electron / u.second` | 1σ flux uncertainty | Non-negative |
| `meta` | `dict` | - | Metadata (target name, airmass, etc.) | - |
| `trace_id` | `int` | - | Source trace ID | Matches Trace.trace_id |
| `extraction_method` | `str` | - | Extraction type (optimal, boxcar, AB_subtracted) | - |
| `quality_metrics` | `QualityMetrics` | - | Quality assessment | Required |

**Methods**:

- `apply_extinction_correction(extinction_curve: Path) -> None`: Correct for atmospheric extinction (FR-009)
- `flux_calibrate(standard_spectrum: Spectrum1D) -> None`: Apply flux calibration from standard star
- `bin_spectral(bin_width: astropy.units.Quantity) -> Spectrum1D`: Rebin to specified width (research.md §6)

**Relationships**:

- Extracted from one `Trace` in one `Spectrum2D`
- Has one `QualityMetrics` object

---

## 14. QualityMetrics

**Description**: Automated quality assessment metrics for reduced spectrum.

**Attributes**:

| Attribute | Type | Units | Description | Validation |
| --------- | ---- | ----- | ----------- | ---------- |
| `median_snr` | `float` | - | Median SNR in continuum regions | > 3 recommended |
| `wavelength_rms` | `float` | `u.Angstrom` | Wavelength solution RMS | < 0.1 Å required (FR-008) |
| `sky_residual_rms` | `float` | `u.electron / u.second` | RMS of sky subtraction residuals | Lower is better |
| `cosmic_ray_fraction` | `float` | - | Fraction of pixels flagged as CRs | < 0.01 typical |
| `saturation_flag` | `bool` | - | Any pixels saturated in extraction | False preferred |
| `ab_subtraction_quality` | `float` | - | AB pair alignment quality (0-1) | > 0.9 good |
| `overall_grade` | `str` | - | Quality grade (Excellent, Good, Fair, Poor) | Based on thresholds |

**Validation Thresholds**:

- **Excellent**: SNR > 10, RMS < 0.05 Å, no saturation
- **Good**: SNR > 5, RMS < 0.1 Å, no saturation
- **Fair**: SNR > 3, RMS < 0.15 Å
- **Poor**: SNR ≤ 3 or RMS ≥ 0.15 Å

**Methods**:

- `compute(spectrum: Spectrum1D, wavelength_solution: WavelengthSolution) -> QualityMetrics`: Calculate all metrics
- `generate_report() -> str`: Human-readable quality summary

**Relationships**:

- Computed for each `Spectrum1D` (FR-010)

---

## 15. PipelineConfig

**Description**: User-configurable pipeline parameters loaded from YAML file.

**Attributes**:

| Attribute | Type | Units | Description | Default Value |
| --------- | ---- | ----- | ----------- | ------------- |
| `gain` | `float` | `u.electron / u.adu` | CCD gain | 1.4 (KOSMOS) |
| `readnoise` | `float` | `u.electron` | Read noise | 3.7 (KOSMOS) |
| `saturate` | `float` | `u.adu` | Saturation level | 58982 (90% of 16-bit) |
| `pixel_scale` | `float` | `u.arcsecond / u.pixel` | Spatial pixel scale | 0.29 (KOSMOS) |
| `trace_detection_sigma` | `float` | - | SNR threshold for trace detection | 3.0 |
| `trace_profile_fwhm` | `float` | `u.pixel` | Expected trace FWHM | 5.0 (2" seeing) |
| `extraction_width` | `int` | `u.pixel` | Extraction aperture half-width | 10 |
| `sky_buffer` | `int` | `u.pixel` | Distance from trace for sky regions | 30 |
| `wavelength_poly_order` | `int` | - | Polynomial order for wavelength fit | 5 |
| `wavelength_rms_threshold` | `float` | `u.Angstrom` | Maximum acceptable RMS residual | 0.1 |
| `binning_spatial_factor` | `int` | - | Spatial binning (pre-extraction) | 1 (no binning) |
| `binning_spectral_width` | `float` | `u.Angstrom` | Spectral binning (post-extraction) | 2.0 |
| `ab_subtraction_iterations` | `int` | - | Max iterations for AB subtraction | 5 |
| `ab_convergence_threshold` | `float` | - | RMS change for convergence | 0.01 |

**Methods**:

- `from_yaml(file_path: Path) -> PipelineConfig`: Load config from YAML file
- `validate() -> List[str]`: Check all parameters in valid ranges
- `to_yaml(file_path: Path) -> None`: Save config to YAML file

**Relationships**:

- Used by all pipeline stages (calibration, extraction, wavelength, flux)

---

## 16. ObservationSet

**Description**: Collection of all raw frames from a single observing night or target.

**Attributes**:

| Attribute | Type | Units | Description | Validation |
| --------- | ---- | ----- | ----------- | ---------- |
| `observation_date` | `astropy.time.Time` | - | Night of observation | - |
| `target_name` | `str` | - | Galaxy/object name | Non-empty |
| `bias_frames` | `List[BiasFrame]` | - | All bias exposures | ≥ 5 recommended |
| `flat_frames` | `List[FlatFrame]` | - | All flat field exposures | ≥ 3 recommended |
| `arc_frames` | `List[ArcFrame]` | - | All arc lamp exposures | ≥ 1 required |
| `science_frames` | `List[ScienceFrame]` | - | All science target exposures | ≥ 1 required |
| `calibration_set` | `CalibrationSet` | - | Combined calibrations | Created from bias/flat frames |

**Methods**:

- `from_directory(input_dir: Path) -> ObservationSet`: Discover and load all FITS files
- `group_ab_pairs() -> List[Tuple[ScienceFrame, ScienceFrame]]`: Match A-B nod pairs
- `validate_completeness() -> bool`: Check all required calibrations present

**Relationships**:

- Contains all `RawFrame` objects (biases, flats, arcs, science)
- Produces one `CalibrationSet`
- Produces multiple `ReducedData` objects (one per science frame)

---

## 17. ReducedData

**Description**: Container for all products from reducing a single science frame.

**Attributes**:

| Attribute | Type | Units | Description | Validation |
| --------- | ---- | ----- | ----------- | ---------- |
| `source_frame` | `ScienceFrame` | - | Original science exposure | Required |
| `spectrum_2d` | `Spectrum2D` | - | Calibrated 2D spectrum | Required |
| `spectra_1d` | `List[Spectrum1D]` | - | Extracted 1D spectra (one per trace) | ≥ 1 |
| `diagnostic_plots` | `Dict[str, Path]` | - | Paths to diagnostic plots | Keys: '2d_spectrum', 'traces', 'wavelength_fit', 'sky_subtraction', 'extraction_profile' |
| `processing_log` | `str` | - | Log of all processing steps | Non-empty |
| `reduction_timestamp` | `astropy.time.Time` | - | When reduction completed | - |

**Methods**:

- `save_to_disk(output_dir: Path) -> None`: Write all products to directory
- `generate_summary_report() -> str`: Create human-readable reduction summary

**Relationships**:

- Created from one `ScienceFrame` and one `CalibrationSet`
- Contains multiple `Spectrum1D` objects (one per `Trace`)

---

## 18. InteractiveSelection

**Purpose**: Encapsulates state for interactive trace selection viewer, enabling user to visually inspect detected traces and select which to process

**Attributes**:

- `spectrum_2d` (Spectrum2D): The 2D spectrum being displayed
- `detected_traces` (list[Trace]): All traces detected by pipeline
- `selected_trace_ids` (list[int]): Trace IDs selected by user
- `matplotlib_figure` (Figure): Matplotlib figure object for interactive viewer
- `matplotlib_widgets` (dict): Dictionary of CheckButtons and Button widgets for interaction

**Methods**:

- `show() -> list[int]`: Display interactive viewer with 2D spectrum and trace overlays, block until user clicks Accept, return selected trace IDs (or empty list if canceled)
- `on_trace_toggle(trace_id: int)`: Callback when user toggles trace checkbox, updates selected_trace_ids list and trace line alpha transparency
- `on_accept()`: Callback when user clicks Accept button, closes figure and returns control to pipeline

**Relationships**:

- References Spectrum2D and list of Trace objects for display
- Used by PipelineRunner in interactive mode (mode='interactive') after trace detection
- Returns filtered list of selected_trace_ids for extraction

**Validation**:

- `spectrum_2d` must be valid Spectrum2D with data and variance
- `detected_traces` must be non-empty list (if empty, no interactive selection needed)
- `matplotlib_figure` must be valid Figure with axes configured per research.md §3 layout

---

## 19. ProcessingLog

**Purpose**: Record of all reduction stages applied to data with timestamps, input/output files, parameters, validation results, and quality flags for full provenance tracking per Constitution Principle III

**Attributes**:

- `timestamps` (list[datetime]): Timestamps for each processing stage
- `input_files` (list[str]): Paths to all input FITS files used in reduction
- `output_files` (list[str]): Paths to all output files generated
- `parameters` (dict): All configuration parameters and runtime settings used
- `validation_results` (list[dict]): Results from validation checks at each stage (pass/fail, metrics, thresholds)
- `quality_flags` (list[str]): Warning/error flags raised during processing
- `software_versions` (dict): Versions of key packages (astropy, specutils, scipy, pipeline version)
- `processing_stages` (list[str]): Names of completed reduction stages in execution order

**Methods**:

- `add_log_entry(stage: str, timestamp: datetime, parameters: dict, validation: dict)`: Add entry for completed processing stage
- `add_quality_flag(flag: str, severity: str)`: Add warning or error flag with severity (CRITICAL/WARNING/INFO)
- `save_to_disk(filepath: str)`: Write log to JSON or YAML file with timestamps
- `load_from_disk(filepath: str) -> ProcessingLog`: Load existing log from file
- `generate_summary() -> str`: Return human-readable summary of processing history

**Relationships**:

- Referenced by ReducedData (processing_log attribute) per data-model.md §17
- Written to logs/ directory for each reduction run per FR-011, FR-012
- Used by quality validation to track stage-by-stage metrics per FR-010

**Validation**:

- `timestamps` must be monotonically increasing (stages logged in execution order)
- `input_files` must all exist and be readable FITS files
- `validation_results` entries must include stage name, timestamp, and pass/fail status
- `software_versions` must include astropy, specutils, scipy, pipeline version

---

## Data Flow Summary

---

## Data Flow Summary

1. **Ingestion**: `ObservationSet.from_directory()` discovers raw FITS files → creates `BiasFrame`, `FlatFrame`, `ArcFrame`, `ScienceFrame` objects
2. **Calibration**: Combine bias/flat frames → `MasterBias`, `MasterFlat` in `CalibrationSet`
3. **Preprocessing**: Apply calibrations to `ScienceFrame` → `Spectrum2D` with cosmic ray cleaning
4. **Trace Detection**: `Spectrum2D.detect_traces()` → list of `Trace` objects
5. **Interactive Selection**: `InteractiveSelection.show()` → user-selected subset of traces
6. **Profile Fitting**: For each trace, fit `SpatialProfile` from continuum regions (excluding emission lines)
7. **Wavelength Calibration**: Fit `WavelengthSolution` from `ArcFrame` → apply to all traces
8. **Extraction**: `Trace.extract_optimal()` → `Spectrum1D` with flux, wavelength, uncertainty
9. **Quality Assessment**: Compute `QualityMetrics` for each `Spectrum1D`
10. **Output**: `ReducedData.save_to_disk()` → FITS files, diagnostic plots, processing logs

---

## Validation Rules Summary

### Data Quality Gates

1. **Bias Combination**: ≥ 5 bias frames, bias level variation < 10 ADU
2. **Flat Combination**: ≥ 3 flat frames, median counts 10k-50k ADU, saturation < 1%
3. **Arc Wavelength Fit**: ≥ 20 arc lines identified, RMS residual < 0.1 Å, polynomial order 3-7
4. **Trace Detection**: SNR > 3.0, profile FWHM 1-20 pixels
5. **Spectrum Extraction**: At least 1 trace user-selected, no saturated pixels in aperture
6. **Overall Quality**: Median SNR > 3, wavelength RMS < 0.1 Å (FR-008, FR-010)

### Constitution Compliance

- **Physics-Validated Packages** (Principle I): All CCDData uses `astropy.nddata`, all Spectrum1D uses `specutils`, all wavelength fits use `astropy.units`
- **Spectroscopy Standards** (Principle II): Wavelength solution follows Chebyshev polynomial convention, optimal extraction follows Horne 1986
- **Data Provenance** (Principle III): All frames track `provenance` list, all processing logged
- **Modular Architecture** (Principle IV): Each entity maps to one module (io/, calibration/, extraction/, wavelength/, quality/)
- **Scientific Validation** (Principle V): `QualityMetrics` enforces RMS < 0.1 Å and SNR thresholds
- **Learning Resources** (Principle VI): Data model aligns with Massey Ch. 2-7 entities

---

## Next Steps

Proceed to **Phase 1 Contracts**:

- Generate [contracts/cli-spec.yaml](./contracts/cli-spec.yaml)
- Generate [contracts/config-schema.yaml](./contracts/config-schema.yaml)
