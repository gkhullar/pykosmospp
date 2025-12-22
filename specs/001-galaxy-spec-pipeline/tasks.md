# Tasks: APO-KOSMOS Galaxy Spectroscopy Pipeline

**Feature**: 001-galaxy-spec-pipeline  
**Input**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/](./contracts/)

---

## Task Format: `- [ ] [ID] [P?] [Story?] Description`

- **Checkbox**: `- [ ]` for incomplete, `- [x]` for complete
- **[ID]**: Sequential task number (T001, T002, ...)
- **[P]**: Parallelizable (different files, no dependencies on incomplete tasks)
- **[Story]**: User story label (US1, US2, US3, US4, US5) - only for story-specific tasks
- **Description**: Clear action with exact file path

---

## Phase 0: Constitution Gates (Pre-Implementation)

**Purpose**: Verify constitution compliance before any code implementation

- [x] T000 [P] Catalog available learning resources per constitution.md §VI: Scan parent directory ../notebooks/ and ../resources/ for relevant materials (wavelength calibration examples, flux calibration workflows, trace extraction patterns, validation datasets); create .specify/memory/learning-resources-catalog.md documenting available notebooks and their use cases for reference during implementation

---

## Phase 1: Setup (Project Initialization)

**Purpose**: Create project structure and foundational configuration

- [x] T001 Create project directory structure per plan.md (src/, tests/, resources/, config/, docs/)
- [x] T002 [P] Initialize pyproject.toml with package metadata, dependencies (astropy≥5.3, specutils≥1.10, scipy≥1.10, numpy≥1.23, matplotlib≥3.6, pyyaml≥6.0), and CLI entry point
- [x] T003 [P] Create src/__init__.py to define package namespace
- [x] T004 [P] Create README.md with installation instructions, quickstart, and references to spec.md and quickstart.md
- [x] T005 [P] Copy resources/pykosmos_reference/ linelists, extinction curves, arc templates, and onedstds to project resources directory (already downloaded)
- [x] T006 [P] Create config/kosmos_defaults.yaml from contracts/config-schema.yaml with KOSMOS detector defaults (gain=1.4, readnoise=3.7, saturate=58982)
- [x] T007 [P] Setup pytest configuration in pyproject.toml with astropy-pytest-plugins
- [x] T008 [P] Create .gitignore for Python (__pycache__, *.pyc, .pytest_cache, *.fits, reduced_*)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story implementation

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Data Models (Foundation)

- [x] T009 [P] Create src/models.py with RawFrame abstract base class per data-model.md §1 (file_path, data as CCDData, header, observation_date, exposure_time, gain, readnoise, saturate attributes; from_fits(), validate_header(), detect_saturation() methods)
- [x] T010 [P] Add BiasFrame class to src/models.py per data-model.md §2 (inherits RawFrame, adds bias_level, validates exposure_time=0)
- [x] T011 [P] Add FlatFrame class to src/models.py per data-model.md §3 (inherits RawFrame, adds lamp_type, saturation_fraction)
- [x] T012 [P] Add ArcFrame class to src/models.py per data-model.md §4 (inherits RawFrame, adds lamp_type, linelist_file, arc lamp detection logic from research.md §8)
- [x] T013 [P] Add ScienceFrame class to src/models.py per data-model.md §5 (inherits RawFrame, adds target_name, ra, dec, airmass, nod_position)
- [x] T014 [P] Add CalibrationSet class to src/models.py per data-model.md §6 (master_bias, master_flat, bad_pixel_mask, apply_to_frame(), validate() methods)
- [x] T015 [P] Add MasterBias class to src/models.py per data-model.md §7 (data, n_combined, bias_level, bias_stdev, provenance)
- [x] T016 [P] Add MasterFlat class to src/models.py per data-model.md §8 (data, n_combined, normalization_region, bad_pixel_fraction, provenance)

### I/O Foundation

- [x] T017 [P] Create src/io/__init__.py module
- [x] T018 Create src/io/fits.py with FITS reading functions (read_fits_as_ccddata(), validate_fits_header(), write_fits_with_provenance()) using astropy.io.fits and astropy.nddata.CCDData
- [x] T019 [P] Create src/io/config.py with YAML configuration loading (PipelineConfig.from_yaml(), validate() per data-model.md §15) using pyyaml
- [x] T020 [P] Create src/io/organize.py with output directory structure creation (create_output_dirs() for calibrations/, reduced_2d/, spectra_1d/, wavelength_solutions/, plots/, logs/)

### Logging and Error Handling Foundation

- [x] T021 [P] Create src/io/logging.py with logger configuration (setup_logger(), log_processing_step(), log_validation_result()) supporting --verbose/--quiet modes per contracts/cli-spec.yaml
- [x] T022 [P] Add error handling classes to src/io/logging.py (CriticalPipelineError for halting failures, QualityWarning for flagged outputs per FR-018)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Push-Button Reduction (Priority: P1) 🎯 MVP

**Goal**: Process typical night's KOSMOS observations (10 science frames + calibrations) from raw data to wavelength-calibrated 1D spectra with minimal interaction

**Independent Test**: Provide complete observation set (science + calibrations), verify pipeline produces 1D spectra files with wavelength solutions and quality metrics without user intervention at each stage

### Implementation for User Story 1

#### Calibration Module (US1)

- [x] T023 [P] [US1] Create src/calibration/__init__.py module
- [x] T024 [US1] Implement src/calibration/combine.py with sigma_clipped_median_combine() function per research.md §7 (uses astropy.stats.sigma_clipped_stats, handles bias and flat frame stacks)
- [x] T025 [US1] Implement src/calibration/bias.py with create_master_bias() function (calls combine.py, creates MasterBias object, validates bias_level variation <10 ADU per data-model.md §7)
- [x] T026 [US1] Implement src/calibration/flat.py with create_master_flat() function (bias-subtracts flats, combines, normalizes to median=1.0, creates MasterFlat object, validates bad_pixel_fraction <0.05 per data-model.md §8)
- [x] T027 [US1] Implement src/calibration/cosmic.py with detect_cosmic_rays() function using scipy-based LA Cosmic algorithm (sigma_clip=5.0, contrast=3.0, max_iterations=5 per config-schema.yaml)

#### I/O Module Completion (US1)

- [x] T028 [US1] Add ObservationSet class to src/models.py per data-model.md §16 (observation_date, target_name, bias_frames, flat_frames, arc_frames, science_frames, calibration_set; from_directory(), group_ab_pairs(), validate_completeness() methods)
- [x] T029 [US1] Implement src/io/organize.py discover_fits_files() function (scans input directories arcs/, flats/, biases/, science/, classifies files by FITS header IMAGETYP keyword per FR-002)

#### Wavelength Calibration Module (US1)

- [X] T030 [P] [US1] Create src/wavelength/__init__.py module
- [X] T031 [US1] Add WavelengthSolution class to src/models.py per data-model.md §12 (coefficients, order, arc_frame, n_lines_identified, rms_residual, wavelength_range; wavelength(), inverse(), validate() methods)
- [X] T032 [US1] Implement src/wavelength/identify.py with detect_arc_lines() function (scipy.signal.find_peaks with detection_threshold=5.0 sigma, returns pixel positions and intensities per research.md §5); **Constitution §VI**: Consult notebooks/ for arc line detection examples before implementation
- [X] T033 [US1] Implement src/wavelength/match.py with match_lines_to_catalog() function (loads pyKOSMOS linelists from resources/pykosmos_reference/linelists/, matches detected lines to catalog within tolerance=2.0 Å per config-schema.yaml)
- [X] T034 [US1] Implement src/wavelength/fit.py with fit_wavelength_solution() function per research.md §5 (Chebyshev polynomial with iterative sigma-clipping, BIC order selection 3-7, validates RMS <0.1 Å per FR-008); **Constitution §VI**: Consult notebooks/ for wavelength fitting workflows before implementation
- [X] T035 [US1] Implement src/wavelength/apply.py with apply_wavelength_to_spectrum() function (evaluates Chebyshev polynomial at pixel positions, creates wavelength axis for Spectrum1D)

#### Trace Detection and Extraction Module (US1)

- [X] T036 [P] [US1] Create src/extraction/__init__.py module
- [X] T037 [US1] Add Spectrum2D class to src/models.py per data-model.md §9 (data, variance, mask, source_frame, traces, cosmic_ray_mask; detect_traces(), subtract_sky(), extract_spectrum() methods)
- [X] T038 [US1] Add Trace class to src/models.py per data-model.md §10 (trace_id, spatial_positions, spectral_pixels, snr_estimate, spatial_profile, wavelength_solution, user_selected; fit_profile(), apply_wavelength_solution(), extract_optimal() methods)
- [X] T039 [US1] Implement src/extraction/trace.py with detect_traces_cross_correlation() function per research.md §1 (scipy.ndimage.correlate1d with Gaussian kernel, scipy.signal.find_peaks, excludes emission lines from profile, returns list of Trace objects with min_snr=3.0)
- [X] T040 [US1] Add SpatialProfile class to src/models.py per data-model.md §11 (profile_type, center, width, amplitude, profile_function, chi_squared)
- [X] T041 [US1] Implement src/extraction/profile.py with fit_spatial_profile() function per research.md §4 (scipy.optimize.curve_fit Gaussian profile on continuum regions, masks emission lines via gradient threshold, fallback to empirical profile if chi_sq >10.0)
- [X] T042 [US1] Implement src/extraction/sky.py with estimate_sky_background() function per research.md (median of spatial regions away from trace, sky_buffer=30 pixels, sigma_clip=3.0, broadcasts to 2D for subtraction)
- [X] T043 [US1] Implement src/extraction/extract.py with extract_optimal() function (Horne 1986 optimal extraction using spatial profile weights, propagates variance per data-model.md §13, aperture_width=10 pixels from config)

#### Quality Assessment Module (US1)

- [X] T044 [P] [US1] Create src/quality/__init__.py module
- [X] T045 [US1] Add QualityMetrics class to src/models.py per data-model.md §14 (median_snr, wavelength_rms, sky_residual_rms, cosmic_ray_fraction, saturation_flag, ab_subtraction_quality, overall_grade; compute(), generate_report() methods)
- [X] T046 [US1] Implement src/quality/validate.py with validate_calibrations() function (checks bias level variation, flat normalization, saturation fraction per FR-010)
- [X] T047 [US1] Implement src/quality/metrics.py with compute_quality_metrics() function (calculates SNR in continuum regions, wavelength RMS from solution, sky residual RMS, spatial profile_consistency_score from chi-squared of Gaussian/Moffat fit per data-model.md §11, assigns grade Excellent/Good/Fair/Poor per data-model.md §14)
- [X] T048 [US1] Implement src/quality/plots.py with LaTeX plot generation functions per research.md §10 (setup_latex_plots(), plot_2d_spectrum(), plot_wavelength_residuals(), plot_extraction_profile(), plot_sky_subtraction() using matplotlib with text.usetex=True, saves to plots/ directory as PDF)

#### Pipeline Orchestration (US1)

- [X] **T049**: Add `ReducedData` class to `models.py`
- [X] **T050**: Create `src/pipeline.py` with `PipelineRunner` class (input_dir, output_dir, config, mode attributes; run() method orchestrates all stages per plan.md)
- [X] **T051**: Implement `PipelineRunner.run()` workflow orchestration
- [X] T052 [US1] Add error handling to PipelineRunner.run() per FR-018 (critical failures raise CriticalPipelineError and halt, quality issues log QualityWarning and continue with flagged outputs)

#### CLI Entry Point (US1)

- [X] **T053**: Create `src/cli.py` with main `kosmos-reduce` command per contracts/cli-spec.yaml (argparse with --input-dir, --output-dir, --config, --mode, --verbose, --log-file, --validate-only, --overwrite, --max-traces options)
- [X] T054 [US1] Implement CLI subcommand: kosmos-reduce calibrate (generates master bias/flat only, outputs to calibrations/)
- [X] T055 [US1] Add CLI exit codes to src/cli.py per contracts/cli-spec.yaml (0=success, 1=missing calibrations, 2=invalid FITS/config, 3=wavelength solution failed, 4=no traces detected, 5=user canceled)
- [X] **T056**: Wire up CLI to `PipelineRunner` in `main()` function (loads config, creates runner, executes run(), handles exceptions, returns appropriate exit code)

**Checkpoint**: At this point, User Story 1 should be fully functional - complete automated pipeline from raw data to wavelength-calibrated 1D spectra in batch mode

---

## Phase 4: User Story 2 - Faint Trace Identification (Priority: P2)

**Goal**: Reliably identify and extract faint extended galaxy traces (SNR ~ 3-5) using adaptive algorithms with interactive user selection

**Independent Test**: Synthetic or real faint galaxy data with known traces at various SNR levels, verify extraction success rate vs. traditional peak-finding

### Implementation for User Story 2

#### Interactive Trace Viewer (US2)

- [ ] T057 [P] [US2] Create src/extraction/interactive.py module
- [ ] T058 [US2] Add InteractiveSelection class to src/models.py per data-model.md §18 (spectrum_2d, detected_traces, selected_trace_ids, matplotlib_figure, matplotlib_widgets; show(), on_trace_toggle(), on_accept() methods)
- [ ] T059 [US2] Implement InteractiveSelection.show() method per research.md §3 (displays 2D spectrum with matplotlib, overlays traces as colored lines, adds CheckButtons for trace selection, Button widgets for Accept/Cancel, blocks until user clicks Accept, returns selected_trace_ids)
- [ ] T060 [US2] Implement InteractiveSelection layout per research.md §3 (main axis: 2D spectrum with log-scale colormap viridis, right sidebar: checkboxes for traces, bottom: spatial profile plot, LaTeX-styled labels)
- [ ] T061 [US2] Add interactive viewer callback InteractiveSelection.on_trace_toggle() (toggles trace line alpha transparency 0.2/0.7, updates selected_trace_ids list)
- [ ] T062 [US2] Add interactive viewer callback InteractiveSelection.on_accept() (closes figure, returns control to pipeline)

#### Enhanced Trace Detection (US2)

- [ ] T063 [US2] Enhance src/extraction/trace.py detect_traces_cross_correlation() to support multiple template widths per research.md §1 (try expected_fwhm from config, also try expected_fwhm ± 2 pixels, merge detections, return all with SNR >min_snr)
- [ ] T064 [US2] Add trace quality metadata to Trace objects in src/extraction/trace.py (snr_estimate, profile_consistency_score, detection_method)
- [ ] T065 [US2] Implement trace deduplication in src/extraction/trace.py (merge detections closer than min_separation=20 pixels, keep highest SNR)
- [ ] T065b [US2] Implement faint trace fallback in src/extraction/trace.py per spec edge case "Trace too faint for identification": if SNR <min_snr after detection, log warning with 3 options: (a) use manual trace_position_hint from config if provided, (b) offer spatial binning via config binning.spatial.enabled=True to boost SNR, (c) skip frame with detailed warning message; document fallback behavior in function docstring

#### Pipeline Integration with Interactive Mode (US2)

- [ ] T066 [US2] Modify PipelineRunner.run() in src/pipeline.py to check config mode: if mode='interactive', call InteractiveSelection.show() after detect_traces() and filter traces to user-selected subset before extraction
- [ ] T067 [US2] Add --mode interactive flag handling to src/cli.py (default to interactive per contracts/cli-spec.yaml, set matplotlib backend if needed)
- [ ] T068 [US2] Handle user cancellation in interactive viewer (InteractiveSelection.show() can return empty list, pipeline exits with code 5 if canceled per contracts/cli-spec.yaml)

**Checkpoint**: User Story 2 complete - pipeline now supports interactive trace selection for faint galaxies with visual confirmation

---

## Phase 5: User Story 3 - Robust Wavelength Calibration (Priority: P2)

**Goal**: Accurate wavelength solutions for galaxy spectra even when arc lamp lines are faint or partially missing

**Independent Test**: Compare pipeline wavelength solutions against known arc line positions and reference spectra, measure RMS residuals across detector

### Implementation for User Story 3

#### Advanced Wavelength Fitting (US3)

- [ ] T069 [P] [US3] Implement src/wavelength/fit.py BIC order selection per research.md §5 (try polynomial orders 3-7, compute BIC for each, select minimum BIC, ensure RMS <0.1 Å threshold)
- [ ] T070 [US3] Add iterative sigma-clipping to src/wavelength/fit.py fit_wavelength_solution() (max 5 iterations, sigma=3.0, track rejected outliers, require ≥80% lines retained per research.md §5)
- [ ] T071 [US3] Add wavelength solution diagnostics to src/wavelength/fit.py (residuals vs. wavelength array, identified line table with pixel/wavelength/residual/intensity, final polynomial order used)

#### Arc Lamp Line Identification (US3)

- [ ] T072 [US3] Enhance src/wavelength/identify.py detect_arc_lines() with saturation detection per research.md §9 (call detect_saturation() from RawFrame, exclude saturated lines from fit, log warning if >1% saturated)
- [ ] T073 [US3] Add robust peak-finding to src/wavelength/identify.py (use scipy.signal.find_peaks with prominence threshold, filter peaks by intensity >detection_threshold sigma above continuum)
- [ ] T074 [US3] Implement src/wavelength/match.py initial wavelength range estimation (use FITS header grating angle/keyword if available, else use config wavelength_range [3500, 7500] Å default for KOSMOS per config-schema.yaml)

#### CLI Wavelength Subcommand (US3)

- [ ] T075 [US3] Implement CLI subcommand: kosmos-reduce wavelength per contracts/cli-spec.yaml (--input-arc, --output, --lamp-type, --poly-order, --rms-threshold, --min-lines, --plot options)
- [ ] T076 [US3] Add wavelength subcommand output files to src/cli.py (wavelength_solution.fits as FITS table, wavelength_fit.pdf diagnostic plot, identified_lines.txt line table)

#### Enhanced Quality Validation (US3)

- [ ] T077 [US3] Add wavelength-specific validation to src/quality/validate.py (validate_wavelength_solution(): checks RMS <0.1 Å per FR-008, n_lines ≥20 recommended, polynomial order reasonable 3-7)
- [ ] T078 [US3] Generate wavelength residuals plot in src/quality/plots.py plot_wavelength_residuals() per research.md §10 (top panel: wavelength vs pixel with Chebyshev fit, bottom panel: residuals with ±0.1 Å threshold lines, LaTeX labels)

**Checkpoint**: User Story 3 complete - wavelength calibration robust to missing/saturated arc lines with detailed diagnostics

---

## Phase 6: User Story 4 - Quality Assessment & Validation (Priority: P3)

**Goal**: Confidence that each reduction stage produces valid results with clear diagnostics when problems occur

**Independent Test**: Introduce problematic data (bad pixels, cosmic rays, incorrect file types), verify pipeline catches issues with informative error messages

### Implementation for User Story 4

#### Comprehensive Validation Framework (US4)

- [ ] T079 [P] [US4] Expand src/quality/validate.py with validate_bias_frames() function (checks bias_level in reasonable range 300-500 ADU, bias_stdev <10 ADU, n_combined ≥5 per data-model.md §7)
- [ ] T080 [P] [US4] Expand src/quality/validate.py with validate_flat_frames() function (checks median counts 10k-50k ADU well-exposed range, saturation_fraction <0.01, bad_pixel_fraction <0.05 per data-model.md §8)
- [ ] T081 [P] [US4] Add src/quality/validate.py validate_spectrum2d() function (checks cosmic_ray_fraction <0.01, saturation in extraction aperture flagged, trace profile consistency per data-model.md §14)
- [ ] T082 [P] [US4] Add src/quality/validate.py validate_spectrum1d() function (checks flux conservation vs 2D aperture photometry within 5% per SC-004, median_snr >min_snr threshold per data-model.md §14)

#### Diagnostic Plotting Suite (US4)

- [ ] T083 [P] [US4] Implement src/quality/plots.py plot_2d_spectrum() per research.md §10 (log-scale colormap, LaTeX axes labels, colorbar, saves to plots/)
- [ ] T084 [P] [US4] Implement src/quality/plots.py plot_traces_overlay() (2D spectrum with all detected traces overlaid as colored lines, trace IDs labeled)
- [ ] T085 [P] [US4] Implement src/quality/plots.py plot_extraction_profile() (spatial profile data vs Gaussian/Moffat fit, residuals, chi-squared labeled)
- [ ] T086 [P] [US4] Implement src/quality/plots.py plot_sky_subtraction() (sky regions highlighted, median sky vs spectral pixel, sky residuals after subtraction)

#### Quality Report Generation (US4)

- [ ] T087 [US4] Implement QualityMetrics.generate_report() in src/models.py per data-model.md §14 (returns formatted string with all metrics, thresholds, pass/fail status, overall grade)
- [ ] T088 [US4] Add ReducedData.generate_summary_report() in src/models.py (aggregates quality metrics for all traces, lists input files, calibration quality, processing time, saved output paths)
- [ ] T089 [US4] Implement quality report writing in PipelineRunner.run() (calls generate_summary_report(), writes to logs/quality_report.txt, logs warnings for any Poor grade spectra)

#### Enhanced Error Messages (US4)

- [ ] T090 [US4] Add actionable error messages to src/io/logging.py (CriticalPipelineError includes: stage failed, criterion violated, suggested remediation per spec User Story 4 scenario 2)
- [ ] T091 [US4] Add validation stage checkpoints to PipelineRunner.run() (after calibrations created, after wavelength solution fit, after extraction, log pass/fail status before proceeding)

**Checkpoint**: User Story 4 complete - comprehensive quality validation with diagnostics at every stage

---

## Phase 7: User Story 5 - Multi-Trace Batch Processing (Priority: P3)

**Goal**: Process all traces efficiently with consistent calibrations for multi-object spectroscopy or multiple targets per night

**Independent Test**: Multi-slit or multi-target observations, verify each trace gets correct calibrations without cross-contamination

### Implementation for User Story 5

#### AB Subtraction for Nod Patterns (US5)

- [ ] T092 [P] [US5] Create src/extraction/ab_subtract.py module
- [ ] T093 [US5] Implement iterative_ab_subtraction() function per research.md §2 (estimates sky in A and B frames independently, sigma-clips to exclude object flux, iterates until convergence <1% RMS change or max 5 iterations, handles spatial misalignment)
- [ ] T094 [US5] Add AB pair matching to ObservationSet.group_ab_pairs() in src/models.py per data-model.md §16 (matches by nod_position='A'/'B' in FITS header, or by observation time proximity <10 minutes per config-schema.yaml)
- [ ] T095 [US5] Integrate AB subtraction into PipelineRunner.run() (if config ab_subtraction.enabled=True and AB pairs detected, call iterative_ab_subtraction() before trace extraction per FR-006)

#### Multi-Trace Processing (US5)

- [ ] T096 [US5] Enhance PipelineRunner.run() to process all detected traces per frame (loop over traces, extract each independently, assign unique trace_id, save separate spectrum_1d_trace{N}.fits files per contracts/cli-spec.yaml output structure)
- [ ] T097 [US5] Add trace metadata tracking to ReducedData in src/models.py (list of all trace_ids processed, which traces user-selected in interactive mode, SNR per trace)
- [ ] T098 [US5] Implement batch mode processing in src/cli.py (if --mode batch, auto-select all detected traces without interactive viewer, process entire ObservationSet)

#### Calibration Association (US5)

- [ ] T099 [US5] Implement calibration timestamp matching in ObservationSet.validate_completeness() per spec User Story 5 scenario 2 (associates calibrations to science frames by observation_date proximity, warns if time gap >1 hour)
- [ ] T100 [US5] Add instrumental configuration validation to src/quality/validate.py (checks FITS header grating angle, slit width match between calibrations and science frames, raises CriticalPipelineError if mismatch per spec edge case)

#### Multi-Target Output Organization (US5)

- [ ] T101 [US5] Enhance src/io/organize.py to create per-target subdirectories (if multiple target_name values in ScienceFrame set, create output_dir/target1/, output_dir/target2/, each with calibrations/, reduced_2d/, spectra_1d/, logs/)
- [ ] T102 [US5] Add master processing log to PipelineRunner (writes output_dir/master_log.txt tracking which calibrations applied to which science frames per spec User Story 5 scenario 3)

#### CLI Combine Subcommand (US5)

- [ ] T103 [US5] Implement CLI subcommand: kosmos-reduce combine per contracts/cli-spec.yaml (--input-spectra, --output, --method, --wavelength-grid, --resolution options for combining multiple 1D spectra)
- [ ] T104 [US5] Implement combine logic in src/wavelength/apply.py (resamples spectra to common wavelength grid, median/mean/weighted combines with variance propagation, uses specutils.manipulation functions)

**Checkpoint**: User Story 5 complete - multi-trace and multi-target batch processing fully supported

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Enhancements affecting multiple user stories, final testing, documentation

- [X] T105 [P] Add spatial binning support to src/extraction/extract.py per research.md §6 (bin_spatial() function, pre-extraction binning if config binning.spatial.enabled=True, factor=2 default)
- [X] T106 [P] Add spectral binning support to src/wavelength/apply.py per research.md §6 (bin_spectral() function using specutils.manipulation.bin_function, post-extraction binning if config binning.spectral.enabled=True, width_angstrom=2.0 default)
- [X] T107 [P] Implement flux calibration module src/flux_calibration/extinction.py (apply_extinction_correction() using pyKOSMOS apoextinct.dat per FR-009, optional step if config flux_calibration.extinction.enabled=True); **Constitution §VI**: Consult notebooks/ for flux calibration workflows before implementation
- [X] T108 [P] Implement flux calibration module src/flux_calibration/sensitivity.py (compute_sensitivity_function() from standard star observations, optional step if standard_star_file provided per FR-009); **Constitution §VI**: Consult notebooks/ for sensitivity function examples before implementation
- [X] T109 [P] Add extraction method selection to src/extraction/extract.py (implement extract_boxcar() simple aperture sum as alternative to optimal extraction, selectable via config extraction.method='optimal' or 'boxcar' per contracts/config-schema.yaml)
- [X] T110 [P] Implement src/extraction/profile.py Moffat profile fitting as alternative to Gaussian per research.md §4 (scipy.optimize.curve_fit with Moffat function, selectable via config spatial_profile.profile_type='Moffat')
- [X] T111 [P] Add cosmic ray detection integration to PipelineRunner.run() (call detect_cosmic_rays() from src/calibration/cosmic.py after flat correction, before trace detection, save cosmic_ray_mask in Spectrum2D per data-model.md §9)
- [X] T112 [P] Implement uncertainty propagation throughout pipeline (read noise + Poisson noise → variance arrays in CCDData, propagate through calibrations/extraction to Spectrum1D.uncertainty per FR-014)
- [ ] T113 Create tests/fixtures/synthetic_data.py to generate synthetic KOSMOS FITS files per quickstart.md (kosmos-generate-testdata command: bias, flat, arc with HeNeAr lines, science with 1-2 galaxy traces at configurable SNR); **Note**: This is a test utility, not a production CLI feature
- [ ] T114 Create tests/integration/test_pipeline_e2e.py end-to-end test (runs full pipeline on synthetic data, validates outputs exist, checks quality metrics meet thresholds per SC-001 through SC-007)
- [ ] T115 [P] Create tests/unit/test_calibration.py (tests combine, bias, flat, cosmic modules independently with synthetic data)
- [ ] T116 [P] Create tests/unit/test_extraction.py (tests trace detection, profile fitting, sky subtraction, optimal extraction with known inputs)
- [ ] T117 [P] Create tests/unit/test_wavelength.py (tests arc line identification, line matching, polynomial fitting, wavelength application with pyKOSMOS linelists)
- [ ] T118 [P] Create tests/unit/test_quality.py (tests validation functions, metrics computation, plot generation functions)
- [ ] T119 Update README.md with comprehensive quickstart per quickstart.md test scenarios (installation, synthetic test data generation, basic reduction, interpreting outputs)
- [ ] T120 [P] Add docstrings to all public functions and classes (numpy-style docstrings with Parameters, Returns, Examples sections)
- [ ] T121 [P] Create docs/API.md documenting Python API for programmatic use per contracts/cli-spec.yaml (PipelineRunner, PipelineConfig classes, example usage)
- [ ] T122 Run quickstart.md Test Scenario 1 validation (generate synthetic data, run kosmos-reduce, verify outputs per quickstart.md expected performance <30 minutes for 10 frames)
- [ ] T123 Run quickstart.md Test Scenario 2 validation (faint galaxy with SNR~3.5, verify detection and extraction succeed with custom config)
- [ ] T124 Run quickstart.md Test Scenario 3 validation (custom wavelength calibration with Krypton lamp, verify RMS <0.05 Å)
- [ ] T125 Run quickstart.md Test Scenario 4 validation (AB nod pair, verify iterative sky subtraction converges and sky lines removed)
- [ ] T126 Run quickstart.md Test Scenario 5 validation (degraded data quality, verify pipeline warnings and quality report correctly identify issues)

---

## Phase 9: Documentation & Tutorials

**Goal**: Create comprehensive documentation and interactive tutorials for users and developers

**Test Criteria**: Documentation builds without errors, tutorial notebook executes successfully, Read the Docs site is live

### Jupyter Tutorial Notebook

- [X] T127 [P] Create examples/tutorial.ipynb notebook structure with 8 sections (Introduction, Data Exploration, Calibration, Wavelength, Extraction, Quality Assessment, Advanced Parameters, Batch Processing)
- [X] T128 Create examples/tutorial.ipynb Section 1: Introduction & Setup (imports, check installation, load configuration, display KOSMOS detector specs)
- [X] T129 [P] Create examples/tutorial.ipynb Section 2: Data Exploration (load sample FITS files, display headers, visualize raw bias/flat/arc/science frames with matplotlib)
- [X] T130 [P] Create examples/tutorial.ipynb Section 3: Calibration Creation (demonstrate create_master_bias, create_master_flat, validate_calibrations, show combined frames)
- [X] T131 [P] Create examples/tutorial.ipynb Section 4: Wavelength Calibration (detect arc lines, match to catalog, fit Chebyshev solution, plot residuals, demonstrate BIC order selection)
- [X] T132 [P] Create examples/tutorial.ipynb Section 5: Trace Detection & Extraction (detect traces with cross-correlation, visualize on 2D spectrum, fit spatial profiles, perform optimal extraction)
- [X] T133 [P] Create examples/tutorial.ipynb Section 6: Quality Assessment (compute quality metrics, display quality report, show diagnostic plots, interpret grades)
- [X] T134 [P] Create examples/tutorial.ipynb Section 7: Advanced - Custom Parameters (modify config, adjust trace detection sensitivity, change polynomial order, compare results)
- [X] T135 [P] Create examples/tutorial.ipynb Section 8: Batch Processing (process multiple objects, organize outputs, generate summary statistics)
- [ ] T136 Add examples/data/ with small synthetic test dataset (3 bias, 3 flat, 1 arc, 1 science frame, total <50 MB for notebook execution)
- [X] T137 Add examples/README.md explaining tutorial notebook, how to run, expected outputs, troubleshooting common issues

### Sphinx Documentation Setup

- [X] T138 [P] Create docs/ directory structure (source/, build/, Makefile, make.bat, requirements.txt)
- [X] T139 Create docs/source/conf.py Sphinx configuration (project metadata, extensions=[sphinx.ext.autodoc, sphinx.ext.napoleon, sphinx.ext.viewcode, sphinx_rtd_theme], paths, version)
- [X] T140 [P] Create docs/source/index.rst main documentation page (project overview, feature highlights, quick links, table of contents)
- [X] T141 Create docs/source/installation.rst (system requirements, pip install, conda environment, development setup, troubleshooting)
- [X] T142 [P] Create docs/source/quickstart.rst (5-minute guide: install, generate test data, run pipeline, view outputs)

### User Guide Documentation

- [X] T143 [P] Create docs/source/user_guide/cli.rst (complete CLI reference: main command, subcommands, options, exit codes, examples)
- [X] T144 [P] Create docs/source/user_guide/python_api.rst (PipelineRunner usage, module imports, programmatic access, code examples)
- [X] T145 [P] Create docs/source/user_guide/configuration.rst (YAML config file structure, all parameters explained, detector settings, algorithm tuning)
- [X] T146 [P] Create docs/source/user_guide/output_products.rst (directory structure, FITS file formats, quality reports, diagnostic plots, metadata)

### Tutorial Documentation

- [ ] T147 [P] Create docs/source/tutorials/basic_reduction.rst (step-by-step first reduction, interpret outputs, common pitfalls)
- [ ] T148 [P] Create docs/source/tutorials/faint_galaxies.rst (optimizing for low SNR, trace detection tuning, extraction parameters)
- [ ] T149 [P] Create docs/source/tutorials/wavelength_calibration.rst (arc lamp types, custom linelists, improving RMS, troubleshooting poor fits)
- [ ] T150 [P] Create docs/source/tutorials/quality_validation.rst (interpreting quality metrics, diagnostic plots, identifying problems, reprocessing)

### API Reference Documentation

- [X] T151 [P] Create docs/source/api/calibration.rst (autodoc for calibration module: combine, bias, flat, cosmic modules)
- [X] T152 [P] Create docs/source/api/wavelength.rst (autodoc for wavelength module: identify, match, fit, apply)
- [X] T153 [P] Create docs/source/api/extraction.rst (autodoc for extraction module: trace, profile, sky, extract)
- [X] T154 [P] Create docs/source/api/quality.rst (autodoc for quality module: validate, metrics, plots)
- [X] T155 [P] Create docs/source/api/models.rst (autodoc for data models: RawFrame hierarchy, calibration classes, spectroscopic data)

### Algorithm Documentation

- [X] T156 [P] Create docs/source/algorithms/trace_detection.rst (cross-correlation method, emission masking, centroid tracing, references to research.md §4)
- [X] T157 [P] Create docs/source/algorithms/wavelength_fitting.rst (Chebyshev polynomials, BIC model selection, sigma-clipping, normalization, references to research.md §8)
- [X] T158 [P] Create docs/source/algorithms/optimal_extraction.rst (Horne 1986 algorithm, variance propagation, aperture fallback, references to research.md §5)
- [X] T159 [P] Create docs/source/algorithms/cosmic_ray_detection.rst (L.A.Cosmic method, parameters, performance, references to research.md §9)

### Documentation Infrastructure

- [X] T160 Create docs/source/troubleshooting.rst (common errors, solutions, FAQ, performance optimization, getting help)
- [X] T161 Create docs/requirements.txt (sphinx, sphinx_rtd_theme, sphinx-autodoc-typehints, dependencies for building docs)
- [X] T162 Create .readthedocs.yaml configuration file (Python version, install dependencies, build commands)
- [X] T163 Add docs build check to CI/CD (GitHub Actions: build Sphinx docs, check for warnings/errors, upload artifacts)
- [X] T164 Register project on readthedocs.org and configure webhook (link GitHub repo, set build settings, enable PR previews) - Setup guide created in docs/READTHEDOCS_SETUP.md
- [X] T165 Update README.md with documentation links (link to Read the Docs site, badge, quick navigation to key sections)
- [X] T166 Create docs/source/contributing.rst (development setup, constitution principles, testing, pull request process)
- [X] T167 Add example notebook to documentation (nbsphinx integration to render tutorial.ipynb in Sphinx docs)

---

## Dependencies & Execution Order

### Phase Dependencies

1. **Setup (Phase 1)**: No dependencies - start immediately
2. **Foundational (Phase 2)**: Depends on Setup - BLOCKS all user stories
3. **User Story 1 (Phase 3)**: Depends on Foundational - MVP implementation
4. **User Story 2 (Phase 4)**: Depends on Foundational + US1 trace extraction infrastructure
5. **User Story 3 (Phase 5)**: Depends on Foundational + US1 wavelength calibration infrastructure
6. **User Story 4 (Phase 6)**: Depends on Foundational + US1 quality module infrastructure
7. **User Story 5 (Phase 7)**: Depends on Foundational + US1 full pipeline + US2 interactive viewer
8. **Polish (Phase 8)**: Depends on all desired user stories complete

### User Story Independence

- **US1 (P1)**: Fully independent after Foundational phase - complete automated pipeline
- **US2 (P2)**: Extends US1 trace detection with interactive viewer - can proceed in parallel with US3/US4 after US1 extraction infrastructure exists
- **US3 (P2)**: Extends US1 wavelength calibration - can proceed in parallel with US2/US4 after US1 wavelength infrastructure exists
- **US4 (P3)**: Extends US1 quality validation - can proceed in parallel with US2/US3 after US1 quality infrastructure exists
- **US5 (P3)**: Depends on US1 complete, integrates US2 interactive viewer - requires most prior work

### Within Each Phase

**Setup (Phase 1)**:
- All tasks T001-T008 can run in parallel (marked [P])

**Foundational (Phase 2)**:
- T009-T016 (data models) can run in parallel [P]
- T017-T020 (I/O foundation) can run in parallel [P], T018 before T019-T020
- T021-T022 (logging) can run in parallel [P]

**User Story 1 (Phase 3)**:
- Calibration: T024 → T025, T026 in parallel
- Wavelength: T032-T035 sequential per algorithm flow
- Extraction: T039 → T041 → T042, T043
- Quality: T046-T048 in parallel [P] after T045
- Pipeline: T050 → T051 → T052 → T053-T056

**User Story 2-5**: Dependencies shown with → in task descriptions, [P] tasks can parallelize within phase

### Parallel Opportunities per User Story

**User Story 1 Parallel Groups**:

```bash
# Can run simultaneously:
Group A: T023, T030, T036, T044 (module __init__ files)
Group B: T024-T027 (calibration functions)
Group C: T031-T035 (wavelength functions) - sequential within group
Group D: T037-T043 (extraction functions) - some sequential dependencies
Group E: T045-T048 (quality functions)
```

**User Story 2 Parallel Groups**:

```bash
# Can run simultaneously after US1 infrastructure:
Group A: T057, T063, T064 (interactive + enhanced trace detection)
```

**User Story 3 Parallel Groups**:

```bash
# Can run simultaneously after US1 infrastructure:
Group A: T069-T074 (enhanced wavelength algorithms)
```

---

## Implementation Strategy

### Recommended Delivery Sequence

1. **Week 1**: Phase 1 Setup + Phase 2 Foundational (T001-T022)
   - Creates project structure, data models, I/O foundation
   - Enables all user story work to begin

2. **Week 2-3**: Phase 3 User Story 1 (T023-T056) 🎯 **MVP Milestone**
   - Complete automated pipeline end-to-end
   - Delivers push-button reduction capability
   - Demonstrates core value proposition

3. **Week 4**: Phase 4 User Story 2 (T057-T068)
   - Adds interactive trace selection
   - Critical for faint galaxy science case

4. **Week 5**: Phase 5 User Story 3 (T069-T078)
   - Enhances wavelength calibration robustness
   - Critical for accurate redshift measurements

5. **Week 6**: Phase 6 User Story 4 (T079-T091)
   - Comprehensive validation framework
   - Essential for production scientific use

6. **Week 7**: Phase 7 User Story 5 (T092-T104)
   - Multi-trace and AB subtraction support
   - Expands capability to advanced observations

7. **Week 8**: Phase 8 Polish (T105-T126)
   - Testing, documentation, optional features
   - Ensures robustness and usability

### MVP Scope (Minimum Viable Product)

**Deliver User Story 1 Only (Phase 1-3)** for initial release:
- Complete project setup (T001-T008)
- Full foundational infrastructure (T009-T022)
- Automated push-button pipeline (T023-T056)
- **Result**: Working pipeline that processes typical KOSMOS night from raw data to 1D spectra in <30 minutes (SC-001) ✓

**Defer to Post-MVP**:
- Interactive trace viewer (US2)
- Advanced wavelength fitting (US3)
- Enhanced diagnostics (US4)
- Multi-trace/AB support (US5)

This delivers **core scientific capability** (automated reduction of faint galaxies) while establishing **architecture for iteration**.

---

## Total Task Count

- **Setup**: 8 tasks
- **Foundational**: 14 tasks
- **User Story 1 (MVP)**: 34 tasks
- **User Story 2**: 12 tasks
- **User Story 3**: 10 tasks
- **User Story 4**: 13 tasks
- **User Story 5**: 13 tasks
- **Polish**: 22 tasks

**Total**: 126 tasks

**MVP Subset** (US1 only): 56 tasks (Setup + Foundational + US1)

---

## Validation Checkpoints

After each phase, verify:

1. **Setup**: Can run `pip install -e .` and `kosmos-reduce --version`
2. **Foundational**: Can import all data models, load config, read/write FITS
3. **US1**: Can process synthetic test data end-to-end, outputs valid 1D spectra
4. **US2**: Interactive viewer displays traces, user can select/deselect
5. **US3**: Wavelength solution fits with RMS <0.1 Å on test arcs
6. **US4**: Quality validation catches intentional errors, generates plots
7. **US5**: Multi-target batch processing completes, AB subtraction converges
8. **Polish**: All quickstart.md scenarios pass, documentation complete

---

**Ready for implementation**: All tasks defined with clear dependencies and file paths. Begin with Phase 1 Setup tasks T001-T008.
