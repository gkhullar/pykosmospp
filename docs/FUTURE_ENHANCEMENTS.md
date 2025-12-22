# Future Enhancements (v1.0.0+)

This document tracks features and improvements planned for future releases of pyKOSMOS++.

**Current Version:** 0.1.0  
**Target Version:** 1.0.0+  
**Last Updated:** December 2025

---

## Validation Scenarios (Deferred from Phase 8)

These validation scenarios from `quickstart.md` were deferred from Phase 8 but should be completed for v1.0.0 release:

### T122: Synthetic Data Validation
- **Goal:** Validate pipeline performance on synthetic test data
- **Criteria:** Process 10 frames in <30 minutes
- **Status:** Deferred to acceptance testing phase
- **Priority:** High

### T123: Faint Galaxy Extraction (SNR~3.5)
- **Goal:** Test pipeline on challenging low-SNR targets
- **Criteria:** Successful trace detection and extraction at SNR threshold
- **Status:** Deferred
- **Priority:** High

### T124: Custom Krypton Wavelength Calibration
- **Goal:** Test alternative arc lamp linelists
- **Criteria:** RMS <0.05Å with Krypton lamp
- **Status:** Deferred
- **Priority:** Medium

### T125: AB Nod Pair Sky Subtraction
- **Goal:** Test iterative sky subtraction for nod pairs
- **Criteria:** Sky lines removed, convergence achieved
- **Status:** Deferred
- **Priority:** Medium

### T126: Degraded Data Quality Handling
- **Goal:** Verify pipeline warnings and quality reports on poor data
- **Criteria:** Correct issue identification and reporting
- **Status:** Deferred
- **Priority:** Medium

---

## Advanced Flux Calibration

### Telluric Absorption Correction
- **Description:** Correct for atmospheric absorption features (O2, H2O bands)
- **Method:** Model telluric spectrum or observe telluric standard stars
- **Dependencies:** High-resolution atmospheric transmission models
- **Priority:** Medium
- **Effort:** 2-3 weeks

### Multi-Order Sensitivity Fitting
- **Description:** Improved sensitivity function with higher-order polynomial fits
- **Method:** Spline interpolation or piecewise polynomial fitting
- **Benefits:** Better flux calibration across full wavelength range
- **Priority:** Low
- **Effort:** 1 week

### Standard Star Catalog Expansion
- **Description:** Add more spectrophotometric standards beyond current catalog
- **Options:** 
  - CALSPEC standards (HST)
  - ESO standards
  - SDSS standards
- **Priority:** Low
- **Effort:** 1 week (data collection + integration)

---

## Enhanced Trace Detection

### Multi-Object Spectroscopy (MOS)
- **Description:** Support for multiple slitlets/fibers
- **Features:**
  - Automatic slitlet detection
  - Independent wavelength solutions per slitlet
  - Batch extraction of all objects
- **Use Case:** Multi-slit masks, fiber spectrographs
- **Priority:** Medium
- **Effort:** 3-4 weeks

### Curved Trace Polynomial Fitting
- **Description:** Fit polynomial functions to curved spectral traces
- **Method:** 2D polynomial fit along dispersion axis
- **Benefits:** Better extraction for distorted spectra
- **Priority:** Low
- **Effort:** 1-2 weeks

### Automatic AB Nod Pair Detection
- **Description:** Automatically identify and pair AB nod observations
- **Features:**
  - Header-based pairing (NODBEAM keyword)
  - Position-based pairing (RA/DEC offsets)
  - Iterative sky subtraction
- **Priority:** Medium
- **Effort:** 2 weeks

---

## Interactive Visualization

### Web-Based Diagnostic Viewer
- **Description:** Interactive HTML/JavaScript interface for diagnostics
- **Features:**
  - Zoom/pan on 2D spectra
  - Interactive trace adjustment
  - Wavelength solution tuning
  - Real-time quality metrics
- **Technology:** Plotly Dash or Bokeh
- **Priority:** High
- **Effort:** 4-6 weeks

### Real-Time Quality Feedback
- **Description:** Live updates of quality metrics during reduction
- **Features:**
  - Progress bar with ETA
  - SNR estimates during extraction
  - Wavelength RMS convergence plots
- **Priority:** Medium
- **Effort:** 2 weeks

---

## Performance Optimizations

### GPU Acceleration
- **Description:** Use GPU for computationally intensive operations
- **Targets:**
  - Cross-correlation (trace detection)
  - 2D convolution (cosmic ray detection)
  - Optimal extraction matrix operations
- **Technology:** CuPy or PyTorch
- **Benefits:** 5-10x speedup on large datasets
- **Priority:** Low
- **Effort:** 4-6 weeks

### Parallel Processing
- **Description:** Process multiple science frames in parallel
- **Method:** Python multiprocessing or Dask
- **Benefits:** Near-linear speedup with CPU cores
- **Priority:** Medium
- **Effort:** 2 weeks

### Memory Optimization
- **Description:** Reduce memory footprint for large datasets
- **Methods:**
  - Memory-mapped FITS files
  - Chunked processing
  - On-disk intermediate products
- **Priority:** Low
- **Effort:** 2-3 weeks

---

## Advanced Sky Subtraction

### Iterative Background Modeling
- **Description:** Iteratively fit and subtract 2D sky background
- **Method:** B-spline surface fitting with sigma clipping
- **Benefits:** Better sky subtraction for extended sources
- **Priority:** Medium
- **Effort:** 2-3 weeks

### Principal Component Analysis (PCA) Sky Modeling
- **Description:** Use PCA to model sky emission line variations
- **Method:** Build PCA basis from sky regions, project and subtract
- **Benefits:** Handles variable sky conditions, OH line suppression
- **Priority:** Low
- **Effort:** 3-4 weeks

---

## Additional Wavelength Calibration Features

### Automatic Line Identification
- **Description:** Machine learning-based arc line identification
- **Method:** Train classifier on labeled arc spectra
- **Benefits:** Robust to poor initial wavelength guess
- **Priority:** Low
- **Effort:** 4-6 weeks (requires training data)

### Multi-Arc Wavelength Solution
- **Description:** Combine multiple arc exposures for improved calibration
- **Benefits:** Better line detection, improved RMS
- **Priority:** Low
- **Effort:** 1 week

---

## Quality Assessment Enhancements

### Machine Learning Quality Classifier
- **Description:** ML model to predict spectrum quality grade
- **Features:**
  - Trained on human-labeled spectra
  - Identifies specific issues (CR contamination, poor sky subtraction, etc.)
- **Priority:** Low
- **Effort:** 4-6 weeks (requires labeled training set)

### Automated Outlier Detection
- **Description:** Flag outlier spectra in batch processing
- **Method:** Statistical comparison to ensemble
- **Priority:** Low
- **Effort:** 1 week

---

## Data Management

### Database Backend
- **Description:** SQL database for organizing observations and metadata
- **Features:**
  - Query by date, target, quality grade
  - Track reduction history
  - Provenance tracking
- **Technology:** SQLite or PostgreSQL
- **Priority:** Low
- **Effort:** 3-4 weeks

### FITS Header Standardization
- **Description:** Ensure all output FITS comply with FITS standards
- **Features:**
  - Automatic WCS keywords
  - Reduction history in headers
  - Full provenance chain
- **Priority:** Medium
- **Effort:** 1-2 weeks

---

## Documentation Tutorials (Phase 9 - Deferred)

From Phase 9 tasks (T147-T150) not yet completed:

### T147: Basic Reduction Tutorial (docs/source/tutorials/)
- Step-by-step first reduction
- Interpret outputs
- Common pitfalls

### T148: Faint Galaxies Tutorial
- Optimizing for low SNR
- Trace detection tuning
- Extraction parameters

### T149: Wavelength Calibration Tutorial
- Arc lamp types
- Custom linelists
- Improving RMS
- Troubleshooting poor fits

### T150: Quality Validation Tutorial
- Interpreting quality metrics
- Diagnostic plots
- Identifying problems
- Reprocessing strategies

**Priority:** High (documentation)  
**Effort:** 1-2 weeks total

---

## Community Features

### Plugin System
- **Description:** Allow users to write custom reduction modules
- **Features:**
  - Plugin discovery and loading
  - Standard plugin API
  - Example plugins (custom extraction, sky models)
- **Priority:** Low
- **Effort:** 3-4 weeks

### Reduction Recipe Sharing
- **Description:** Community repository of reduction configurations
- **Platform:** GitHub repo with YAML recipes
- **Priority:** Low
- **Effort:** 1 week setup + ongoing maintenance

---

## Testing & CI/CD

### Continuous Integration Expansion
- **Description:** Expand CI/CD pipeline
- **Features:**
  - Test on multiple Python versions (3.10-3.12)
  - Test on multiple OS (Linux, macOS, Windows)
  - Automated performance benchmarks
  - Documentation build tests
- **Priority:** Medium
- **Effort:** 1-2 weeks

### Regression Testing Suite
- **Description:** Compare outputs against reference data
- **Method:** Run pipeline on fixed datasets, compare results
- **Priority:** Medium
- **Effort:** 2 weeks

---

## Observatory-Specific Support

### Multi-Observatory Configuration
- **Description:** Pre-configured settings for other observatories
- **Targets:**
  - Keck/LRIS
  - Gemini/GMOS
  - VLT/FORS2
  - Magellan/LDSS3
- **Priority:** Low
- **Effort:** 1-2 weeks per instrument

---

## Version Roadmap

### v0.2.0 (Q1 2025)
- Complete validation scenarios (T122-T126)
- Documentation tutorials (T147-T150)
- Performance profiling and optimization
- Bug fixes from user feedback

### v0.3.0 (Q2 2025)
- Web-based diagnostic viewer
- Multi-object spectroscopy support
- Parallel processing
- Advanced flux calibration

### v1.0.0 (Q3 2025)
- Full feature completeness
- Comprehensive documentation
- Multi-observatory support
- Production-ready for all KOSMOS observing modes

---

## Contributing

We welcome community contributions! Priority areas for contributions:

1. **Documentation:** Tutorials, examples, troubleshooting guides
2. **Testing:** Test cases, validation datasets, bug reports
3. **Features:** Implement items from this roadmap (coordinate via GitHub issues)
4. **Observatory Support:** Add support for your favorite spectrograph

See `CONTRIBUTING.md` for guidelines.

---

## References

- Phase 8 Completion Summary: `PHASE8_COMPLETION.md`
- Original Task List: `specs/001-galaxy-spec-pipeline/tasks.md`
- API Documentation: `docs/API.md`
