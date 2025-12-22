<!--
SYNC IMPACT REPORT
==================
Version Change: 1.0.0 → 1.1.0
Last Amendment: 2025-12-22

Changes in v1.1.0 (MINOR):
- Added VI. Learning Resources & Reference Materials section
- Mandates consultation of notebooks/ and resources/ in parent directory
- Specifies when and how to use reference materials during development
- Reinforces physics-first approach with worked examples

Original Principles (unchanged):
- I. Physics-Validated Packages: Mandatory use of established astronomy/physics Python packages
- II. Spectroscopy Standards Alignment: Compliance with pyKOSMOS/PypeIt conventions
- III. Data Provenance & Reproducibility: Full pipeline traceability requirements
- IV. Modular Pipeline Architecture: Independent, testable reduction stages
- V. Scientific Validation: Unit tests must include physics validation cases

Templates Requiring Updates:
✅ plan-template.md - Constitution Check section references all principles
✅ spec-template.md - Functional requirements align with validation needs
✅ tasks-template.md - Task phases respect modular architecture
⚠️  research.md guidance - Should reference learning resources for algorithm selection

Follow-up TODOs:
- Developers should catalog available notebooks and create index if not present
- Consider adding constitution gate: "Consulted relevant notebooks/resources before implementation"

Commit Message:
docs: add Learning Resources section to constitution v1.1.0
-->

# pyKOSMOS Spectroscopic Reduction Pipeline Constitution

## Core Principles

### I. Physics-Validated Packages

All spectroscopic data processing MUST use established, peer-reviewed Python packages that implement validated physics and astronomy algorithms. This ensures scientific correctness and community alignment.

**Mandatory packages:**

- **astropy**: Fundamental astronomy data structures (FITS handling, coordinates, units, time)
- **specutils**: Spectroscopic data manipulation (Spectrum1D objects, line fitting, continuum normalization)
- **scipy**: Signal processing and optimization (filtering, interpolation, curve fitting)
- **numpy**: Numerical array operations (the foundation for all computation)
- **matplotlib**: Visualization following astronomical plotting conventions

**Rationale:** Spectroscopic reduction involves complex physics (wavelength calibration, flux conservation, error propagation). Using battle-tested packages prevents reinvention of validated algorithms and ensures results align with community standards established by pyKOSMOS, PypeIt, and similar pipelines.

**Forbidden:** Custom implementations of wavelength calibration, flux calibration, or cosmic ray rejection without explicit justification and validation against established methods.

### II. Spectroscopy Standards Alignment

Code patterns, terminology, and algorithms MUST align with established spectroscopic reduction pipelines, particularly pyKOSMOS and PypeIt.

**Requirements:**

- Use standard terminology (arc lamps, flat fields, bias frames, trace extraction, wavelength solution)
- Follow FITS header conventions from pyKOSMOS for metadata propagation
- Adopt reduction workflow patterns: calibration → extraction → wavelength → flux → combination
- Document deviations from reference pipelines with scientific justification

**Rationale:** Consistency with established pipelines enables cross-validation of results, simplifies code review by domain experts, and facilitates community adoption. Spectroscopy has well-established best practices that should be preserved.

### III. Data Provenance & Reproducibility

Every processed data product MUST carry complete provenance: input files, processing steps, software versions, and parameters. Any scientist must be able to reproduce results from raw data.

**Requirements:**

- Log all reduction parameters (extraction aperture, polynomial orders, sigma clipping thresholds)
- Record software versions (package versions via pip freeze or equivalent)
- Preserve input file paths and timestamps in output FITS headers
- Generate processing logs with timestamps for each reduction stage
- Store parameter files alongside outputs (JSON or YAML format)

**Rationale:** Scientific reproducibility is non-negotiable. Spectroscopic reductions involve subjective choices (polynomial orders, masking regions) that must be documented. Astronomy operates on decade timescales—data reduced today must be reproducible in 2035.

### IV. Modular Pipeline Architecture

The reduction pipeline MUST be decomposed into independent, single-responsibility modules that can be tested, debugged, and executed separately.

**Structure:**

```text
src/
├── calibration/        # Bias, dark, flat field processing
├── extraction/         # Trace identification, optimal extraction
├── wavelength/         # Arc line identification, wavelength solution
├── flux_calibration/   # Standard star flux calibration
├── combination/        # Multi-exposure combining, cosmic ray rejection
└── io/                # FITS I/O, header management
```

**Requirements:**

- Each module operates on standard data structures (astropy.nddata.CCDData, specutils.Spectrum1D)
- Modules accept explicit inputs, return outputs (no hidden state)
- Each module has unit tests with sample FITS files
- CLI entry points for each major module

**Rationale:** Spectroscopic reduction is complex (10+ stages). Monolithic scripts are undebuggable. Modular architecture enables incremental development, isolated testing, and allows users to run partial reductions (e.g., reprocess only wavelength calibration).

### V. Scientific Validation

Unit tests MUST include physics validation cases, not just code coverage. Tests verify scientific correctness, not just execution.

**Validation requirements:**

- Wavelength calibration: Synthetic arc lines → RMS residual < 0.1 Å
- Flux calibration: Known standard star → photometric accuracy within 5%
- Cosmic ray rejection: Injected cosmic rays → 95% recovery rate
- Error propagation: Poisson noise → variance matches expectation
- Continuum normalization: Synthetic spectrum → recover true continuum

**Test data:**

- Include minimal real FITS files from APO/KOSMOS (with permission)
- Generate synthetic spectra with known properties
- Test edge cases (faint sources, saturated lines, bad columns)

**Rationale:** Spectroscopy is physics, not just software. Tests must validate that wavelength solutions are accurate, flux is conserved, and errors are properly propagated. Code that executes without error but produces wrong wavelengths is worse than code that crashes.

## Learning Resources & Reference Materials

Developers implementing spectroscopic reduction features MUST consult the reference materials in `/Users/gkhullar/Desktop/projects/UWashington/apo/reductions/` (parent directory) for physics background, algorithm details, and worked examples.

**Required reference materials:**

**Jupyter Notebooks:**

- Check `notebooks/` directory for interactive tutorials demonstrating:
  - Wavelength calibration workflows with arc lamp data
  - Flux calibration procedures using standard stars
  - Cosmic ray rejection algorithms with real examples
  - Error propagation through reduction pipeline stages
  - Comparison with pyKOSMOS and PypeIt reference outputs

**Resources Directory:**

- `resources/` contains essential reference materials:
  - Arc line atlases and wavelength calibration references
  - Standard star catalogs and flux calibration data
  - FITS header specifications for APO/KOSMOS
  - Algorithm documentation from published papers
  - Validation datasets with known ground truth

**Usage requirements:**

- BEFORE implementing wavelength calibration: Review notebook examples for algorithm selection and parameter tuning
- BEFORE implementing flux calibration: Consult standard star resources for photometric system details
- DURING testing: Compare outputs against validation datasets in resources
- WHEN debugging: Reference worked examples in notebooks to identify discrepancies
- FOR documentation: Cite specific notebooks/resources that guided implementation decisions

**Rationale:** Spectroscopic reduction is a mature field with established best practices. The notebooks and resources directory contains domain expertise accumulated from prior reductions, published algorithms, and community standards. Reinventing algorithms without consulting these materials risks producing incorrect results. The notebooks serve as living documentation showing how physics translates to code.

## Technology Stack Constraints

**Language:** Python 3.10+ (required for modern astropy/specutils)

**Core dependencies (MUST use these):**

- astropy >= 5.3
- specutils >= 1.10
- scipy >= 1.10
- numpy >= 1.23
- matplotlib >= 3.6

**Allowed for specific needs:**

- photutils (for aperture photometry, trace finding)
- ccdproc (for CCD image calibration workflows)
- pandas (for tabular data like line lists)
- pyyaml (for configuration files)

**Testing:** pytest with astropy-pytest-plugins

**Forbidden without justification:**

- Web frameworks (Flask, Django) — this is a reduction pipeline, not a web service
- Heavy ML frameworks (TensorFlow, PyTorch) — unless for specific feature extraction justified in research.md
- Custom FITS readers — use astropy.io.fits exclusively

## Development Workflow

### Constitution Gates

**Pre-implementation (during /speckit.plan):**

- [ ] All dependencies listed in plan.md are from approved package list
- [ ] No custom wavelength/flux algorithms without explicit justification
- [ ] Modular architecture matches principle IV structure

**Pre-commit (during /speckit.implement):**

- [ ] Each module has physics validation tests (principle V)
- [ ] FITS headers include provenance fields (principle III)
- [ ] Code follows terminology from pyKOSMOS/PypeIt (principle II)

**Violation handling:**

- Minor violations (non-standard terminology): Document in code comments, fix in next refactor
- Major violations (custom wavelength calibration): BLOCK until justified in research.md with validation plan
- Critical violations (no provenance): BLOCK until corrected

### Simplicity Principle

Start with the simplest scientifically valid approach:

- Use specutils.fitting for line fitting (don't write custom Gaussian fitters)
- Use scipy.ndimage for image operations (don't implement custom convolutions)
- Use ccdproc for standard CCD reduction (don't reimplement bias subtraction)

Only add complexity when scientific requirements demand it, with explicit justification in plan.md Complexity Tracking section.

## Governance

This constitution supersedes all other development practices for the pyKOSMOS spectroscopic reduction pipeline.

**Amendment Process:**

1. Proposed changes documented with scientific rationale
2. Constitution version incremented per semantic versioning:
   - MAJOR: Remove/redefine core principles
   - MINOR: Add new principle or expand existing
   - PATCH: Clarify wording, fix ambiguities
3. All dependent templates (plan, spec, tasks) updated for consistency
4. Sync Impact Report generated documenting cascade effects

**Compliance Reviews:**

- `/speckit.analyze` verifies spec.md, plan.md, tasks.md against constitution
- Constitution violations flagged as CRITICAL require explicit justification or remediation
- Principle reinterpretation not permitted—amendments must be formal

**Runtime Guidance:**

- See `.github/copilot-instructions.md` for AI assistant guidance aligned with these principles
- Constitution principles take precedence over convenience or development speed
- When in doubt, reference pyKOSMOS or PypeIt as canonical examples

**Version**: 1.1.0 | **Ratified**: 2025-12-22 | **Last Amended**: 2025-12-22
