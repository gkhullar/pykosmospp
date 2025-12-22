# Learning Resources Catalog

**Purpose**: Per constitution.md §VI, this catalog documents available learning resources in parent directory (`../`) that must be consulted before implementing spectroscopic reduction features.

**Constitution Requirement**: Developers MUST consult relevant notebooks and resources before implementing wavelength calibration, flux calibration, trace extraction, and validation algorithms.

---

## Jupyter Notebooks

### Location: `../../notebooks/`

#### apo05workflow.ipynb (38 MB)
- **Size**: 38 MB (large notebook with extensive outputs)
- **Relevant For**: 
  - Complete APO-KOSMOS reduction workflow
  - Real-world data processing examples
  - Algorithm parameter tuning demonstrations
  - Output validation patterns
- **Use Cases**:
  - **BEFORE implementing PipelineRunner.run()**: Review workflow sequence and stage dependencies
  - **BEFORE implementing wavelength calibration (T032-T035)**: Check arc line detection and fitting examples
  - **BEFORE implementing trace extraction (T039-T043)**: Review spatial profile fitting and sky subtraction
  - **DURING testing (T114-T126)**: Compare pipeline outputs against notebook results
- **Key Sections to Review**:
  - Data loading and FITS header inspection
  - Calibration frame creation (bias, flat)
  - Wavelength solution fitting with arc lamps
  - Trace detection and optimal extraction
  - Quality assessment and diagnostic plots

---

## Reference Materials

### Location: `../../resources/`

#### massey_astrospectra.pdf (1.6 MB)
- **Title**: User's Guide to CCD Reductions with IRAF (Massey et al.)
- **Relevant For**:
  - Standard spectroscopic CCD reduction methodology
  - Bias estimation procedures
  - Flat field normalization
  - Wavelength calibration theory
  - Error propagation through reduction stages
- **Use Cases**:
  - **BEFORE implementing calibration module (T023-T027)**: Review bias and flat field best practices
  - **BEFORE implementing wavelength fitting (T034)**: Understand polynomial order selection and residual analysis
  - **DURING validation design (T079-T082)**: Reference physics-based validation criteria
  - **WHEN writing documentation (T119-T126)**: Cite standard methodology for scientific justification
- **Key Sections**:
  - §2: CCD Characteristics (readnoise, gain, saturation)
  - §3: Bias and Dark Frames
  - §4: Flat Fielding
  - §5: Wavelength Calibration
  - §6: Spectral Extraction
  - §7: Error Analysis

---

## Project-Local Resources

### Location: `./resources/pykosmos_reference/`

**Note**: These are downloaded from pyKOSMOS repository per spec.md references section.

Expected Contents (per T005):
- `linelists/`: Arc lamp emission line catalogs (He-Ne-Ar, Ar, Kr, Th-Ar, Cu-Ar)
- `extinction/`: Observatory-specific atmospheric extinction curves (apoextinct.dat for APO)
- `arctemplates/`: Pre-extracted arc lamp spectra for KOSMOS grating configurations
- `onedstds/`: Spectrophotometric standard star templates for flux calibration

**Status**: Verify these exist before implementing:
- T033: Line matching (requires linelists/)
- T107: Extinction correction (requires extinction/apoextinct.dat)
- T108: Sensitivity function (requires onedstds/)

---

## Usage Guidelines (Per Constitution §VI)

### Required Consultation Timeline

**Phase 3 (User Story 1 - MVP)**:
1. **T024-T027 (Calibration)**: Review massey_astrospectra.pdf §3-4 before implementing combine/bias/flat
2. **T032-T035 (Wavelength)**: Review apo05workflow.ipynb wavelength section AND massey_astrospectra.pdf §5 before implementing arc line detection and fitting
3. **T039-T043 (Extraction)**: Review apo05workflow.ipynb trace extraction section before implementing spatial profiles and optimal extraction
4. **T046-T048 (Quality)**: Review apo05workflow.ipynb diagnostic plots section before implementing validation plots

**Phase 8 (Polish)**:
1. **T107-T108 (Flux Calibration)**: Review apo05workflow.ipynb flux calibration section (if present) before implementing sensitivity functions
2. **T113 (Synthetic Data)**: Review apo05workflow.ipynb to understand realistic FITS structure and header keywords

### How to Use Notebooks

1. **Before Implementation**:
   - Execute relevant notebook cells to see algorithm in action
   - Note parameter values used (sigma clipping thresholds, polynomial orders, aperture widths)
   - Identify edge cases handled (saturated lines, faint traces, cosmic rays)

2. **During Implementation**:
   - Cross-reference function signatures and return types
   - Validate intermediate outputs match notebook results
   - Use notebook as integration test baseline

3. **During Testing**:
   - Generate comparison outputs using same input data
   - Verify RMS residuals, SNR estimates, quality grades match
   - Document any intentional deviations with justification

### How to Use PDFs

1. **Algorithm Selection**: Consult massey_astrospectra.pdf for standard methodology consensus
2. **Parameter Justification**: Cite massey_astrospectra.pdf when setting default thresholds (e.g., sigma=3.0 for clipping)
3. **Error Propagation**: Follow massey_astrospectra.pdf §7 formulas for variance arrays
4. **Documentation**: Reference massey_astrospectra.pdf sections in docstrings and plan.md

---

## Compliance Checklist

Before implementing each module, verify:

- [ ] T000: This catalog created and reviewed ✅ (YOU ARE HERE)
- [ ] Relevant notebook sections identified for module
- [ ] Notebook executed to observe algorithm behavior
- [ ] Key parameters noted from notebook examples
- [ ] Massey PDF sections read for theoretical foundation
- [ ] Edge cases from notebook documented in tasks.md or code comments
- [ ] Implementation plan includes notebook comparison tests

**Constitution §VI Compliance**: Developers who skip notebook consultation without documented justification violate constitution principle and risk producing incorrect scientific results.

---

## Updates Log

- **2025-12-22**: Initial catalog created (T000)
  - Documented apo05workflow.ipynb (38 MB, complete APO workflow)
  - Documented massey_astrospectra.pdf (1.6 MB, standard methodology)
  - Verified resources/pykosmos_reference/ exists locally
  - Linked resources to specific task IDs (T032-T035, T039-T043, T107-T108)
  
**Next Review**: After Phase 3 (US1) completion, catalog additional notebooks if discovered during implementation.
