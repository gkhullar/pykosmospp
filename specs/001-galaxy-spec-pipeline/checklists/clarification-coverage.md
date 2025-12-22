# Clarification Coverage Report: Galaxy Spec Pipeline

**Generated**: 2025-12-22  
**Feature**: 001-galaxy-spec-pipeline  
**Session**: 2025-12-22 clarification workflow

## Overview

This report documents the clarification opportunities identified during specification review and tracks resolution status.

## Ambiguity Taxonomy Analysis

### 1. Functional Scope - RESOLVED ✅

| Topic | Ambiguity | Resolution | Status |
| ----- | --------- | ---------- | ------ |
| Trace selection | How to handle multiple detected traces (target vs. noise) | Interactive pop-up viewer for user selection | ✅ Resolved |
| Spectroscopy methodology | What workflow and steps to follow | Follow Massey (NED Level 5) and PypeIt documentation for essential vs. optional steps | ✅ Resolved |
| Essential vs. optional steps | Which reduction steps are required | Essential: bias estimation, wavelength calibration, object finding, extraction, background subtraction; Optional: flux calibration, telluric correction, AB differencing | ✅ Resolved |
| AB difference imaging | Support for nod/dither observing modes | User-specified in YAML configuration when applicable | ✅ Resolved |

### 2. Data Model - RESOLVED ✅

| Topic | Ambiguity | Resolution | Status |
| ----- | --------- | ---------- | ------ |
| Configuration format | JSON vs. YAML vs. INI | YAML with KOSMOS defaults, target-specific overrides | ✅ Resolved |
| File organization | Single-dir vs. product-type dirs | Input: split by type; Output: product-type per galaxy | ✅ Resolved |

### 3. Interaction Flows - RESOLVED ✅

| Topic | Ambiguity | Resolution | Status |
| ----- | --------- | ---------- | ------ |
| Interactive GUI scope | Pure CLI vs. GUI components | Interactive trace viewer required | ✅ Resolved |
| Sky subtraction method | pyKOSMOS vs. PypeIt approach | pyKOSMOS primary, PypeIt fallback | ✅ Resolved |

### 4. Non-Functional Requirements - RESOLVED ✅

| Topic | Ambiguity | Resolution | Status |
| ----- | --------- | ---------- | ------ |
| Error handling philosophy | Fail-fast vs. continue-on-error | Tiered: critical halt, quality issues produce flagged outputs | ✅ Resolved |

### 2. Data Model - RESOLVED ✅

| Topic | Ambiguity | Resolution | Status |
|-------|-----------|------------|--------|
| Configuration format | JSON vs. YAML vs. INI | YAML with KOSMOS defaults, target-specific overrides | ✅ Resolved |
| File organization | Single-dir vs. product-type dirs | Input: split by type (arcs/, flats/, biases/, science/); Output: product-type per galaxy (calibrations/, reduced_2d/, spectra_1d/, logs/) | ✅ Resolved |

### 3. Interaction Flows - RESOLVED ✅

| Topic | Ambiguity | Resolution | Status |
|-------|-----------|------------|--------|
| Interactive GUI scope | Pure CLI vs. GUI components | Interactive trace viewer required (matplotlib-based expected) | ✅ Resolved |
| Sky subtraction method | pyKOSMOS vs. PypeIt approach | pyKOSMOS primary, PypeIt fallback/alternative | ✅ Resolved |

### 4. Non-Functional Requirements - RESOLVED ✅

| Topic | Ambiguity | Resolution | Status |
|-------|-----------|------------|--------|
| Error handling philosophy | Fail-fast vs. continue-on-error | Tiered: critical failures halt immediately; quality issues produce outputs with quality flags and warnings | ✅ Resolved |

### 5. Integrations - NONE IDENTIFIED ✓

No integration ambiguities detected. Pipeline is standalone with standard FITS I/O.

### 6. Edge Cases - RESOLVED ✅

| Topic | Ambiguity | Resolution | Status |
| ----- | --------- | ---------- | ------ |
| AB differencing mode | Clarified observing mode specific | User specifies nod/dither pattern in YAML | ✅ Resolved |
| Flux calibration scope | Originally out of scope | Optional step within scope when standard star data provided | ✅ Resolved |
| Telluric correction scope | Originally out of scope | Optional step within scope when requested in config | ✅ Resolved |

### 7. Constraints - RESOLVED ✅

| Topic | Ambiguity | Resolution | Status |
| ----- | --------- | ---------- | ------ |
| Reduction workflow standards | Which methodology to follow | Massey (NED Level 5) and PypeIt documentation | ✅ Resolved |

### 8. Terminology - NONE IDENTIFIED ✓

Spectroscopy terminology is standard and well-defined in referenced documents (Massey, PypeIt).

## Clarification Questions Summary

**Total questions asked**: 7  
**User-answered**: 7  
**Resolved via defaults**: 0  
**Outstanding**: 0

### Q1: Trace Selection Strategy
- **Context**: FR-005, User Story 2
- **Question**: When multiple potential traces are detected (target galaxy, serendipitous sources, noise), how should the pipeline identify the primary science target?
- **Answer**: Identify multiple unique traces and prompt user with interactive pop-up viewer to select all traces that look good
- **Impact**: Added FR-005 interactive viewer requirement; moved from "Out of Scope" to required feature

### Q2: Configuration Format
- **Context**: FR-016, User Story 1
- **Question**: What configuration approach balances flexibility for different observing modes with simplicity for typical users?
- **Answer**: YAML file with sensible KOSMOS defaults, requiring only target-specific overrides (trace position hints, SNR thresholds)
- **Impact**: Updated FR-016 configuration format specification

### Q3: File Organization
- **Context**: FR-012, Data Model
- **Question**: How should the pipeline organize input and output files to support both single-target and multi-target observations?
- **Answer**: Input structure: directories split into arcs/, flats/, biases/, science/; Output structure: product-type organization (calibrations/, reduced_2d/, spectra_1d/, logs/) within each galaxy subdirectory
- **Impact**: Updated FR-012 output directory structure; added Assumptions about input organization

### Q4: Sky Subtraction Method
- **Context**: FR-006, User Story 2
- **Question**: For longslit spectroscopy of faint galaxies, how should the pipeline estimate and subtract sky background?
- **Answer**: Follow pyKOSMOS sky subtraction method as primary approach, with PypeIt framework patterns as fallback/alternative
- **Impact**: Updated FR-006 to specify pyKOSMOS primary methodology

### Q5: Error Handling Philosophy
- **Context**: FR-018, User Story 4
- **Question**: What is the pipeline's error handling philosophy for balancing robustness with productivity?
- **Answer**: Tiered approach - critical failures (corrupt FITS, wrong instrument configuration) halt immediately; quality issues (low SNR, missing arc lines) produce outputs with quality flags and warnings
- **Impact**: Updated FR-018 with detailed tiered error handling specification

### Q6: Spectroscopy Reduction Methodology
- **Context**: Overall pipeline workflow, Functional Requirements
- **Question**: What methodology should guide the spectroscopic reduction workflow?
- **Answer**: Follow standard spectroscopy reduction methodology from Massey (NED Level 5) and PypeIt documentation for essential steps: bias estimation, wavelength calibration with arcs, object finding, object extraction, background subtraction, and AB difference imaging (if applicable); flux calibration and telluric correction are optional steps
- **Impact**: Added References section; updated FR-001, FR-003, FR-005, FR-006, FR-007, FR-009; updated Assumptions and Out of Scope to clarify essential vs. optional steps

### Q7: Reference Data Resources
- **Context**: FR-007 (arc line identification), FR-009 (flux/extinction calibration)
- **Question**: What reference data should be used for wavelength calibration and extinction correction?
- **Answer**: Utilize pyKOSMOS resources directory for arc lamp linelists, atmospheric extinction curves, arc templates, and standard star spectra; downloaded locally to resources/pykosmos_reference/ for offline use
- **Impact**: Updated FR-007 to reference pyKOSMOS linelists (apohenear.dat, argon.dat, krypton.dat); updated FR-009 to reference extinction files (apoextinct.dat) and onedstds templates; added pyKOSMOS Resources to References section with subdirectory descriptions; added Assumption about local resource availability

## Specification Updates

### Sections Added
- **Clarifications** (new section after Input): Documents all Q&A from clarification session
- **References** (new section after Clarifications): Links to Massey paper, PypeIt docs, pyKOSMOS repo, pyKOSMOS resources with subdirectory breakdown

### Sections Modified
- **User Story 2**: Acceptance scenario #2 updated for interactive trace selection
- **FR-001**: Added reference to standard spectroscopy methodology (Massey, PypeIt)
- **FR-002**: Updated for input directory structure validation
- **FR-003**: Marked bias estimation as "essential step"
- **FR-005**: Added "object finding essential step" terminology and interactive viewer requirement
- **FR-006**: Added "object extraction" and "background subtraction" as essential steps; added AB differencing as optional step with YAML trigger
- **FR-007**: Marked "wavelength calibration with arcs" as essential step; added reference to pyKOSMOS linelists (apohenear.dat, argon.dat, krypton.dat)
- **FR-009**: Added optional flux calibration and telluric correction capabilities; added reference to pyKOSMOS extinction curves (apoextinct.dat) and onedstds templates
- **FR-012**: Updated output directory structure to product-type organization per galaxy
- **FR-016**: Specified YAML configuration format with KOSMOS defaults
- **FR-018**: Detailed tiered error handling strategy
- **Assumptions**: Added 3 items about input directory structure, interactive viewer capability, essential vs. optional steps, AB differencing requirements, pyKOSMOS resources availability
- **Out of Scope**: Removed flux calibration and telluric correction; added note that these are optional steps within scope

## Quality Validation Status

All requirements checklist items remain passing after clarification updates:

- ✅ No implementation details
- ✅ Focused on user value and business needs
- ✅ Written for non-technical stakeholders
- ✅ All mandatory sections completed
- ✅ No [NEEDS CLARIFICATION] markers remain
- ✅ Requirements are testable and unambiguous
- ✅ Success criteria are measurable
- ✅ Success criteria are technology-agnostic
- ✅ All acceptance scenarios defined
- ✅ Edge cases identified
- ✅ Scope clearly bounded
- ✅ Dependencies and assumptions identified
- ✅ All functional requirements have clear acceptance criteria
- ✅ User scenarios cover primary flows
- ✅ Feature meets measurable outcomes
- ✅ No implementation details leak into specification

## Readiness Assessment

**Specification Status**: ✅ READY FOR PLANNING

The specification has been thoroughly reviewed and clarified with:
- 7 critical ambiguities resolved through user consultation
- Standard spectroscopy methodology incorporated (Massey, PypeIt)
- Essential vs. optional pipeline steps clearly defined
- Interactive vs. CLI components scoped
- Error handling philosophy established
- File organization patterns defined
- pyKOSMOS reference resources identified and downloaded locally

**Next Phase**: Execute `/speckit.plan` to generate technical implementation plan.

## Notes

- User preference for interactive trace viewer overrode initial assumption of pure CLI operation
- YAML configuration format aligns with modern Python astronomy pipelines (astropy, PypeIt patterns)
- Product-type directory organization per galaxy supports multi-target batch workflows
- Tiered error handling balances scientific rigor (don't hide problems) with productivity (produce outputs when possible)
- Essential vs. optional steps distinction allows MVP focus on wavelength-calibrated 1D extraction, with flux/telluric calibration as enhancements
- pyKOSMOS reference resources (linelists: 19 files including apohenear.dat, argon.dat, krypton.dat; extinction: 4 observatories including APO; arctemplates: 23 grating configurations; onedstds: 15 standard star spectra) downloaded to resources/pykosmos_reference/ for local access during implementation
