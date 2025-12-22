# pyKOSMOS Reference Resources

**Source**: <https://github.com/jradavenport/pykosmos/tree/main/pykosmos/resources>  
**Downloaded**: 2025-12-22  
**Purpose**: Reference data for KOSMOS spectroscopy reduction pipeline

This directory contains essential reference data from the pyKOSMOS pipeline for wavelength calibration, atmospheric extinction correction, and flux calibration.

## Directory Structure

### `linelists/`

Arc lamp emission line catalogs for wavelength calibration. Line positions in Angstroms.

**Key files**:

- `apohenear.dat` - He-Ne-Ar arc lamp lines (recommended for APO-KOSMOS)
- `argon.dat` - Argon arc lamp lines
- `krypton.dat` - Krypton arc lamp lines
- `henear.dat` - He-Ne-Ar composite linelist
- `thar.dat` - Thorium-Argon arc lamp lines
- `cuar.dat` - Copper-Argon arc lamp lines

**Format**: Two-column ASCII (wavelength in Angstroms, relative intensity)

### `extinction/`

Observatory-specific atmospheric extinction curves for airmass correction.

**Files**:

- `apoextinct.dat` - Apache Point Observatory (APO) extinction curve
- `ctioextinct.dat` - Cerro Tololo Inter-American Observatory
- `kpnoextinct.dat` - Kitt Peak National Observatory
- `ormextinct.dat` - Observatorio del Roque de los Muchachos

**Format**: Two-column ASCII (wavelength in Angstroms, magnitude per airmass)

### `arctemplates/`

Pre-extracted arc lamp template spectra for different KOSMOS grating configurations.

**Naming convention**: `{lamp}{grating}-{resolution}.spec`

- Lamps: Ar (Argon), Kr (Krypton), Ne (Neon)
- Gratings: Blue, Red
- Resolutions: 0.86-high, 1.18-ctr, 2.0-low

**Examples**:

- `ArBlue0.86-high.spec` - Argon arc with Blue grating at 0.86"/pix (high resolution)
- `KrRed2.0-low.spec` - Krypton arc with Red grating at 2.0"/pix (low resolution)
- `NeBlue1.18-ctr.spec` - Neon arc with Blue grating at 1.18"/pix (central)

### `onedstds/`

Spectrophotometric standard star templates for flux calibration.

Standard stars with known absolute flux distributions used to convert instrumental counts to physical flux units.

## Usage in Pipeline

### Wavelength Calibration (Essential)

1. Load appropriate linelist for arc lamp used (e.g., `apohenear.dat` for He-Ne-Ar)
2. Identify arc emission lines in observed spectrum using peak-finding
3. Match detected peaks to reference wavelengths from linelist
4. Fit polynomial wavelength solution with sigma-clipped robust fitting
5. Apply solution to science spectra

**Relevant FR**: FR-007, FR-008, FR-009

### Atmospheric Extinction Correction (Optional)

1. Load observatory extinction curve (e.g., `apoextinct.dat` for APO)
2. Calculate airmass for observation from FITS header (RA, Dec, time, observatory location)
3. Interpolate extinction curve to science spectrum wavelength grid
4. Apply correction: `flux_corrected = flux_observed * 10^(0.4 * extinction * airmass)`

**Relevant FR**: FR-009 (optional step)

### Flux Calibration (Optional)

1. Observe spectrophotometric standard star from `onedstds/` catalog
2. Reduce standard star spectrum through pipeline (wavelength calibration, extraction)
3. Compare extracted standard spectrum to reference template
4. Derive sensitivity function (instrumental response)
5. Apply sensitivity function to science spectra to obtain absolute flux calibration

**Relevant FR**: FR-009 (optional step)

## pyKOSMOS Integration

The reference resources follow pyKOSMOS conventions:

- Linelists are compatible with `pykosmos.identify.loadlinelist()` function
- Extinction files follow IRAF 2-column format used by `pykosmos.fluxcal.obs_extinction()`
- Arc templates can be used for cross-correlation wavelength initialization
- Standard star catalogs follow standard spectrophotometry conventions

## References

- **pyKOSMOS Repository**: <https://github.com/jradavenport/pykosmos>
- **IRAF Linelists**: <https://github.com/joequant/iraf/tree/master/noao/lib/linelists> (original source)
- **Massey et al. CCD Reductions Guide**: `../massey_ccd_reductions.pdf` (downloaded from NED Level 5)
- **Specification**: `specs/001-galaxy-spec-pipeline/spec.md` (FR-007, FR-009)

## Notes

- Arc templates (`arctemplates/`) are grating-specific - ensure correct template for observing configuration
- APO extinction curve (`apoextinct.dat`) should be used for all APO-KOSMOS observations
- Linelist selection depends on arc lamp configuration (typically He-Ne-Ar for KOSMOS)
- Standard star observations are optional but recommended for absolute flux calibration
