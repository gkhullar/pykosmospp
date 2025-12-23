# pyKOSMOS++ API Reference

Complete API documentation for pyKOSMOS++ spectroscopic reduction pipeline.

**Version:** 0.1.0  
**Last Updated:** 2024

---

## Table of Contents

- [Pipeline Runner](#pipeline-runner)
- [Calibration Frames](#calibration-frames)
- [Wavelength Calibration](#wavelength-calibration)
- [Trace Detection](#trace-detection)
- [Spectral Extraction](#spectral-extraction)
- [Binning](#binning)
- [Flux Calibration](#flux-calibration)
- [Quality Assessment](#quality-assessment)
- [Data Models](#data-models)
- [Configuration](#configuration)

---

## Pipeline Runner

### `PipelineRunner`

Main entry point for automated spectroscopic reduction.

**Class:** `pykosmospp.pipeline.runner.PipelineRunner`

```python
from pykosmospp.pipeline import PipelineRunner

runner = PipelineRunner(
    input_dir="data/2024-01-15/galaxy/",
    output_dir="reduced/2024-01-15/galaxy/",
    config_path="config.yaml",
    mode="auto"
)
results = runner.run()
```

#### Parameters

- **input_dir** (*str* or *Path*): Directory containing raw FITS files organized by type (biases/, flats/, arcs/, science/)
- **output_dir** (*str* or *Path*): Output directory for reduced data
- **config_path** (*str* or *Path*, optional): Path to YAML configuration file. If None, uses default config
- **mode** (*str*, optional): Pipeline mode. Options:
  - `"auto"`: Fully automated reduction
  - `"interactive"`: Allow manual trace selection and parameter adjustment
  - `"diagnostic"`: Generate extensive diagnostic plots
  - Default: `"auto"`

#### Methods

##### `run() -> List[ReducedData]`

Execute the full reduction pipeline.

**Returns:**
- List of `ReducedData` objects, one per science frame

**Steps performed:**
1. Load and validate configuration
2. Create master bias (median combine with sigma clipping)
3. Create master flat (bias-subtracted, normalized)
4. Wavelength calibration (arc line detection and fitting)
5. Process science frames (bias/flat correction, CR rejection, trace detection, extraction)
6. Generate diagnostic plots and quality reports
7. Write output FITS files with complete reduction history

**Example:**
```python
runner = PipelineRunner("data/my_galaxy/", "reduced/my_galaxy/")
results = runner.run()

for result in results:
    print(f"Science frame: {result.filename}")
    print(f"  SNR: {result.snr:.1f}")
    print(f"  Grade: {result.grade}")
    print(f"  Output: {result.output_path}")
```

---

## Calibration Frames

### Bias Frames

#### `create_master_bias(bias_files, output_path, method='median', sigma=3.0)`

Create master bias frame from individual bias exposures.

**Module:** `pykosmospp.calibration.bias`

**Parameters:**
- **bias_files** (*list of Path*): Input bias FITS files
- **output_path** (*Path*): Output master bias FITS path
- **method** (*str*, optional): Combination method (`'median'` or `'mean'`). Default: `'median'`
- **sigma** (*float*, optional): Sigma clipping threshold. Default: 3.0

**Returns:**
- **CCDData**: Master bias frame with variance

**Example:**
```python
from pykosmospp.calibration import create_master_bias
from pathlib import Path

bias_files = list(Path("data/biases/").glob("bias*.fits"))
master_bias = create_master_bias(
    bias_files,
    output_path=Path("calibration/master_bias.fits"),
    sigma=3.0
)
print(f"Master bias: mean={master_bias.data.mean():.1f} ADU")
```

### Flat Fields

#### `create_master_flat(flat_files, master_bias, output_path, method='median', sigma=3.0)`

Create normalized master flat from individual flat exposures.

**Module:** `pykosmospp.calibration.flat`

**Parameters:**
- **flat_files** (*list of Path*): Input flat FITS files
- **master_bias** (*CCDData*): Master bias frame for subtraction
- **output_path** (*Path*): Output master flat FITS path
- **method** (*str*, optional): Combination method. Default: `'median'`
- **sigma** (*float*, optional): Sigma clipping threshold. Default: 3.0

**Returns:**
- **CCDData**: Normalized master flat (mean=1.0) with variance

**Example:**
```python
from pykosmospp.calibration import create_master_flat

flat_files = list(Path("data/flats/").glob("flat*.fits"))
master_flat = create_master_flat(
    flat_files,
    master_bias,
    output_path=Path("calibration/master_flat.fits")
)
```

---

## Wavelength Calibration

### `calibrate_wavelength(arc_frame, lamp_type='HeNeAr', fit_order=None, sigma_clip=3.0)`

Calibrate wavelength solution from arc lamp spectrum.

**Module:** `pykosmospp.wavelength.calibrate`

**Parameters:**
- **arc_frame** (*CCDData*): Processed arc lamp frame
- **lamp_type** (*str*, optional): Lamp type (`'HeNeAr'`, `'Krypton'`, `'Argon'`). Default: `'HeNeAr'`
- **fit_order** (*int*, optional): Chebyshev polynomial order. If None, use BIC to select optimal order (3-6)
- **sigma_clip** (*float*, optional): Line detection threshold. Default: 3.0

**Returns:**
- **WavelengthSolution**: Object containing:
  - `wavelengths`: 1D array of wavelengths (Å)
  - `coefficients`: Chebyshev polynomial coefficients
  - `rms_residual`: RMS fit residual (Å)
  - `n_lines`: Number of matched lines
  - `fit_order`: Polynomial order used

**Example:**
```python
from pykosmospp.wavelength import calibrate_wavelength
from pykosmospp.io import ArcFrame

arc = ArcFrame.from_fits("data/arcs/arc_001.fits")
wcs = calibrate_wavelength(arc.data, lamp_type='HeNeAr')

print(f"Wavelength solution:")
print(f"  RMS: {wcs.rms_residual:.4f} Å")
print(f"  Lines matched: {wcs.n_lines}")
print(f"  Polynomial order: {wcs.fit_order}")
print(f"  Wavelength range: {wcs.wavelengths[0]:.1f} - {wcs.wavelengths[-1]:.1f} Å")
```

### DTW Wavelength Identification (New in v0.2.1)

#### `identify_dtw(arc_spectrum, template_waves, template_flux, peak_threshold=0.3, ...)`

Identify arc line wavelengths using Dynamic Time Warping (DTW) alignment. This method is more robust than traditional line matching and **does not require an initial dispersion guess**.

**Module:** `pykosmospp.wavelength.dtw`

**Parameters:**
- **arc_spectrum** (*ndarray*): 1D observed arc spectrum (flux vs pixel)
- **template_waves** (*ndarray*): Template wavelengths (Å)
- **template_flux** (*ndarray*): Template flux values
- **peak_threshold** (*float*, optional): Peak detection threshold (relative to max flux). Default: 0.3
- **min_peak_separation** (*int*, optional): Minimum pixel separation between peaks. Default: 5
- **step_pattern** (*str*, optional): DTW step pattern. Default: 'symmetric2'

**Returns:**
- **pixel_positions** (*ndarray*): Detected peak positions in observed spectrum (sub-pixel accuracy)
- **wavelengths** (*ndarray*): Matched wavelengths from template (Å)

**Raises:**
- **ValueError**: If fewer than 10 lines detected (required for polynomial fitting)

**Example:**
```python
from pykosmospp.wavelength.dtw import identify_dtw
from pykosmospp.wavelength.match import load_arc_template, get_arc_template_name
import numpy as np

# Load arc template automatically based on header
lamp, grating, arm = get_arc_template_name('cuar', arc_frame.header)
template_waves, template_flux = load_arc_template(lamp, grating, arm)

# Collapse 2D arc to 1D spectrum
arc_spectrum_1d = np.median(arc_frame.data, axis=1)

# Perform DTW identification
pixels, wavelengths = identify_dtw(
    arc_spectrum_1d,
    template_waves,
    template_flux,
    peak_threshold=0.3,
    min_peak_separation=5
)

print(f"DTW identified {len(pixels)} arc lines")
print(f"Wavelength range: {wavelengths.min():.1f}-{wavelengths.max():.1f} Å")
```

**Advantages over traditional line matching:**
- ✅ No initial dispersion guess needed
- ✅ More robust to spectral distortion and shift
- ✅ Automatic wavelength range detection
- ✅ Works with full spectrum, not discrete lines

**Algorithm:** Based on pyKOSMOS DTW implementation (Davenport et al. 2023)

---

#### `load_arc_template(lamp, grating, arm)`

Load arc lamp template spectrum from pyKOSMOS resources.

**Module:** `pykosmospp.wavelength.match`

**Parameters:**
- **lamp** (*str*): Lamp type (`'Ar'`, `'Kr'`, `'Ne'`)
- **grating** (*str*): Grating identifier (`'0.86-high'`, `'1.18-ctr'`, `'2.0-low'`)
- **arm** (*str*): Spectrograph arm (`'Blue'`, `'Red'`)

**Returns:**
- **wavelengths** (*ndarray*): Template wavelengths (Å), sorted
- **flux** (*ndarray*): Template flux values, sorted by wavelength

**Raises:**
- **ValueError**: If invalid lamp/grating/arm combination
- **FileNotFoundError**: If template file not found

**Example:**
```python
from pykosmospp.wavelength.match import load_arc_template

# Load Argon blue template for 1.18 grating (standard setup)
waves, flux = load_arc_template('Ar', '1.18-ctr', 'Blue')

print(f"Template range: {waves[0]:.1f}-{waves[-1]:.1f} Å")
print(f"Template points: {len(waves)}")
```

**Available templates:**
- Lamps: Ar (Argon), Kr (Krypton), Ne (Neon)
- Gratings: 0.86-high, 1.18-ctr, 2.0-low
- Arms: Blue, Red
- Total: 18 templates (3 lamps × 3 gratings × 2 arms)

---

#### `get_arc_template_name(lamp_type, header)`

Auto-select appropriate arc template based on lamp type and FITS header metadata.

**Module:** `pykosmospp.wavelength.match`

**Parameters:**
- **lamp_type** (*str*): Lamp type keyword from header (e.g., `'argon'`, `'henear'`, `'cuar'`)
- **header** (*fits.Header*): FITS header containing grating/arm keywords

**Returns:**
- **lamp** (*str*): Normalized lamp name (`'Ar'`, `'Kr'`, `'Ne'`)
- **grating** (*str*): Extracted grating identifier
- **arm** (*str*): Extracted arm identifier

**Mapping rules:**
- `'argon'` → `'Ar'`
- `'henear'`, `'cuar'`, `'apohenear'` → `'Ar'` (default)
- Grating extracted from `'GRISM'` or `'GRATING'` keywords
- Arm extracted from `'ARM'` or `'CAMID'` keywords
- Defaults: `'1.18-ctr'` grating, `'Blue'` arm if not found

**Example:**
```python
from pykosmospp.wavelength.match import get_arc_template_name
from astropy.io import fits

# Read arc frame header
header = fits.getheader('arc_001.fits')

# Auto-select template
lamp, grating, arm = get_arc_template_name('cuar', header)

print(f"Selected template: {lamp}{arm} grating={grating}")
# Output: Selected template: ArBlue grating=1.18-ctr
```

---

#### `fit_wavelength_solution(pixels, wavelengths, **kwargs)`

Fit polynomial wavelength solution with robust sigma-clipping and BIC order selection.

**Module:** `pykosmospp.wavelength.fit`

**Parameters:**
- **pixels** (*ndarray*): Pixel positions of matched arc lines
- **wavelengths** (*ndarray*): Catalog wavelengths (Å)
- **arc_frame** (*ArcFrame*, optional): Source arc frame for metadata. Default: None
- **poly_type** (*str*, optional): Polynomial type. Default: `'chebyshev'`
- **order** (*int*, optional): Polynomial order. If None, uses BIC selection. Default: None
- **sigma_clip** (*float*, optional): Sigma clipping threshold. Default: 3.0
- **max_iterations** (*int*, optional): Maximum clipping iterations. Default: 5
- **min_order** (*int*, optional): Minimum order for BIC selection. Default: 3
- **max_order** (*int*, optional): Maximum order for BIC selection. Default: 7
- **use_bic** (*bool*, optional): Use BIC for order selection. Default: True
- **order_range** (*tuple*, optional): Alternative to (min_order, max_order). Default: None
- **strict_rms** (*bool*, optional): Raise error if RMS > 0.2 Å. Set False for testing. Default: True
- **calibration_method** (*str*, optional): Method used (`'dtw'` or `'line_matching'`). Default: `'line_matching'`
- **template_used** (*str*, optional): Template filename (for DTW provenance). Default: None
- **dtw_parameters** (*dict*, optional): DTW parameters (for provenance tracking). Default: None

**Returns:**
- **WavelengthSolution**: Fitted solution with provenance metadata

**Raises:**
- **ValueError**: If fewer than 10 lines or RMS exceeds 0.2 Å (when strict_rms=True)

**Provenance tracking (Constitution Principle III):**
The returned `WavelengthSolution` includes:
- `calibration_method`: Which method was used (`'dtw'` or `'line_matching'`)
- `template_used`: Arc template filename (if DTW method)
- `dtw_parameters`: DTW parameters used (if DTW method)
- `timestamp`: UTC timestamp of calibration

**Example:**
```python
from pykosmospp.wavelength.fit import fit_wavelength_solution
from pykosmospp.wavelength.dtw import identify_dtw

# After DTW identification
pixels, waves = identify_dtw(arc_spectrum, template_waves, template_flux)

# Fit with provenance tracking
solution = fit_wavelength_solution(
    pixels,
    waves,
    order_range=(3, 7),
    sigma_clip=3.0,
    calibration_method='dtw',
    template_used='ArBlue1.18-ctr.spec',
    dtw_parameters={'peak_threshold': 0.3, 'min_peak_separation': 5}
)

print(f"Method: {solution.calibration_method}")
print(f"Template: {solution.template_used}")
print(f"RMS: {solution.rms_residual:.4f} Å")
print(f"Timestamp: {solution.timestamp}")
```

---

## Trace Detection

### `detect_trace(science_frame, method='cross_correlation', smoothing=5, threshold=3.0)`

Detect spectral trace position in 2D spectrum.

**Module:** `pykosmospp.trace.detect`

**Parameters:**
- **science_frame** (*CCDData*): Processed science frame
- **method** (*str*, optional): Detection method:
  - `'cross_correlation'`: Cross-correlate with Gaussian template (default)
  - `'brightest_pixel'`: Find brightest row
  - `'median_profile'`: Use median spatial profile
- **smoothing** (*int*, optional): Spectral smoothing width (pixels). Default: 5
- **threshold** (*float*, optional): Detection threshold (σ). Default: 3.0

**Returns:**
- **Trace**: Object containing:
  - `positions`: 1D array of trace positions (pixels) vs spectral pixel
  - `profile`: Spatial profile model (Gaussian or Moffat)
  - `snr`: Trace signal-to-noise ratio
  - `confidence`: Detection confidence score

**Example:**
```python
from pykosmospp.trace import detect_trace

trace = detect_trace(science_frame.data, threshold=3.0)
print(f"Trace detected at y={trace.positions.mean():.1f} pixels (SNR={trace.snr:.1f})")
```

---

## Spectral Extraction

### `extract_spectrum(data, variance, trace, method='optimal', **kwargs)`

Extract 1D spectrum from 2D spectral image.

**Module:** `pykosmospp.extraction.extract`

**Parameters:**
- **data** (*ndarray*): 2D spectral data (spatial × spectral)
- **variance** (*ndarray*): 2D variance array
- **trace** (*Trace*): Trace object defining extraction aperture
- **method** (*str*, optional): Extraction method:
  - `'optimal'`: Variance-weighted optimal extraction (Horne 1986) - **default**
  - `'boxcar'`: Simple boxcar summation
- **kwargs**: Method-specific parameters:
  - For `'boxcar'`:
    - `aperture_width` (*int*): Aperture width in pixels. Default: 10
  - For `'optimal'`:
    - `profile_sigma` (*float*): Spatial profile width (pixels). Default: auto-detect
    - `reject_cosmics` (*bool*): Enable cosmic ray rejection. Default: True
    - `rejection_threshold` (*float*): CR rejection threshold (σ). Default: 5.0

**Returns:**
- **Spectrum1D** (specutils): 1D spectrum with:
  - `wavelength`: Wavelength array with units
  - `flux`: Extracted flux with units
  - `uncertainty`: StdDevUncertainty propagated from variance
  - `meta`: Extraction metadata (method, SNR, etc.)

**Example:**
```python
from pykosmospp.extraction import extract_spectrum

# Optimal extraction (recommended)
spectrum = extract_spectrum(
    science_frame.data,
    science_frame.variance,
    trace,
    method='optimal',
    reject_cosmics=True
)

# Boxcar extraction (faster, lower SNR)
spectrum_boxcar = extract_spectrum(
    science_frame.data,
    science_frame.variance,
    trace,
    method='boxcar',
    aperture_width=10
)

print(f"Optimal SNR: {spectrum.meta['snr']:.1f}")
print(f"Boxcar SNR: {spectrum_boxcar.meta['snr']:.1f}")
```

---

## Binning

### Spatial Binning

#### `bin_spatial(data, variance, bin_factor=2)`

Bin pixels along spatial axis to improve SNR.

**Module:** `pykosmospp.binning.spatial`

**Parameters:**
- **data** (*ndarray*): 2D spectral data (spatial × spectral)
- **variance** (*ndarray* or *None*): 2D variance array
- **bin_factor** (*int*, optional): Number of spatial pixels to combine. Default: 2

**Returns:**
- **data_binned** (*ndarray*): Binned data (spatial/bin_factor × spectral)
- **variance_binned** (*ndarray* or *None*): Propagated variance

**Notes:**
- SNR improves by ~√bin_factor
- Spatial resolution decreases by bin_factor
- Flux is conserved (summed, not averaged)

**Example:**
```python
from pykosmospp.binning import bin_spatial

# 2x spatial binning
binned_data, binned_var = bin_spatial(
    science_frame.data,
    science_frame.variance,
    bin_factor=2
)
print(f"Original shape: {science_frame.data.shape}")
print(f"Binned shape: {binned_data.shape}")
print(f"Expected SNR improvement: ~{np.sqrt(2):.2f}x")
```

### Spectral Binning

#### `bin_spectral(spectrum, bin_width_angstrom=5.0, conserve_flux=True)`

Bin spectrum to lower spectral resolution.

**Module:** `pykosmospp.binning.spectral`

**Parameters:**
- **spectrum** (*Spectrum1D*): Input 1D spectrum
- **bin_width_angstrom** (*float*, optional): Target bin width in Å. Default: 5.0
- **conserve_flux** (*bool*, optional): Conserve total flux (sum) vs preserve flux density (mean). Default: True

**Returns:**
- **Spectrum1D**: Binned spectrum with propagated uncertainties

**Example:**
```python
from pykosmospp.binning import bin_spectral

# Bin to 5Å resolution
binned_spectrum = bin_spectral(spectrum, bin_width_angstrom=5.0)
print(f"Original: {len(spectrum.wavelength)} pixels")
print(f"Binned: {len(binned_spectrum.wavelength)} pixels")
```

---

## Flux Calibration

### Extinction Correction

#### `apply_extinction_correction(spectrum, airmass, observatory='APO')`

Correct spectrum for atmospheric extinction.

**Module:** `pykosmospp.flux_calibration.extinction`

**Parameters:**
- **spectrum** (*Spectrum1D*): Input spectrum
- **airmass** (*float*): Observation airmass (sec z)
- **observatory** (*str*, optional): Observatory name for extinction curve. Options: `'APO'`, `'KPNO'`, `'CTIO'`. Default: `'APO'`

**Returns:**
- **Spectrum1D**: Extinction-corrected spectrum

**Notes:**
- Uses standard extinction curves for each observatory
- Correction factor = 10^(0.4 × extinction(λ) × airmass)
- Blue wavelengths corrected more than red

**Example:**
```python
from pykosmospp.flux_calibration import apply_extinction_correction

# Get airmass from header
airmass = science_frame.header.get('AIRMASS', 1.0)

# Apply correction
corrected_spectrum = apply_extinction_correction(
    spectrum,
    airmass=airmass,
    observatory='APO'
)
```

### Sensitivity Calibration

#### `compute_sensitivity_function(observed_spectrum, standard_spectrum, standard_name)`

Compute instrumental sensitivity function from standard star observation.

**Module:** `pykosmospp.flux_calibration.sensitivity`

**Parameters:**
- **observed_spectrum** (*Spectrum1D*): Observed standard star spectrum
- **standard_spectrum** (*Spectrum1D*): Known flux-calibrated spectrum
- **standard_name** (*str*): Standard star name (e.g., 'BD+28d4211')

**Returns:**
- **wavelengths** (*ndarray*): Wavelength array (Å)
- **sensitivity** (*ndarray*): Sensitivity function (erg/cm²/s/Å/count)

**Example:**
```python
from pykosmospp.flux_calibration import compute_sensitivity_function, apply_sensitivity_correction

# Observe a spectrophotometric standard
standard_obs = extract_spectrum(standard_frame.data, standard_frame.variance, trace)

# Get catalog spectrum
from specutils import Spectrum1D
standard_cat = Spectrum1D.read("standards/bd28d4211.fits")

# Compute sensitivity
wavelengths, sensitivity = compute_sensitivity_function(
    standard_obs,
    standard_cat,
    standard_name='BD+28d4211'
)

# Apply to science spectrum
flux_calibrated = apply_sensitivity_correction(spectrum, wavelengths, sensitivity)
```

---

## Quality Assessment

### `compute_snr(spectrum, wavelength_range=(5500, 5600))`

Compute signal-to-noise ratio in continuum region.

**Module:** `pykosmospp.quality.snr`

**Parameters:**
- **spectrum** (*Spectrum1D*): Input spectrum
- **wavelength_range** (*tuple*, optional): Wavelength range (Å) for SNR calculation. Default: (5500, 5600)

**Returns:**
- **snr** (*float*): Median SNR in specified range

**Example:**
```python
from pykosmospp.quality import compute_snr

snr = compute_snr(spectrum, wavelength_range=(5500, 5600))
print(f"Continuum SNR: {snr:.1f}")
```

### `assess_quality(spectrum, trace, wavelength_solution)`

Comprehensive quality assessment with automatic grading.

**Module:** `pykosmospp.quality.assess`

**Parameters:**
- **spectrum** (*Spectrum1D*): Extracted spectrum
- **trace** (*Trace*): Trace detection result
- **wavelength_solution** (*WavelengthSolution*): Wavelength calibration

**Returns:**
- **QualityReport**: Object containing:
  - `snr`: Signal-to-noise ratio
  - `wavelength_rms`: Wavelength fit RMS (Å)
  - `trace_confidence`: Trace detection confidence
  - `profile_consistency`: Spatial profile χ² score
  - `grade`: Overall grade (`'Excellent'`, `'Good'`, `'Fair'`, `'Poor'`)

**Grading Criteria:**
- **Excellent**: SNR >20, RMS <0.1Å, trace SNR >5σ
- **Good**: SNR >10, RMS <0.15Å, trace SNR >3σ
- **Fair**: SNR >5, RMS <0.2Å, trace SNR >2σ
- **Poor**: Below Fair thresholds

**Example:**
```python
from pykosmospp.quality import assess_quality

report = assess_quality(spectrum, trace, wcs)
print(f"Quality Grade: {report.grade}")
print(f"  SNR: {report.snr:.1f}")
print(f"  Wavelength RMS: {report.wavelength_rms:.4f} Å")
print(f"  Trace confidence: {report.trace_confidence:.2f}")
```

---

## Data Models

### `BiasFrame`

Container for bias frames.

**Attributes:**
- `data` (*ndarray*): 2D bias data
- `header` (*fits.Header*): FITS header
- `meta` (*dict*): Metadata

**Methods:**
- `from_fits(filepath)`: Load from FITS file
- `to_fits(filepath)`: Save to FITS file

### `FlatFrame`

Container for flat field frames.

### `ArcFrame`

Container for arc lamp frames.

### `ScienceFrame`

Container for science target frames.

**Additional attributes:**
- `variance` (*ndarray*): Variance array
- `trace` (*Trace*): Detected trace
- `wavelength_solution` (*WavelengthSolution*): Wavelength calibration

---

## Configuration

### `PipelineConfig`

Configuration management for pipeline parameters.

**YAML Configuration Example:**

```yaml
# config.yaml
pipeline:
  mode: auto  # auto, interactive, diagnostic
  
calibration:
  bias:
    combination_method: median
    sigma_clip: 3.0
  flat:
    combination_method: median
    sigma_clip: 3.0
    
wavelength:
  lamp_type: HeNeAr  # HeNeAr, Krypton, Argon
  fit_order: null  # null = auto-select with BIC
  sigma_clip: 3.0
  rms_threshold: 0.2  # Å
  
trace:
  method: cross_correlation
  threshold: 3.0
  smoothing: 5
  
extraction:
  method: optimal  # optimal, boxcar
  reject_cosmics: true
  rejection_threshold: 5.0
  aperture_width: 10  # For boxcar
  
quality:
  snr_wavelength_range: [5500, 5600]  # Å
  minimum_snr: 3.0
  
output:
  save_diagnostics: true
  diagnostic_format: png
  save_intermediate: false
```

**Loading configuration:**
```python
from pykosmospp.io import PipelineConfig

config = PipelineConfig.from_yaml("config.yaml")
runner = PipelineRunner(input_dir, output_dir, config=config)
```

---

## Testing Utilities

### Synthetic Data Generator

#### `generate_test_dataset(output_dir, num_bias=10, num_flat=10, num_arc=3, num_science=5)`

Generate synthetic KOSMOS FITS files for testing.

**Module:** `tests.fixtures.synthetic_data`

**Parameters:**
- **output_dir** (*Path*): Output directory
- **num_bias** (*int*): Number of bias frames. Default: 10
- **num_flat** (*int*): Number of flat frames. Default: 10
- **num_arc** (*int*): Number of arc frames. Default: 3
- **num_science** (*int*): Number of science frames. Default: 5

**Returns:**
- **dict**: Paths to generated files by type

**Notes:**
- **KOSMOS Format Matching**: The synthetic data generator produces FITS files that exactly match real KOSMOS observatory data:
  * **Shape:** (2148, 4096) pixels - spatial × spectral (FITS standard)
  * **Data type:** int32 (matching KOSMOS detector)
  * **Bias level:** ~3346 ADU (from real data mean)
  * **Read noise:** ~18.2 ADU (from real data std)
  * **Saturation:** 262143 ADU (18-bit detector)
  * **Headers:** Complete APO/KOSMOS metadata (OBSERVAT, LATITUDE, LONGITUD, TELAZ, TELALT, LST, WCS keywords)
  * **IMAGETYP values:** 'Bias', 'Comp', 'Object' (capitalized, matching real data)
- This ensures tests are reproducible without requiring real observatory data files

**Example:**
```python
from tests.fixtures.synthetic_data import generate_test_dataset
from pathlib import Path

# Generate test dataset
files = generate_test_dataset(
    Path("test_data/"),
    num_bias=10,
    num_flat=10,
    num_arc=3,
    num_science=5
)

print(f"Generated {len(files['bias'])} bias frames")
print(f"Generated {len(files['flat'])} flat frames")
print(f"Generated {len(files['arc'])} arc frames")
print(f"Generated {len(files['science'])} science frames")

# Run pipeline on synthetic data
runner = PipelineRunner("test_data/", "test_reduced/")
results = runner.run()
```

---

## Complete Example

Full reduction workflow from raw FITS to science-ready spectrum:

```python
from pathlib import Path
from pykosmospp.pipeline import PipelineRunner
from pykosmospp.io import PipelineConfig
import matplotlib.pyplot as plt

# 1. Setup paths
data_dir = Path("data/2024-01-15/my_galaxy/")
output_dir = Path("reduced/2024-01-15/my_galaxy/")
config_path = Path("config.yaml")

# 2. Run pipeline
runner = PipelineRunner(
    input_dir=data_dir,
    output_dir=output_dir,
    config_path=config_path,
    mode="auto"
)
results = runner.run()

# 3. Examine results
for result in results:
    print(f"\nScience frame: {result.filename}")
    print(f"  Grade: {result.grade}")
    print(f"  SNR: {result.snr:.1f}")
    print(f"  Wavelength RMS: {result.wavelength_rms:.4f} Å")
    
    # Plot spectrum
    spectrum = result.spectrum
    plt.figure(figsize=(12, 6))
    plt.plot(spectrum.wavelength, spectrum.flux)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.title(f'{result.filename} (Grade: {result.grade}, SNR: {result.snr:.1f})')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f"{result.filename}_spectrum.png")
    plt.close()

print(f"\n✅ Reduction complete! Results saved to {output_dir}")
```

---

## Version History

**v0.1.0** (Current)
- Initial API release
- Core reduction pipeline
- Optimal extraction
- Wavelength calibration with BIC model selection
- Quality assessment and grading
- Batch processing
- Advanced features: binning, flux calibration, multiple extraction methods

---

## Support

**Documentation:** [ReadTheDocs](https://pykosmos-spec-ai.readthedocs.io/)  
**Repository:** [GitHub](https://github.com/gkhullar/pykosmos_spec_ai)  
**Issues:** [GitHub Issues](https://github.com/gkhullar/pykosmos_spec_ai/issues)

**Citation:**
```bibtex
@software{pykosmospp2024,
  author = {Khullar, Gourav},
  title = {pyKOSMOS++: AI-Assisted Spectroscopic Reduction Pipeline},
  year = {2024},
  url = {https://github.com/gkhullar/pykosmos_spec_ai}
}
```
