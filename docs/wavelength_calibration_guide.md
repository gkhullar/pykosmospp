# Comprehensive Wavelength Calibration Guide

**pyKOSMOS++ v0.2.1**  
**Constitution Compliance: Principle VI (Learning Resources)**

---

## Overview

Wavelength calibration is the process of mapping pixel positions in your 2D spectrum to physical wavelengths (in Angstroms). This is essential for:

- **Spectral line identification**: Match observed features to known transitions
- **Multi-object comparison**: Combine spectra from different observations
- **Physical measurements**: Accurate redshift, line widths, equivalent widths
- **Flux calibration**: Apply wavelength-dependent sensitivity corrections

pyKOSMOS++ provides **two methods** for wavelength calibration:

1. **DTW (Dynamic Time Warping)** - RECOMMENDED ✅
2. **Traditional Line Matching** - Fallback option

---

## Method 1: DTW (Dynamic Time Warping)

### When to Use DTW

✅ **Recommended for:**
- First-time wavelength calibration
- No good dispersion guess available
- Spectra with distortion or non-linear dispersion
- Automated pipelines requiring robustness
- Unknown or poorly documented instrument setups

❌ **Not suitable for:**
- Extremely low signal-to-noise arc spectra (SNR < 3)
- Arc lamps without available templates
- Spectra with < 10 detectable lines

### How DTW Works

DTW aligns your observed arc spectrum with a reference template using dynamic time warping, which finds the optimal mapping between the two sequences even if they are stretched, compressed, or shifted. Key advantages:

- **No dispersion guess needed**: Works without knowing Å/pixel
- **Robust to distortion**: Handles non-linear effects automatically
- **Full spectrum matching**: Uses entire spectrum, not just peaks
- **Automatic range detection**: Determines wavelength range from best match

### Quick Start with DTW

```python
from pykosmospp.wavelength.dtw import identify_dtw
from pykosmospp.wavelength.match import load_arc_template, get_arc_template_name
from pykosmospp.wavelength.fit import fit_wavelength_solution
import numpy as np

# 1. Load arc frame
arc_frame = ArcFrame.from_fits("arc_001.fits")

# 2. Auto-select template based on header
lamp, grating, arm = get_arc_template_name('cuar', arc_frame.header)
template_waves, template_flux = load_arc_template(lamp, grating, arm)

# 3. Collapse 2D arc to 1D spectrum
arc_spectrum_1d = np.median(arc_frame.data, axis=1)

# 4. Perform DTW identification
pixels, wavelengths = identify_dtw(
    arc_spectrum_1d,
    template_waves,
    template_flux,
    peak_threshold=0.3,
    min_peak_separation=5
)

print(f"DTW identified {len(pixels)} arc lines")

# 5. Fit wavelength solution with provenance tracking
solution = fit_wavelength_solution(
    pixels,
    wavelengths,
    order_range=(3, 7),
    sigma_clip=3.0,
    calibration_method='dtw',
    template_used=f"{lamp}{arm}{grating}.spec",
    dtw_parameters={'peak_threshold': 0.3, 'min_peak_separation': 5}
)

print(f"RMS: {solution.rms_residual:.4f} Å")
```

### DTW Parameters

#### `peak_threshold` (default: 0.3)

Controls which peaks are detected in the **observed** spectrum after DTW alignment.

- **Lower values** (0.1-0.2): Detect more lines, including weak ones
  - Use for: High SNR arcs, dense line spectra
  - Risk: May detect noise as lines
  
- **Higher values** (0.4-0.5): Detect only strong lines
  - Use for: Low SNR arcs, sparse line spectra
  - Risk: May miss real lines, need ≥10 for fitting

**Tuning tip:** Start with default (0.3). If you get < 10 lines, lower to 0.25. If RMS > 0.2 Å, raise to 0.35-0.4.

#### `min_peak_separation` (default: 5 pixels)

Minimum distance between detected peaks to avoid double-counting blended lines.

- **Lower values** (3-4): Allow close line detection
  - Use for: High dispersion (many Å/pixel), well-separated lines
  - Risk: Blended lines counted twice
  
- **Higher values** (7-10): Enforce larger separation
  - Use for: Low dispersion (few Å/pixel), blended line spectra
  - Risk: Lose real close lines

**Tuning tip:** Set based on your dispersion. If dispersion ≈ 2 Å/pixel and lines are typically 10 Å apart, use `min_peak_separation = 10 / 2 = 5`.

#### `step_pattern` (default: 'symmetric2')

DTW alignment constraint. Advanced parameter, rarely needs adjustment.

- `'symmetric2'`: Standard symmetric pattern (default, recommended)
- `'asymmetric'`: Allow more stretching
- `'rabinerJuangStepPattern'`: Classic DTW pattern

**Tuning tip:** Leave as default unless alignment clearly fails (< 5 lines detected with good arc).

---

## Method 2: Traditional Line Matching

### When to Use Line Matching

✅ **Use when:**
- You have a good initial dispersion guess (e.g., from previous calibration)
- DTW fails (very noisy arc, unsupported lamp)
- You need to match specific lines (manual wavelength selection)
- Legacy pipeline compatibility required

❌ **Avoid when:**
- No dispersion guess available (→ use DTW instead)
- First time calibrating a new setup
- Dispersion varies significantly (use DTW for robustness)

### How Line Matching Works

1. **Detect peaks** in observed arc spectrum
2. **Match to catalog** using initial dispersion guess
3. **Refine matches** iteratively
4. **Fit polynomial** to matched pairs

Requires knowing approximate Å/pixel to search for catalog lines near each detected peak.

### Quick Start with Line Matching

```python
from pykosmospp.wavelength.match import match_lines_to_catalog, detect_arc_lines

# 1. Detect peaks in observed spectrum
peaks = detect_arc_lines(arc_frame.data, min_snr=5.0)

# 2. Match to catalog with initial guess
pixels, wavelengths, intensities = match_lines_to_catalog(
    peaks,
    lamp_type='cuar',
    initial_dispersion=2.0,  # Å/pixel (REQUIRED)
    wavelength_range=(5000, 8000),  # Expected range
    match_tolerance=5.0  # Matching window (Å)
)

print(f"Matched {len(pixels)} lines")

# 3. Fit solution
solution = fit_wavelength_solution(
    pixels,
    wavelengths,
    order_range=(3, 7),
    calibration_method='line_matching'
)
```

### Line Matching Parameters

#### `initial_dispersion` (REQUIRED)

Your best guess for Å/pixel. Critical for successful matching.

**How to estimate:**
- From previous calibration of same setup: Use that value
- From instrument specs: wavelength_range / n_pixels
- From grating equation: Use blaze angle and groove density
- First guess: Try 1.0-3.0 Å/pixel for typical longslit

**Example:** If you know 5000-8000 Å spans 2048 pixels:
```python
initial_dispersion = (8000 - 5000) / 2048  # ≈ 1.46 Å/pixel
```

#### `wavelength_range` (recommended)

Expected wavelength coverage (Å). Limits catalog search range.

- **Tighter range** (e.g., 4000-6000 Å): Faster, fewer false matches
- **Wider range** (e.g., 3000-9000 Å): More conservative, slower

**Tuning tip:** Use known instrument wavelength range. For KOSMOS Blue: 4000-5800 Å, Red: 5500-8500 Å.

#### `match_tolerance` (default: 5.0 Å)

Maximum wavelength difference for catalog line match.

- **Tighter** (2-3 Å): Reduce false matches, requires accurate dispersion guess
- **Looser** (7-10 Å): More forgiving, but more ambiguity

**Tuning tip:** Set to 2-3× your expected dispersion error. If unsure, start with 5 Å.

---

## Arc Lamp Templates

### Available Templates

pyKOSMOS++ includes **18 pre-calibrated arc templates** from the pyKOSMOS resource directory:

| Lamp | Gratings | Arms | Use Case |
|------|----------|------|----------|
| **Ar** (Argon) | 0.86-high, 1.18-ctr, 2.0-low | Blue, Red | Most common, dense line spectrum |
| **Kr** (Krypton) | 0.86-high, 1.18-ctr, 2.0-low | Blue, Red | Alternative dense spectrum |
| **Ne** (Neon) | 0.86-high, 1.18-ctr, 2.0-low | Blue, Red | Sparse, high-precision lines |

### Template Selection

#### Automatic Selection (Recommended)

```python
lamp, grating, arm = get_arc_template_name('cuar', arc_header)
# Automatically extracts grating and arm from FITS header
```

**Mapping rules:**
- `'argon'` → Ar
- `'henear'`, `'cuar'`, `'apohenear'` → Ar (default for compound lamps)
- Grating extracted from `GRISM` or `GRATING` keywords
- Arm extracted from `ARM` or `CAMID` keywords

**Defaults if header missing:**
- Grating: `'1.18-ctr'` (standard KOSMOS grating)
- Arm: `'Blue'`

#### Manual Selection

```python
# Load specific template
waves, flux = load_arc_template('Ar', '1.18-ctr', 'Blue')
```

### Template Coverage

- **Ar Blue (1.18-ctr)**: 3000-5800 Å, ~1000 lines, dense
- **Ar Red (1.18-ctr)**: 5000-9000 Å, ~1200 lines, dense
- **Ne Blue (1.18-ctr)**: 3500-5500 Å, ~200 lines, sparse but precise
- **Kr Blue (1.18-ctr)**: 3500-6000 Å, ~800 lines, alternative to Ar

**Choosing a template:**
- Match your observed lamp: If you used CuAr → use Ar template
- Match your wavelength range: Blue for < 6000 Å, Red for > 5500 Å
- Match your grating: Use template for your grating setting

---

## Troubleshooting

### Problem: "ValueError: Need at least 10 matched lines"

**Cause:** DTW or line matching found < 10 lines, insufficient for polynomial fitting.

**Solutions:**

1. **Lower peak threshold** (DTW):
   ```python
   pixels, waves = identify_dtw(..., peak_threshold=0.25)
   ```

2. **Check arc SNR**: Is spectrum noisy? Increase arc exposure time.

3. **Verify lamp type**: Using correct template? CuAr ≠ HeNeAr.

4. **Adjust wavelength range** (line matching):
   ```python
   match_lines_to_catalog(..., wavelength_range=(4000, 7000))
   ```

5. **Reduce min_peak_separation** (DTW):
   ```python
   identify_dtw(..., min_peak_separation=3)
   ```

---

### Problem: "Wavelength RMS exceeds 0.2 Å"

**Cause:** Polynomial fit has high residuals, indicates poor line matches or distortion.

**Solutions:**

1. **Increase sigma clipping** to reject outliers:
   ```python
   fit_wavelength_solution(..., sigma_clip=5.0)
   ```

2. **Adjust polynomial order range**: Try higher orders for complex distortion:
   ```python
   fit_wavelength_solution(..., order_range=(3, 9))
   ```

3. **Use DTW instead of line matching**: More robust to distortion.

4. **Check for systematic errors**:
   - Plot residuals vs pixel: Systematic pattern?
   - Plot residuals vs wavelength: Edge effects?

5. **Verify template matches your setup**:
   - Wrong grating? Wrong arm?
   - Template wavelength range covers your spectrum?

---

### Problem: "DTW identifies < 5 lines"

**Cause:** DTW alignment failed, possibly wrong template or very different dispersion.

**Solutions:**

1. **Verify template selection**:
   ```python
   print(f"Using: {lamp}{arm} {grating}")
   # Should match your actual lamp and setup
   ```

2. **Try different template**: If auto-selection wrong, load manually:
   ```python
   waves, flux = load_arc_template('Ar', '2.0-low', 'Red')
   ```

3. **Check arc spectrum quality**:
   ```python
   plt.plot(arc_spectrum_1d)
   plt.show()
   # Should show clear emission lines, not flat/noisy
   ```

4. **Increase template wavelength range**: Try adjacent arm's template.

5. **Fall back to line matching** with manual dispersion guess.

---

### Problem: "Lines detected but poor matches"

**Cause:** Detected peaks don't correspond to catalog lines.

**Solutions:**

1. **Verify lamp type**: Are you using the right catalog?
   - Check FITS header: `arc_header['LAMPID']` or `arc_header['LAMPS']`

2. **Adjust match tolerance** (line matching):
   ```python
   match_lines_to_catalog(..., match_tolerance=10.0)
   ```

3. **Improve dispersion guess** (line matching):
   - Too far off → no matches
   - Calculate from wavelength range / n_pixels

4. **Check for blended lines**: Lower min_peak_separation.

5. **Use DTW** - doesn't require individual line matches.

---

### Problem: "Wavelength solution doesn't apply correctly"

**Cause:** Solution fitted but produces wrong wavelengths when applied.

**Solutions:**

1. **Check pixel range**: Solution normalizes pixels to [-1, 1].
   ```python
   print(solution.pixel_range)  # Should match your data range
   ```

2. **Verify wavelength range**:
   ```python
   print(solution.wavelength_range)  # Should match expected coverage
   ```

3. **Test on calibration data**:
   ```python
   test_waves = solution.wavelength(pixels)
   residuals = wavelengths - test_waves
   print(f"RMS: {np.sqrt(np.mean(residuals**2)):.4f} Å")
   ```

4. **Check polynomial type**: Should be `'chebyshev'` (default).

5. **Refit with different order**: Try order=4 or 5 explicitly.

---

## Quality Assessment

### Target Metrics (Constitution Principle V)

- **RMS Residual**:
  - **< 0.1 Å**: Excellent (implementation target)
  - **< 0.2 Å**: Good (acceptance criterion)
  - **> 0.2 Å**: Poor - investigate and refit

- **Number of Lines**:
  - **≥ 10**: Minimum for polynomial fitting
  - **15-30**: Good for robust fit
  - **> 30**: Excellent, high confidence

- **Outlier Fraction**:
  - **< 20%**: Good (sigma clipping should reject outliers)
  - **> 20%**: Check line identification quality

### Diagnostic Plots

```python
import matplotlib.pyplot as plt

# 1. Wavelength solution fit
pixel_grid = np.linspace(0, 4096, 1000)
wave_grid = solution.wavelength(pixel_grid)

plt.figure(figsize=(12, 4))
plt.plot(pixels, wavelengths, 'ro', label='Matched lines')
plt.plot(pixel_grid, wave_grid, 'b-', label=f'Fit (order {solution.order})')
plt.xlabel('Pixel')
plt.ylabel('Wavelength (Å)')
plt.legend()
plt.title(f'{solution.calibration_method.upper()} - RMS {solution.rms_residual:.4f} Å')
plt.grid(True, alpha=0.3)
plt.show()

# 2. Residuals
fitted_waves = solution.wavelength(pixels)
residuals = wavelengths - fitted_waves

plt.figure(figsize=(12, 4))
plt.plot(pixels, residuals, 'ro')
plt.axhline(0, color='b', linewidth=2)
plt.axhline(solution.rms_residual, color='r', linestyle='--', label=f'RMS = {solution.rms_residual:.4f} Å')
plt.axhline(-solution.rms_residual, color='r', linestyle='--')
plt.xlabel('Pixel')
plt.ylabel('Residual (Å)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 3. Check for systematic patterns
plt.figure(figsize=(12, 4))
plt.plot(wavelengths, residuals, 'ro')
plt.axhline(0, color='b', linewidth=2)
plt.xlabel('Wavelength (Å)')
plt.ylabel('Residual (Å)')
plt.title('Residuals vs Wavelength (check for edge effects)')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Best Practices

### ✅ Do:

1. **Use DTW by default**: More robust, no dispersion guess needed
2. **Track provenance**: Pass `calibration_method`, `template_used`, `dtw_parameters`
3. **Check RMS < 0.2 Å**: Quality gate before proceeding
4. **Plot residuals**: Look for systematic patterns
5. **Use auto-template selection**: `get_arc_template_name()` extracts from header
6. **Save wavelength solutions**: Reuse for multiple science frames from same night
7. **Document your choice**: Record why you chose DTW vs line matching

### ❌ Don't:

1. **Skip quality checks**: Low RMS doesn't guarantee correct solution
2. **Ignore outliers**: > 20% rejected → check line identification
3. **Use wrong template**: Ar ≠ HeNeAr, Blue ≠ Red
4. **Fit with < 10 lines**: Underconstrained polynomial
5. **Extrapolate far beyond calibrated range**: Solution unreliable at edges
6. **Reuse old solutions for new setups**: Grating change → recalibrate
7. **Disable provenance tracking**: Always track method and parameters

---

## Example Workflows

### Workflow 1: Automated DTW (Recommended)

```python
# Complete automated wavelength calibration with DTW

from pykosmospp.wavelength.dtw import identify_dtw
from pykosmospp.wavelength.match import get_arc_template_name, load_arc_template
from pykosmospp.wavelength.fit import fit_wavelength_solution
from pykosmospp.io import ArcFrame
import numpy as np

# Load arc
arc = ArcFrame.from_fits("arc_001.fits")

# Auto-select template
lamp, grating, arm = get_arc_template_name('cuar', arc.header)
template_waves, template_flux = load_arc_template(lamp, grating, arm)

# Collapse and identify
arc_1d = np.median(arc.data, axis=1)
pixels, waves = identify_dtw(arc_1d, template_waves, template_flux)

# Fit with provenance
solution = fit_wavelength_solution(
    pixels, waves,
    calibration_method='dtw',
    template_used=f"{lamp}{arm}{grating}.spec",
    dtw_parameters={'peak_threshold': 0.3, 'min_peak_separation': 5}
)

# Validate
assert solution.rms_residual < 0.2, "RMS too high"
assert solution.n_lines_identified >= 15, "Too few lines"

print(f"✓ Wavelength calibration successful")
print(f"  Method: {solution.calibration_method}")
print(f"  RMS: {solution.rms_residual:.4f} Å")
print(f"  Lines: {solution.n_lines_identified}")
```

### Workflow 2: DTW with Manual Template

```python
# When auto-selection fails or you know the correct template

# Load specific template (Red arm, low dispersion grating)
template_waves, template_flux = load_arc_template('Ar', '2.0-low', 'Red')

# Rest is same as Workflow 1
arc_1d = np.median(arc.data, axis=1)
pixels, waves = identify_dtw(arc_1d, template_waves, template_flux, peak_threshold=0.35)

solution = fit_wavelength_solution(
    pixels, waves,
    calibration_method='dtw',
    template_used='ArRed2.0-low.spec',
    dtw_parameters={'peak_threshold': 0.35, 'min_peak_separation': 5}
)
```

### Workflow 3: Line Matching with Known Dispersion

```python
# When you have reliable dispersion from previous calibration

from pykosmospp.wavelength.match import match_lines_to_catalog, detect_arc_lines

# Detect peaks
peaks = detect_arc_lines(arc.data, min_snr=5.0)

# Match with known dispersion
pixels, waves, intensities = match_lines_to_catalog(
    peaks,
    lamp_type='cuar',
    initial_dispersion=1.95,  # From previous calibration
    wavelength_range=(5000, 8000),
    match_tolerance=4.0
)

# Fit
solution = fit_wavelength_solution(
    pixels, waves,
    calibration_method='line_matching'
)
```

---

## Advanced Topics

### Polynomial Order Selection

pyKOSMOS++ uses **Bayesian Information Criterion (BIC)** to automatically select optimal polynomial order (default range: 3-7).

**BIC balances:**
- **Fit quality** (lower residuals)
- **Model complexity** (fewer parameters preferred)

**Override automatic selection:**
```python
solution = fit_wavelength_solution(
    pixels, waves,
    order=5,  # Force order 5
    use_bic=False  # Disable BIC selection
)
```

**When to override:**
- Strong prior knowledge of needed order
- BIC selects order 3 but residuals show clear higher-order pattern
- Computational constraints (higher orders slower)

### Sigma Clipping

Iterative outlier rejection to handle mis-identified lines.

**Parameters:**
- `sigma_clip=3.0`: Rejection threshold (default)
- `max_iterations=5`: Maximum clipping cycles

**How it works:**
1. Fit polynomial to all points
2. Calculate residuals
3. Compute RMS of residuals
4. Reject points with |residual| > sigma × RMS
5. Refit without rejected points
6. Repeat until convergence or max iterations

**Tuning:**
- **Tighter** (sigma=2.0): More aggressive outlier rejection
- **Looser** (sigma=5.0): Keep more points, useful for noisy data

### Chebyshev vs Standard Polynomials

pyKOSMOS++ uses **Chebyshev polynomials** by default (more numerically stable).

**Chebyshev advantages:**
- Better conditioned for high orders
- Normalized domain [-1, 1] reduces rounding errors
- Industry standard for wavelength solutions

**Use standard polynomials:**
```python
solution = fit_wavelength_solution(
    pixels, waves,
    poly_type='polynomial'  # Standard power series
)
```

Only recommended for legacy compatibility or specific edge cases.

---

## References

**Algorithm:**
- Davenport, J. R. A. et al. (2023). *pyKOSMOS: Dynamic Time Warping for Wavelength Calibration*. [DOI:10.5281/zenodo.10152905](https://doi.org/10.5281/zenodo.10152905)
- Salvador, S. & Chan, P. (2007). *FastDTW: Toward Accurate Dynamic Time Warping in Linear Time and Space*. Intelligent Data Analysis 11(5):561-580

**Wavelength Calibration Theory:**
- Pogge, R. W. (2010). *Longslit Spectroscopy: Wavelength Calibration*. OSU Astronomy Lecture Notes
- Massey, P. et al. (1992). *Spectrophotometric Standards*. ApJ 358:344

**pyKOSMOS Resources:**
- pyKOSMOS GitHub: https://github.com/jradavenport/pykosmos
- Arc lamp linelists: `resources/pykosmos_reference/linelists/`
- Arc templates: `resources/pykosmos_reference/arctemplates/`

---

## Support

**Questions?** Open an issue: https://github.com/gkhullar/pykosmos_spec_ai/issues  
**Constitution:** See `.specify/memory/constitution.md` for development principles  
**API Reference:** See `docs/API.md` for complete function documentation
