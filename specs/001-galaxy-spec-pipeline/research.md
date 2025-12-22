# Phase 0: Research & Algorithm Selection

**Feature**: 001-galaxy-spec-pipeline  
**Date**: 2025-12-22  
**Purpose**: Resolve all technical unknowns before Phase 1 design

## Overview

This document resolves algorithm selection, library choices, and implementation patterns for the KOSMOS galaxy spectroscopy pipeline. All "NEEDS CLARIFICATION" items from Technical Context are researched here.

## Research Areas

### 1. Faint Trace Detection Algorithms

**Challenge**: Detect low surface brightness galaxy traces (SNR ~ 3-5) in presence of noise, cosmic rays, and serendipitous sources.

**Options Evaluated**:

1. **Cross-Correlation with Template Profile**
   - **Method**: Convolve 2D spectrum with expected spatial profile (Gaussian)
   - **Pros**: Robust to noise; standard in pyKOSMOS
   - **Cons**: Assumes profile shape; may miss irregular galaxies
   - **Reference**: pyKOSMOS `findtrace()` function

2. **Matched Filter Detection**
   - **Method**: Optimal linear filter = signal/noise^2 in Fourier domain
   - **Pros**: Maximizes SNR for known profile shape
   - **Cons**: Computationally expensive; requires noise model
   - **Reference**: PypeIt `detect_lines` using matched filtering

3. **Adaptive Thresholding with Collapse**
   - **Method**: Collapse 2D spectrum spatially (median/mean); find peaks above threshold
   - **Pros**: Simple; no assumptions about profile shape
   - **Cons**: Sensitive to noise; may miss faint traces
   - **Reference**: Massey Ch. 7 on aperture selection

**Decision**: **Cross-correlation with Gaussian template** (Option 1)

**Rationale**:

- Aligns with pyKOSMOS methodology (Constitution Principle II)
- scipy.ndimage.correlate provides efficient implementation
- Can augment with multiple template widths for different seeing conditions
- User will interactively confirm detected traces, so false positives acceptable

**Implementation Notes**:

- Use scipy.ndimage.correlate1d for spatial collapse with Gaussian kernel
- Width parameter σ = FWHM/2.355 from config (default 3-5 pixels for KOSMOS)
- Peak-finding with scipy.signal.find_peaks (prominence threshold from config)
- **Exclude emission lines**: Use continuum regions only for profile estimation
  - Mask regions with strong gradients (|dF/dλ| > threshold)
  - Fit smooth function (spline/polynomial) to continuum, use for template

---

### 2. AB Difference Imaging (Iterative Subtraction)

**Challenge**: Subtract nod-dither pairs (A-B) to remove sky/background when telescope offset between exposures. Must handle spatial misalignment and flux variations.

**Options Evaluated**:

1. **Direct Pixel Subtraction**
   - **Method**: A - B after spatial alignment
   - **Pros**: Simple; standard for well-aligned data
   - **Cons**: Fails with misalignment; residuals from flexure
   - **Reference**: Basic spectroscopy textbooks

2. **Iterative Sky Modeling**
   - **Method**: Model sky in each frame independently, then subtract models
   - **Pros**: Handles misalignment; robust to flexure
   - **Cons**: May over-subtract if object flux contaminated sky regions
   - **Reference**: PypeIt `skysub.local_skysub_extract`

3. **Cross-Correlation Alignment + Subtraction**
   - **Method**: Cross-correlate A and B to find offset; shift and subtract
   - **Pros**: Corrects for spatial/spectral shifts
   - **Cons**: Requires overlap region; computationally expensive
   - **Reference**: astropy.nddata.CCDData with wcs alignment

**Decision**: **Iterative Sky Modeling with Outlier Rejection** (Option 2)

**Rationale**:

- Most robust approach for faint galaxies where object flux may be present in both A and B positions
- Follows pyKOSMOS sky subtraction philosophy (user clarification Q4)
- Allows sigma-clipping to reject object flux from sky regions iteratively
- Handles flexure and small misalignments without explicit registration

**Implementation Algorithm**:

```python
# Iterative AB subtraction (pseudo-code)
for iteration in range(max_iterations):
    # 1. Estimate sky in A frame
    sky_A = median(spatial_regions_away_from_trace, axis=spatial)
    sky_A_2d = broadcast_to_2d(sky_A)
    
    # 2. Subtract sky from A
    A_minus_sky = A - sky_A_2d
    
    # 3. Estimate sky in B frame (excluding object trace from A)
    trace_mask_B = shift_trace_mask(A_trace_position, A_B_offset)
    sky_B = sigma_clipped_median(B, mask=trace_mask_B, sigma=3.0)
    sky_B_2d = broadcast_to_2d(sky_B)
    
    # 4. Subtract sky from B
    B_minus_sky = B - sky_B_2d
    
    # 5. Check convergence (RMS of sky residuals)
    if sky_residual_rms < threshold:
        break
    
    # 6. Update trace mask based on residuals
    update_trace_mask_from_residuals(A_minus_sky, B_minus_sky)

# Final AB difference
AB_diff = A_minus_sky - shift(B_minus_sky, offset)
```

**Key Implementation Details**:

- Use astropy.stats.sigma_clipped_stats for robust sky estimation
- Spatial regions: 20-50 pixels away from trace center (config parameter)
- Convergence criterion: RMS change < 1% or max 5 iterations
- **Visualize each iteration**: Plot sky-subtracted frames to assess quality

---

### 3. Interactive Trace Viewer (Matplotlib Patterns)

**Challenge**: Display 2D spectrum with detected traces; allow user to select/deselect traces interactively.

**Reference Implementation**: [pyKOSMOS interactive_trace_example.ipynb](https://github.com/jradavenport/pykosmos-notebooks/blob/main/interactive_trace_example.ipynb)

**Key Patterns from pyKOSMOS Notebook**:

1. **Display Strategy**:
   - Show 2D spectrum with log-scale colormap for dynamic range
   - Overlay detected traces as horizontal lines (different colors)
   - Add text labels for trace IDs

2. **Interaction Model**:
   - matplotlib Button widgets for "Accept"/"Reject" per trace
   - CheckButtons for multi-select
   - matplotlib.widgets.RectangleSelector for manual region selection (advanced)

3. **Layout**:
   - Main axis: 2D spectrum image
   - Right sidebar: Control buttons/checkboxes
   - Bottom: Spatial profile plot (collapsed spectrum)

**Decision**: **Matplotlib Interactive Widget-Based Viewer**

**Implementation Pattern**:

```python
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import numpy as np

class InteractiveTraceSelector:
    def __init__(self, spectrum_2d, detected_traces):
        self.fig, (self.ax_spec, self.ax_profile) = plt.subplots(
            2, 1, figsize=(12, 8), height_ratios=[3, 1]
        )
        
        # Display 2D spectrum with LaTeX styling
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        
        self.im = self.ax_spec.imshow(
            spectrum_2d, aspect='auto', cmap='viridis',
            norm=LogNorm(vmin=np.percentile(spectrum_2d, 1),
                         vmax=np.percentile(spectrum_2d, 99))
        )
        self.ax_spec.set_xlabel(r'Spectral Pixel')
        self.ax_spec.set_ylabel(r'Spatial Pixel')
        self.ax_spec.set_title(r'\\textbf{2D Spectrum with Detected Traces}')
        
        # Overlay traces
        self.trace_lines = []
        for i, trace in enumerate(detected_traces):
            line, = self.ax_spec.plot(
                trace.spectral_pixels, trace.spatial_positions,
                label=f'Trace {i+1}', linewidth=2, alpha=0.7
            )
            self.trace_lines.append(line)
        
        # Spatial profile (collapsed spectrum)
        profile = np.median(spectrum_2d, axis=1)
        self.ax_profile.plot(profile)
        self.ax_profile.set_xlabel(r'Spatial Pixel')
        self.ax_profile.set_ylabel(r'Flux (counts)')
        self.ax_profile.axvline for trace in detected_traces)
        
        # Interactive checkbox widgets
        trace_labels = [f'Trace {i+1}' for i in range(len(detected_traces))]
        self.check = CheckButtons(
            plt.axes([0.02, 0.4, 0.15, 0.3]),
            trace_labels,
            [True] * len(detected_traces)
        )
        
        self.check.on_clicked(self.on_trace_toggle)
        
        # Accept button
        self.btn_accept = Button(plt.axes([0.02, 0.2, 0.15, 0.075]), 'Accept Selection')
        self.btn_accept.on_clicked(self.on_accept)
        
        self.selected_traces = list(range(len(detected_traces)))
        
    def on_trace_toggle(self, label):
        trace_idx = int(label.split()[1]) - 1
        if trace_idx in self.selected_traces:
            self.selected_traces.remove(trace_idx)
            self.trace_lines[trace_idx].set_alpha(0.2)
        else:
            self.selected_traces.append(trace_idx)
            self.trace_lines[trace_idx].set_alpha(0.7)
        self.fig.canvas.draw()
    
    def on_accept(self, event):
        plt.close(self.fig)
    
    def show(self):
        plt.tight_layout()
        plt.show()
        return self.selected_traces
```

**Key Features**:

- LaTeX rendering for publication-quality labels
- Log-scale normalization for dynamic range
- Interactive checkbox selection per trace
- Visual feedback (alpha transparency) for selected/deselected states
- Spatial profile plot for context

---

### 4. Spatial Profile Modeling for Optimal Extraction

**Challenge**: Model spatial profile of trace for optimal (variance-weighted) extraction. Must handle seeing variations, extended sources, and irregular profiles.

**Options Evaluated**:

1. **Gaussian Profile**
   - **Model**: $P(y) = A \exp\left[-\frac{(y - y_0)^2}{2\sigma^2}\right]$
   - **Pros**: Simple; 2 parameters (center, width); analytic derivatives
   - **Cons**: Poor for extended/irregular sources
   - **Reference**: Horne 1986 optimal extraction

2. **Moffat Profile**
   - **Model**: $P(y) = A \left[1 + \left(\frac{y - y_0}{\alpha}\right)^2\right]^{-\beta}$
   - **Pros**: Better fits for seeing-limited PSFs (wings)
   - **Cons**: 3 parameters; more complex fitting
   - **Reference**: PypeIt extraction code

3. **Empirical Profile (Data-Driven)**
   - **Model**: Direct extraction of spatial profile from continuum regions
   - **Pros**: No parametric assumptions; handles irregular shapes
   - **Cons**: Requires high SNR; sensitive to noise
   - **Reference**: Massey Ch. 7 on profile extraction

**Decision**: **Gaussian Profile with Empirical Fallback**

**Rationale**:

- Gaussian sufficient for typical seeing-limited KOSMOS observations (~2" FWHM)
- Simple, fast fitting with scipy.optimize.curve_fit
- For low SNR or poor fits (χ² > threshold), fall back to empirical profile (median of continuum regions)
- User clarification emphasized not using emission lines for profile estimation

**Implementation Strategy**:

```python
def fit_spatial_profile(spectrum_2d, trace_center, continuum_mask):
    """
    Fit Gaussian profile to spatial direction, excluding emission lines.
    
    Parameters
    ----------
    spectrum_2d : ndarray
        2D spectrum (spatial x spectral)
    trace_center : ndarray
        Trace center position vs. wavelength
    continuum_mask : ndarray
        Boolean mask: True for continuum regions (no emission lines)
    
    Returns
    -------
    profile : ndarray
        Normalized spatial profile (1D array)
    profile_func : callable
        Gaussian function for interpolation
    """
    # Extract continuum regions only
    continuum_2d = spectrum_2d[:, continuum_mask]
    
    # Collapse spectrally to get average spatial profile
    spatial_profile = np.median(continuum_2d, axis=1)
    
    # Fit Gaussian
    y_pixels = np.arange(len(spatial_profile))
    try:
        popt, pcov = curve_fit(
            gaussian, y_pixels, spatial_profile,
            p0=[spatial_profile.max(), trace_center.mean(), 3.0],  # A, y0, sigma
            bounds=([0, trace_center.mean()-10, 0.5],
                    [np.inf, trace_center.mean()+10, 20.0])
        )
        A, y0, sigma = popt
        profile_func = lambda y: gaussian(y, A, y0, sigma)
        
        # Check fit quality
        chi_sq = np.sum((spatial_profile - profile_func(y_pixels))**2) / len(y_pixels)
        if chi_sq > CHI_SQ_THRESHOLD:
            # Fall back to empirical profile
            profile_func = interp1d(y_pixels, spatial_profile, kind='cubic', fill_value='extrapolate')
    
    except RuntimeError:
        # Fit failed, use empirical
        profile_func = interp1d(y_pixels, spatial_profile, kind='cubic', fill_value='extrapolate')
    
    return profile_func
```

---

### 5. Wavelength Solution Fitting (Polynomial Order & Robust Fitting)

**Challenge**: Fit wavelength solution λ(pixel) from identified arc lines. Must handle outliers (misidentified lines) and choose appropriate polynomial order.

**Massey Guidance** (Ch. 5):

- Typical polynomial orders: 3-5 for low dispersion, 5-7 for high dispersion
- Use sigma-clipping (3σ) to reject outliers iteratively
- Validate with RMS residuals < 0.1 Å target

**pyKOSMOS Approach**:

- Uses np.polyfit with degree=3-5
- Manual outlier rejection based on residuals

**PypeIt Approach**:

- Legendre polynomial with iterative sigma-clipping
- Cross-validation to select order

**Decision**: **Chebyshev Polynomial with Iterative Sigma-Clipping**

**Rationale**:

- Chebyshev polynomials better conditioned than standard polynomials for fitting
- numpy.polynomial.chebyshev.chebfit provides robust implementation
- Iterative sigma-clipping (3σ) handles outliers per Massey recommendation
- Order selection via BIC (Bayesian Information Criterion) balances fit quality and complexity

**Implementation**:

```python
from numpy.polynomial import chebyshev as C
from astropy.stats import sigma_clip

def fit_wavelength_solution(pixel_positions, wavelengths, max_order=7):
    """
    Fit Chebyshev polynomial wavelength solution with outlier rejection.
    
    Parameters
    ----------
    pixel_positions : ndarray
        Pixel positions of identified arc lines
    wavelengths : ndarray
        Reference wavelengths of identified lines
    max_order : int
        Maximum polynomial order to consider
    
    Returns
    -------
    wave_solution : Chebyshev polynomial object
    rms_residual : float (Angstroms)
    n_lines_used : int
    """
    best_bic = np.inf
    best_fit = None
    
    for order in range(3, max_order + 1):
        # Normalize pixel values to [-1, 1] for Chebyshev
        pix_norm = 2 * (pixel_positions - pixel_positions.min()) / \
                   (pixel_positions.max() - pixel_positions.min()) - 1
        
        # Iterative sigma-clipping fit
        mask = np.ones(len(pixel_positions), dtype=bool)
        for iteration in range(5):
            fit = C.chebfit(pix_norm[mask], wavelengths[mask], order)
            residuals = wavelengths[mask] - C.chebval(pix_norm[mask], fit)
            
            # Sigma-clip outliers
            clipped = sigma_clip(residuals, sigma=3.0, maxiters=1)
            mask[mask] = ~clipped.mask
            
            if not np.any(clipped.mask):
                break  # Converged
        
        # Compute BIC
        n_params = order + 1
        n_data = np.sum(mask)
        rms = np.sqrt(np.mean(residuals[~clipped.mask]**2))
        bic = n_data * np.log(rms**2) + n_params * np.log(n_data)
        
        if bic < best_bic:
            best_bic = bic
            best_fit = (fit, rms, n_data, order)
    
    wave_solution, rms_residual, n_lines_used, final_order = best_fit
    
    return wave_solution, rms_residual, n_lines_used, final_order
```

**Validation Criteria**:

- RMS residuals < 0.1 Å (FR-008 requirement)
- At least 80% of lines retained after sigma-clipping
- Visual inspection plot: residuals vs. wavelength

---

### 6. Customizable Spatial and Spectral Binning

**Challenge**: Allow user to customize binning to improve SNR without losing resolution. Must handle rectangular bins and preserve wavelength calibration.

**User Requirements** (from input):

- "custom define/select/edit the binning in spatial and spectral direction"
- Spatial binning improves SNR for faint extended sources
- Spectral binning trades resolution for SNR

**Options**:

1. **Pre-Extraction Binning** (2D spectrum)
   - **Method**: Rebin 2D array before trace extraction
   - **Pros**: Simple; reduces memory; faster extraction
   - **Cons**: Loses spatial resolution for trace fitting; must re-wavelength-calibrate

2. **Post-Extraction Binning** (1D spectrum)
   - **Method**: Bin extracted 1D spectrum after wavelength calibration
   - **Pros**: Preserves full resolution data; flexible binning per spectrum
   - **Cons**: Requires careful variance propagation

**Decision**: **Hybrid Approach**

- **Spatial binning**: Pre-extraction (2D) for computational efficiency
- **Spectral binning**: Post-extraction (1D) to preserve wavelength calibration

**Implementation**:

```python
# Spatial binning (pre-extraction)
def bin_spatial(spectrum_2d, bin_factor=2):
    """Bin 2D spectrum along spatial axis."""
    n_spatial, n_spectral = spectrum_2d.shape
    n_spatial_binned = n_spatial // bin_factor
    binned = np.zeros((n_spatial_binned, n_spectral))
    
    for i in range(n_spatial_binned):
        binned[i, :] = np.sum(spectrum_2d[i*bin_factor:(i+1)*bin_factor, :], axis=0)
    
    return binned

# Spectral binning (post-extraction)
def bin_spectral(spectrum_1d, wavelength, bin_width_angstrom=2.0):
    """
    Bin 1D spectrum to specified wavelength width, preserving variance.
    
    Uses specutils.manipulation.bin_function for proper variance propagation.
    """
    from specutils import Spectrum1D
    from specutils.manipulation import bin_function
    import astropy.units as u
    
    spec = Spectrum1D(
        spectral_axis=wavelength * u.Angstrom,
        flux=spectrum_1d.flux * u.count,
        uncertainty=spectrum_1d.uncertainty
    )
    
    binned_spec = bin_function(spec, bin_width=bin_width_angstrom * u.Angstrom)
    
    return binned_spec
```

**Configuration Parameters** (YAML):

```yaml
binning:
  spatial:
    enabled: false
    factor: 2  # Bin 2 pixels -> 1 (increases SNR by sqrt(2))
  spectral:
    enabled: false
    width_angstrom: 2.0  # Bin to 2 Angstrom width
```

---

### 7. Bias Frame Combination (Median vs. Mean)

**Challenge**: Combine multiple bias frames into master bias. Must reject outliers (hot pixels, cosmic rays).

**Massey Guidance** (Ch. 2):

- Use median combine for bias frames (robust to outliers)
- Minimum 5 bias frames recommended
- Check for bias level consistency across frames

**User Requirement**: "biases should be medianed/statistically added"

**Decision**: **Median Combine with Sigma-Clipping**

**Rationale**:

- Median naturally rejects outliers (hot pixels, cosmic rays)
- Sigma-clipping before median handles systematic deviations
- Preserves read noise statistics
- Aligns with pyKOSMOS/PypeIt practices

**Implementation**:

```python
from astropy.stats import sigma_clipped_stats
from astropy.nddata import CCDData
import astropy.units as u

def combine_bias_frames(bias_file_list):
    """
    Combine bias frames using sigma-clipped median.
    
    Parameters
    ----------
    bias_file_list : list of str
        Paths to bias FITS files
    
    Returns
    -------
    master_bias : CCDData
        Combined bias frame with provenance metadata
    """
    bias_data = []
    for bias_file in bias_file_list:
        ccd = CCDData.read(bias_file, unit=u.adu)
        bias_data.append(ccd.data)
    
    bias_stack = np.array(bias_data)
    
    # Sigma-clipped median along stack axis
    master_bias_data = np.zeros_like(bias_stack[0])
    for i in range(bias_stack.shape[1]):
        for j in range(bias_stack.shape[2]):
            pixel_values = bias_stack[:, i, j]
            clipped = sigma_clip(pixel_values, sigma=3.0, maxiters=2)
            master_bias_data[i, j] = np.median(pixel_values[~clipped.mask])
    
    # Alternative: use median directly if sigma_clip not needed per-pixel
    # master_bias_data = np.median(bias_stack, axis=0)
    
    # Create CCDData object with provenance
    master_bias = CCDData(
        master_bias_data,
        unit=u.adu,
        meta={'NCOMBINE': len(bias_file_list),
              'BIASLVL': np.median(master_bias_data),
              'HISTORY': f'Combined {len(bias_file_list)} bias frames via sigma-clipped median'}
    )
    
    # Validate: check bias level consistency
    bias_levels = [np.median(data) for data in bias_data]
    if np.std(bias_levels) > 10.0:  # ADU threshold
        warnings.warn(f"Bias level variation {np.std(bias_levels):.1f} ADU exceeds 10 ADU")
    
    return master_bias
```

---

### 8. Arc Lamp Identification from Filenames

**Challenge**: Automatically determine which arc lamps were used (He-Ne-Ar, Ar, Kr, etc.) from FITS filenames to load correct linelist.

**User Requirement**: "for arcs, use the file names to figure out which lamps are being used for wavelength calibration"

**Strategy**:

1. Parse filename for lamp keywords: "henear", "argon", "krypton", "thar", "cuar"
2. Fallback to FITS header keyword (LAMPID, OBJECT, IMAGETYP)
3. Map to pyKOSMOS linelist files in resources/pykosmos_reference/linelists/

**Implementation**:

```python
import re

LAMP_KEYWORD_MAP = {
    'henear': 'apohenear.dat',
    'he-ne-ar': 'apohenear.dat',
    'argon': 'argon.dat',
    'ar': 'argon.dat',
    'krypton': 'krypton.dat',
    'kr': 'krypton.dat',
    'thar': 'thar.dat',
    'th-ar': 'thar.dat',
    'cuar': 'cuar.dat',
    'cu-ar': 'cuar.dat',
}

def identify_arc_lamp(arc_filename, fits_header=None):
    """
    Identify arc lamp type from filename or FITS header.
    
    Parameters
    ----------
    arc_filename : str
        Filename of arc lamp exposure (e.g., "arc_henear_001.fits")
    fits_header : astropy.io.fits.Header, optional
        FITS header with LAMPID or OBJECT keyword
    
    Returns
    -------
    linelist_file : str
        Path to linelist file in resources/pykosmos_reference/linelists/
    """
    # Try filename first
    filename_lower = arc_filename.lower()
    for keyword, linelist in LAMP_KEYWORD_MAP.items():
        if keyword in filename_lower:
            return os.path.join(RESOURCES_DIR, 'pykosmos_reference', 'linelists', linelist)
    
    # Try FITS header
    if fits_header is not None:
        for key in ['LAMPID', 'OBJECT', 'IMAGETYP']:
            if key in fits_header:
                header_value = fits_header[key].lower()
                for keyword, linelist in LAMP_KEYWORD_MAP.items():
                    if keyword in header_value:
                        return os.path.join(RESOURCES_DIR, 'pykosmos_reference', 'linelists', linelist)
    
    # Default to He-Ne-Ar (most common for KOSMOS)
    warnings.warn(f"Could not identify arc lamp from {arc_filename}, defaulting to He-Ne-Ar")
    return os.path.join(RESOURCES_DIR, 'pykosmos_reference', 'linelists', 'apohenear.dat')
```

---

### 9. Saturation Detection in Flats and Arcs

**Challenge**: Detect saturated pixels in flat fields and arc lamps to flag bad data or adjust exposure times.

**User Requirement**: "Flats and Arcs are likely unsaturated, but if they are saturated, figure that out as well"

**Strategy**:

1. Check against saturation level from FITS header (SATURATE keyword) or detector specs
2. If not available, use empirical threshold (e.g., 90% of 16-bit ADC max = 58,982 ADU)
3. Flag frames with >1% saturated pixels as problematic
4. For arcs: saturated lines are useless for wavelength calibration (exclude from fit)
5. For flats: saturated regions cannot normalize properly (mask bad pixels)

**Implementation**:

```python
def detect_saturation(ccd_data, saturation_level=None):
    """
    Detect saturated pixels in CCD data.
    
    Parameters
    ----------
    ccd_data : CCDData
        CCD image to check
    saturation_level : float, optional
        Saturation threshold in ADU. If None, read from header or use default.
    
    Returns
    -------
    saturation_mask : ndarray (bool)
        True for saturated pixels
    fraction_saturated : float
        Fraction of pixels saturated
    """
    if saturation_level is None:
        # Try FITS header
        if 'SATURATE' in ccd_data.meta:
            saturation_level = ccd_data.meta['SATURATE']
        else:
            # Default: 90% of 16-bit max
            saturation_level = 0.9 * 65535
    
    saturation_mask = ccd_data.data >= saturation_level
    fraction_saturated = np.sum(saturation_mask) / saturation_mask.size
    
    if fraction_saturated > 0.01:  # More than 1% saturated
        warnings.warn(
            f"Frame has {fraction_saturated*100:.2f}% saturated pixels "
            f"(threshold: {saturation_level} ADU)"
        )
    
    return saturation_mask, fraction_saturated
```

---

### 10. LaTeX-Styled Matplotlib Plots

**Challenge**: Generate publication-quality diagnostic plots with LaTeX rendering for all intermediate products.

**User Requirement**: "Make pretty latex-fied matplotlib plots to show all visual products"

**Strategy**:

- Enable LaTeX rendering globally with `plt.rcParams['text.usetex'] = True`
- Use serif fonts (`font.family = 'serif'`)
- Format labels with LaTeX math mode: `r'$\lambda$ (\r\AA)'`
- Use colorblind-friendly colormaps (viridis, plasma)
- Save high-DPI figures (300 dpi) for publication

**Implementation Pattern**:

```python
import matplotlib.pyplot as plt
import matplotlib as mpl

# Global LaTeX styling configuration
def setup_latex_plots():
    """Configure matplotlib for LaTeX-styled plots."""
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

# Example: 2D spectrum plot
def plot_2d_spectrum(spectrum_2d, wavelength, spatial, title="2D Spectrum"):
    """Plot 2D spectrum with LaTeX labels."""
    setup_latex_plots()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(
        spectrum_2d, aspect='auto', cmap='viridis',
        extent=[wavelength.min(), wavelength.max(), spatial.min(), spatial.max()],
        norm=mpl.colors.LogNorm()
    )
    
    ax.set_xlabel(r'Wavelength (\AA)')
    ax.set_ylabel(r'Spatial Position (pixels)')
    ax.set_title(rf'\textbf{{{title}}}')
    
    cbar = plt.colorbar(im, ax=ax, label=r'Flux (counts)')
    
    plt.tight_layout()
    return fig

# Example: Wavelength calibration residuals
def plot_wavelength_residuals(pixel_pos, wavelength, wave_solution):
    """Plot wavelength solution fit and residuals."""
    setup_latex_plots()
    
    fig, (ax_fit, ax_resid) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])
    
    # Fit
    wave_model = wave_solution(pixel_pos)
    ax_fit.plot(pixel_pos, wavelength, 'o', label='Arc Lines', markersize=5, alpha=0.7)
    ax_fit.plot(pixel_pos, wave_model, '-', label='Fit', linewidth=2)
    ax_fit.set_ylabel(r'Wavelength (\AA)')
    ax_fit.set_title(r'\textbf{Wavelength Solution}')
    ax_fit.legend()
    ax_fit.grid(True, alpha=0.3)
    
    # Residuals
    residuals = wavelength - wave_model
    rms = np.sqrt(np.mean(residuals**2))
    ax_resid.plot(pixel_pos, residuals, 'o', markersize=5, alpha=0.7)
    ax_resid.axhline(0, color='k', linestyle='--', linewidth=1)
    ax_resid.axhline(0.1, color='r', linestyle=':', linewidth=1, label=r'$\pm 0.1$ \AA')
    ax_resid.axhline(-0.1, color='r', linestyle=':', linewidth=1)
    ax_resid.set_xlabel(r'Pixel Position')
    ax_resid.set_ylabel(r'Residual (\AA)')
    ax_resid.set_title(rf'\textbf{{RMS Residual: {rms:.3f} \AA}}')
    ax_resid.legend()
    ax_resid.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

---

## Technology Stack Summary

| Component | Technology | Rationale |
| --------- | ---------- | --------- |
| **Language** | Python 3.10+ | Constitution requirement; astropy/specutils compatibility |
| **FITS I/O** | astropy.io.fits, CCDData | Standard astronomy FITS handling; provenance support |
| **Spectrum Handling** | specutils.Spectrum1D | Standard 1D spectrum representation with units/WCS |
| **Array Operations** | numpy ≥1.23 | Foundation for all numerical computation |
| **Signal Processing** | scipy.signal, scipy.ndimage | Cross-correlation, filtering, peak-finding |
| **Optimization** | scipy.optimize | Profile fitting (Gaussian), wavelength solution |
| **Polynomial Fitting** | numpy.polynomial.chebyshev | Robust wavelength solution fitting |
| **Statistics** | astropy.stats | Sigma-clipping, robust statistics |
| **Configuration** | pyyaml | YAML config file parsing |
| **Plotting** | matplotlib ≥3.6 | LaTeX-styled plots, interactive widgets |
| **Testing** | pytest, astropy-pytest-plugins | Unit/integration tests with FITS fixtures |

---

## Resolved Unknowns

All "NEEDS CLARIFICATION" items from Technical Context now resolved:

1. ✅ **Trace Detection**: Cross-correlation with Gaussian template (excluding emission lines)
2. ✅ **AB Subtraction**: Iterative sky modeling with outlier rejection
3. ✅ **Interactive Viewer**: Matplotlib CheckButtons widget pattern from pyKOSMOS notebooks
4. ✅ **Spatial Profile**: Gaussian fit with empirical fallback
5. ✅ **Wavelength Fitting**: Chebyshev polynomial with BIC order selection, sigma-clipping
6. ✅ **Binning**: Hybrid spatial (pre-extraction) + spectral (post-extraction)
7. ✅ **Bias Combination**: Sigma-clipped median
8. ✅ **Arc Lamp ID**: Filename parsing with LAMP_KEYWORD_MAP
9. ✅ **Saturation Detection**: Header-based or empirical threshold (90% of max ADU)
10. ✅ **LaTeX Plots**: `text.usetex = True`, serif fonts, high-DPI output

---

## Next Steps

Proceed to **Phase 1: Design**

- Generate [data-model.md](./data-model.md) with entity definitions
- Generate [contracts/](./contracts/) with CLI and config schemas
- Generate [quickstart.md](./quickstart.md) with test scenarios
