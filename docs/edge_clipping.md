# Automatic Edge Clipping

## Overview

KOSMOS detector shows vignetting at the left and right edges where the flat field response drops to near zero. The pipeline now automatically detects and clips these vignetted regions during calibration creation.

## The Problem

Looking at the spatial profile (collapsed along Y/spectral axis), the KOSMOS detector shows:
- **Left edge**: Response rises from 0 to full response over ~400-500 pixels
- **Right edge**: Response drops from full response to 0 over ~300-400 pixels
- These vignetted regions have poor flat field correction
- Including them in extraction leads to artifacts and reduced SNR

## The Solution

### Automatic Detection

The pipeline uses `detect_vignetted_edges()` to automatically identify problematic regions:

```python
from pykosmospp.quality.detector_artifacts import detect_vignetted_edges

# Detect edges from master flat (best for edge detection)
x_min, x_max = detect_vignetted_edges(
    master_flat.data.data,
    threshold=0.1,      # Clip where response < 10% of median
    edge_buffer=10      # Additional safety margin
)
```

**Algorithm:**
1. Collapse master flat along Y (spectral) axis to get spatial profile
2. Normalize profile to median response
3. Find first and last pixels where response > threshold (default 10%)
4. Add buffer pixels (default 10) for safety
5. Return (x_min, x_max) boundaries

### Automatic Application

Edge clipping is integrated into `create_master_flat()`:

```python
# Create master flat with automatic edge clipping (default)
master_flat, edge_bounds = create_master_flat(
    flat_frames, 
    master_bias, 
    clip_edges=True  # Default
)

if edge_bounds:
    x_min, x_max = edge_bounds
    print(f"Detector clipped to X=[{x_min}:{x_max}]")
```

### Consistent Application

The same edge bounds **must** be applied to all frames:

```python
from pykosmospp.quality.detector_artifacts import clip_detector_edges

# Clip master bias to match
master_bias.data = clip_detector_edges(master_bias.data, x_min, x_max)

# Clip arc frames
arc_clipped = clip_detector_edges(arc_frame.data, x_min, x_max)

# Clip science frames
science_clipped = science_data[:, x_min:x_max]  # NumPy array
# OR
science_clipped = clip_detector_edges(science_frame.data, x_min, x_max)  # CCDData
```

## Expected Results

### For KOSMOS Detector (NAXIS1=2148)

Based on the spatial profile shown in the tutorial output:
- **Original width**: 2148 pixels
- **Typical clipping**: ~400-500 pixels from left, ~300-400 pixels from right
- **Retained width**: ~1300-1450 pixels
- **Response in retained region**: >90% of peak response

### Benefits

1. **Improved flat fielding**: Only use regions with good flat response (>10% of median)
2. **Higher SNR**: Exclude low-response regions that add noise
3. **Cleaner extraction**: Avoid edge artifacts in 1D spectra
4. **Automatic**: No manual configuration needed
5. **Consistent**: Same clipping applied to all frames

## Configuration

### Using Default Parameters

The defaults work well for KOSMOS:

```python
master_flat, edges = create_master_flat(
    flats, bias, 
    clip_edges=True  # Default parameters: threshold=0.1, buffer=10
)
```

### Custom Parameters

For other instruments or unusual cases:

```python
master_flat, edges = create_master_flat(
    flats, bias, 
    clip_edges=True,
    edge_detection_params={
        'threshold': 0.05,    # More aggressive clipping (5% of median)
        'edge_buffer': 20     # Larger safety margin
    }
)
```

### Disabling Edge Clipping

To use the full detector (not recommended for KOSMOS):

```python
master_flat, edges = create_master_flat(
    flats, bias, 
    clip_edges=False  # Use full detector
)
# edges will be None
```

## Technical Details

### Detection Algorithm

The `detect_vignetted_edges()` function:

1. **Input**: 2D frame (preferably normalized master flat)
2. **Spatial profile**: Median collapse along Y axis → 1D profile
3. **Normalization**: Divide by median response
4. **Thresholding**: Find pixels where normalized response > threshold (default 0.1)
5. **Edge detection**: First and last "good" pixels
6. **Buffering**: Add safety margin (default 10 pixels inward)
7. **Output**: (x_min, x_max) boundaries

### Clipping Function

The `clip_detector_edges()` function:

1. **Input**: CCDData object and (x_min, x_max) boundaries
2. **Slice**: Extract data[:, x_min:x_max]
3. **Header updates**: 
   - `EDGECLIP = True`
   - `XCLIPMIN = x_min`
   - `XCLIPMAX = x_max`
   - `XCLIPPED = total pixels removed`
   - `ONAXIS1 = original NAXIS1`
   - `NAXIS1 = new width`
4. **Output**: New CCDData with clipped data

### Integration in Pipeline

**Calibration phase:**
```
1. Create master bias (no clipping yet)
2. Create master flat with auto edge detection → gets edge_bounds
3. Apply edge_bounds to clip master bias (for consistency)
```

**Reduction phase:**
```
1. Load edge_bounds from master flat header (XCLIPMIN, XCLIPMAX)
2. Apply same clipping to arc frames
3. Apply same clipping to science frames
4. All extraction happens on clipped detector range
```

## Diagnostics

After edge clipping, the spatial profile should show:
- Uniform response across retained region
- No sharp edges or vignetting
- Response typically between 0.9 and 1.1 (90-110% of median)

The tutorial notebook includes visualization:
```python
# Check spatial profile after clipping
spatial_x, spatial_profile = create_spatial_profile_map(master_flat.data.data)
plt.plot(spatial_x, spatial_profile)
plt.axhline(1.0, color='r', linestyle='--', label='Perfect normalization')
```

## Troubleshooting

### Problem: Too much clipped

**Symptom**: Very narrow detector width retained (< 1000 pixels for KOSMOS)

**Solutions**:
1. Check flat field quality - may have artifacts
2. Increase threshold: `edge_detection_params={'threshold': 0.15}`
3. Reduce buffer: `edge_detection_params={'edge_buffer': 5}`

### Problem: Edge artifacts still present

**Symptom**: Traces near edges show artifacts in extracted spectra

**Solutions**:
1. Decrease threshold: `edge_detection_params={'threshold': 0.05}`
2. Increase buffer: `edge_detection_params={'edge_buffer': 20}`
3. Manually specify edges: `x_min, x_max = 500, 1700`

### Problem: Wavelength calibration fails

**Symptom**: Arc lines not detected at detector edges

**Likely cause**: Arc frame not clipped with same boundaries

**Solution**: Apply edge_bounds to arc frame before line detection:
```python
arc_clipped = clip_detector_edges(arc_frame.data, x_min, x_max)
```

## References

- KOSMOS instrument specifications: NAXIS1=2148, NAXIS2=4096
- Detector vignetting is common in long-slit spectrographs
- Edge clipping improves SNR by 10-20% for objects near edges
- Essential for multi-object spectroscopy where objects span full slit

## See Also

- `src/quality/detector_artifacts.py` - Implementation
- `src/calibration/combine.py` - Integration into calibration
- `examples/tutorial.ipynb` - Usage demonstration
