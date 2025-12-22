# Examples Directory

This directory contains tutorial notebooks and example scripts for pyKOSMOS++.

**Author:** Gourav Khullar

## Contents

### Jupyter Notebooks

#### `tutorial.ipynb` - Complete Pipeline Walkthrough

Comprehensive tutorial demonstrating the full pyKOSMOS_specllm reduction workflow by Gourav Khullar:

1. **Introduction & Setup** - Imports, configuration, detector specs
2. **Data Exploration** - FITS inspection, header keywords, visualization
3. **Calibration Creation** - Master bias/flat with validation
4. **Wavelength Calibration** - Arc line detection, catalog matching, polynomial fitting
5. **Trace Detection & Extraction** - Cross-correlation, optimal extraction
6. **Quality Assessment** - Metrics computation, grading, diagnostic plots
7. **Advanced Parameters** - Custom configurations, sensitivity tuning
8. **Batch Processing** - Automated pipeline for multiple observations

**Usage:**

```bash
# Launch Jupyter notebook
jupyter notebook tutorial.ipynb

# Or use JupyterLab
jupyter lab tutorial.ipynb
```

**Requirements:**

- pyKOSMOS_specllm installed (`pip install -e .`)
- Sample KOSMOS data in `data/` subdirectory
- Jupyter notebook server (`pip install jupyter`)

**Expected Runtime:** 5-10 minutes (depending on dataset size)

### Sample Data

#### `data/` - Example KOSMOS Observations

Synthetic test dataset for tutorial execution:

- `biases/` - 3 bias frames (1024×2048 pixels)
- `flats/` - 3 flat field frames
- `arcs/` - 1 He-Ne-Ar arc lamp frame
- `science/` - 1 galaxy longslit observation

**Data Generation:**

To generate synthetic test data:

```python
from pykosmos_spec_ai.testing.synthetic import generate_test_dataset

generate_test_dataset(
    output_dir="examples/data",
    n_bias=3,
    n_flat=3,
    n_arc=1,
    n_science=1
)
```

**Note:** Full KOSMOS datasets are typically 1-10 GB per night. The example dataset is <50 MB for tutorial purposes.

## Running Examples

### Quick Start

Process example data with default parameters:

```bash
cd examples
python ../scripts/reduce_example.py --input data --output reduced_output
```

### Interactive Tutorial

Step through the Jupyter notebook for detailed explanations:

```bash
jupyter notebook tutorial.ipynb
```

Each cell includes:
- Code with inline comments
- Expected outputs
- Visualizations
- Quality checks

### Batch Processing

Process multiple observations automatically:

```python
from pykosmos_spec_ai.pipeline import PipelineRunner
from pathlib import Path

runner = PipelineRunner(
    input_dir=Path("data"),
    output_dir=Path("reduced_output"),
    mode="batch"
)

reduced_data_list = runner.run()
```

## Output Structure

After running the tutorial or examples, outputs are saved to `reduced_output/`:

```
reduced_output/
├── calibrations/
│   ├── master_bias.fits         # Combined bias
│   ├── master_flat.fits         # Normalized flat
│   └── wavelength_solution.pkl  # Arc line fit
├── reduced_2d/
│   └── science_001_reduced.fits # Calibrated 2D spectrum
├── spectra_1d/
│   └── science_001_trace1.fits  # Extracted 1D spectrum (wavelength calibrated)
├── quality_reports/
│   └── science_001_quality.yaml # SNR, RMS, grade
└── diagnostic_plots/
    ├── wavelength_solution.png  # Fit + residuals
    ├── science_001_2d.png       # 2D spectrum with traces
    ├── science_001_1d.png       # Extracted 1D spectrum
    └── science_001_profile.png  # Spatial profile
```

## Troubleshooting

### Jupyter Kernel Issues

If the notebook kernel crashes:

```bash
# Reinstall kernel
pip install ipykernel
python -m ipykernel install --user --name pykosmos
```

### Missing Sample Data

If `data/` directory is empty:

1. Download sample data from [GitHub Releases](https://github.com/your-repo/pykosmos_spec_ai/releases)
2. Or generate synthetic data using the script above
3. Or use your own KOSMOS observations

### Import Errors

Ensure pyKOSMOS is installed:

```bash
pip install -e ..  # From examples/ directory
python -c "import pykosmos_spec_ai; print('OK')"
```

## Contributing Examples

To contribute new examples:

1. **Create notebook**: Use `tutorial.ipynb` as template
2. **Add documentation**: Include markdown cells explaining each step
3. **Test execution**: Run entire notebook from clean kernel
4. **Clear outputs**: Clear all cell outputs before committing (`Cell > All Output > Clear`)
5. **Submit PR**: Include description of example use case

See `docs/developer/contributing.rst` for full guidelines.

### Additional Resources

- **Author**: Gourav Khullar
- **Full Documentation**: [Read the Docs](https://pykosmos-specllm.readthedocs.io)
- **API Reference**: `docs/source/api/`
- **User Guides**: `docs/source/user_guide/`
- **FAQ**: `docs/source/faq.rst`
