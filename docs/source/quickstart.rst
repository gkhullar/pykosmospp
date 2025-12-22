Quick Start Guide
==================

Get started with pyKOSMOS++ in 5 minutes: install, prepare data, and run your first spectroscopic reduction.

**Author:** Gourav Khullar


Prerequisites
-------------

* Python ≥ 3.10 installed
* pyKOSMOS installed (see :doc:`installation`)
* KOSMOS FITS data organized in directories


Step 1: Organize Your Data
---------------------------

Arrange your FITS files in subdirectories by frame type:

.. code-block:: text

    data/2024-01-15/galaxy_NGC1234/
    ├── biases/
    │   ├── bias_001.fits
    │   ├── bias_002.fits
    │   └── bias_003.fits
    ├── flats/
    │   ├── flat_001.fits
    │   ├── flat_002.fits
    │   └── flat_003.fits
    ├── arcs/
    │   ├── arc_HeNeAr_001.fits
    │   └── arc_HeNeAr_002.fits
    └── science/
        ├── galaxy_NGC1234_001.fits
        └── galaxy_NGC1234_002.fits

**Frame Classification Requirements:**

* FITS headers must contain ``IMAGETYP`` keyword with values:
  
  * ``'BIAS'`` or ``'bias'`` → bias frames
  * ``'FLAT'`` or ``'flat'`` → flat fields
  * ``'ARC'`` or ``'arc'`` or ``'COMP'`` → arc lamps
  * ``'OBJECT'`` or ``'object'`` → science frames

* Minimum required frames:
  
  * ≥3 bias frames
  * ≥3 flat frames
  * ≥1 arc frame
  * ≥1 science frame


Step 2: Run the Pipeline
-------------------------

**Method 1: Python Script**

Create ``reduce.py``:

.. code-block:: python

    from pykosmos_spec_ai.pipeline import PipelineRunner
    from pathlib import Path

    # Configure pipeline
    runner = PipelineRunner(
        input_dir=Path("data/2024-01-15/galaxy_NGC1234"),
        output_dir=Path("reduced_output"),
        mode="batch",  # Automatic processing
        max_traces=5   # Maximum traces per frame
    )

    # Run reduction
    reduced_data_list = runner.run()

    # Print summary
    print(f"\nProcessed {len(reduced_data_list)} frame(s)")
    for reduced_data in reduced_data_list:
        print(f"  {reduced_data.source_frame.file_path.name}")
        print(f"    • Traces: {len(reduced_data.spectra_1d)}")
        print(f"    • Grade: {reduced_data.quality_metrics.overall_grade}")
        print(f"    • Median SNR: {reduced_data.quality_metrics.median_snr:.2f}")

Run the script:

.. code-block:: bash

    python reduce.py

**Method 2: Interactive Python Session**

.. code-block:: python

    >>> from pykosmos_spec_ai.pipeline import PipelineRunner
    >>> from pathlib import Path
    >>> 
    >>> runner = PipelineRunner(
    ...     input_dir=Path("data/2024-01-15/galaxy_NGC1234"),
    ...     output_dir=Path("reduced_output"),
    ...     mode="batch"
    ... )
    >>> 
    >>> reduced_data_list = runner.run()
    Processing calibrations...
    Creating master bias from 3 frames...
    Creating master flat from 3 frames...
    Detecting arc lines...
    Fitting wavelength solution (RMS = 0.087 Å)...
    Processing science frames...
    Frame 1/2: galaxy_NGC1234_001.fits
      Detected 1 trace (SNR=15.3)
      Extracting spectrum...
      Grade: Good
    Frame 2/2: galaxy_NGC1234_002.fits
      Detected 1 trace (SNR=12.8)
      Extracting spectrum...
      Grade: Good
    ✓ Pipeline complete!


Step 3: Examine Outputs
------------------------

The pipeline creates the following output structure:

.. code-block:: text

    reduced_output/
    ├── calibrations/
    │   ├── master_bias.fits         # Combined bias frame
    │   ├── master_flat.fits         # Normalized flat field
    │   └── wavelength_solution.pkl  # Arc line fit
    ├── reduced_2d/
    │   ├── galaxy_NGC1234_001_reduced.fits  # Calibrated 2D spectrum
    │   └── galaxy_NGC1234_002_reduced.fits
    ├── spectra_1d/
    │   ├── galaxy_NGC1234_001_trace1.fits   # Extracted 1D spectrum
    │   └── galaxy_NGC1234_002_trace1.fits
    ├── quality_reports/
    │   ├── galaxy_NGC1234_001_quality.yaml  # Quality metrics
    │   └── galaxy_NGC1234_002_quality.yaml
    └── diagnostic_plots/
        ├── wavelength_solution.png          # Wavelength fit + residuals
        ├── galaxy_NGC1234_001_2d.png        # 2D spectrum with traces
        ├── galaxy_NGC1234_001_1d.png        # Extracted 1D spectrum
        └── galaxy_NGC1234_001_profile.png   # Spatial profile


**Key Output Files:**

* ``spectra_1d/*.fits``: Wavelength-calibrated 1D spectra (ready for analysis)
* ``quality_reports/*.yaml``: SNR, wavelength RMS, quality grade
* ``diagnostic_plots/*.png``: Visual QA for each reduction step


Step 4: Inspect Results
------------------------

**Load Extracted Spectrum:**

.. code-block:: python

    from astropy.io import fits
    from pathlib import Path

    # Load 1D spectrum
    spectrum_file = Path("reduced_output/spectra_1d/galaxy_NGC1234_001_trace1.fits")
    with fits.open(spectrum_file) as hdul:
        wavelength = hdul[1].data['wavelength']  # Å
        flux = hdul[1].data['flux']              # Calibrated flux
        uncertainty = hdul[1].data['uncertainty']

    # Plot spectrum
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.plot(wavelength, flux, 'k-', linewidth=0.5)
    plt.fill_between(wavelength, flux - uncertainty, flux + uncertainty, 
                     alpha=0.3, color='gray')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.title('Extracted 1D Spectrum')
    plt.grid(True, alpha=0.3)
    plt.show()

**Check Quality Metrics:**

.. code-block:: python

    import yaml

    # Load quality report
    quality_file = Path("reduced_output/quality_reports/galaxy_NGC1234_001_quality.yaml")
    with open(quality_file, 'r') as f:
        metrics = yaml.safe_load(f)

    print(f"Median SNR: {metrics['median_snr']:.2f}")
    print(f"Wavelength RMS: {metrics['wavelength_rms']:.4f} Å")
    print(f"Overall Grade: {metrics['overall_grade']}")

**View Diagnostic Plots:**

Open PNG files in ``diagnostic_plots/`` directory:

* ``wavelength_solution.png``: Verify wavelength calibration quality
* ``*_2d.png``: Inspect 2D spectrum and trace positions
* ``*_1d.png``: Review extracted 1D spectrum
* ``*_profile.png``: Check spatial profile for optimal extraction


Common Adjustments
------------------

**Adjust Trace Detection Sensitivity**

For fainter sources, lower the SNR threshold:

.. code-block:: python

    runner = PipelineRunner(
        input_dir=Path("data/faint_galaxy"),
        output_dir=Path("reduced_output"),
        mode="batch",
        detection_params={'min_snr': 2.0}  # Default: 3.0
    )

**Use Interactive Mode**

For visual trace selection:

.. code-block:: python

    runner = PipelineRunner(
        input_dir=Path("data/multi_object_slit"),
        output_dir=Path("reduced_output"),
        mode="interactive"  # Launches GUI for trace selection
    )

**Custom Configuration**

Create ``custom_config.yaml``:

.. code-block:: yaml

    detector:
      gain: 1.0
      readnoise: 3.5
      saturate: 50000

    wavelength:
      max_order: 7
      initial_dispersion: 1.0

    extraction:
      expected_fwhm: 4.0
      min_snr: 3.0
      sky_buffer: 30

Load custom config:

.. code-block:: python

    from pykosmos_spec_ai.io.config import PipelineConfig

    config = PipelineConfig.from_yaml("custom_config.yaml")
    runner = PipelineRunner(
        input_dir=Path("data/2024-01-15/galaxy_NGC1234"),
        output_dir=Path("reduced_output"),
        config=config
    )


Troubleshooting
---------------

**Error: "No FITS files found"**

Check directory structure and file extensions:

.. code-block:: bash

    ls data/2024-01-15/galaxy_NGC1234/science/*.fits

Ensure files have ``.fits`` extension (not ``.fit`` or ``.FITS``).

**Error: "Insufficient calibration frames"**

Pipeline requires ≥3 bias, ≥3 flat, ≥1 arc. Check:

.. code-block:: python

    from pathlib import Path
    
    data_dir = Path("data/2024-01-15/galaxy_NGC1234")
    print(f"Bias frames: {len(list((data_dir / 'biases').glob('*.fits')))}")
    print(f"Flat frames: {len(list((data_dir / 'flats').glob('*.fits')))}")
    print(f"Arc frames: {len(list((data_dir / 'arcs').glob('*.fits')))}")

**Error: "No traces detected"**

Lower detection threshold or check data quality:

.. code-block:: python

    runner = PipelineRunner(
        input_dir=Path("data/faint_galaxy"),
        output_dir=Path("reduced_output"),
        mode="batch",
        detection_params={'min_snr': 2.0}
    )

**Poor Quality Grade (Fair/Poor)**

Common causes:

* Low SNR → Increase integration time
* Poor wavelength calibration → Check arc lamp exposure
* Bad calibrations → Review master bias/flat statistics

View diagnostic plots for detailed inspection.


Next Steps
----------

* **Detailed Tutorial**: :doc:`tutorial` - Jupyter notebook with step-by-step walkthrough
* **Configuration Guide**: :doc:`user_guide/configuration` - Customize pipeline parameters
* **Quality Assessment**: :doc:`user_guide/quality` - Interpret quality metrics
* **Batch Processing**: :doc:`user_guide/batch_processing` - Process full observing runs
* **API Reference**: :doc:`api/index` - Complete function documentation


Need Help?
----------

* **Issues**: `GitHub Issues <https://github.com/gkhullar/pykosmospp/issues>`_
* **Discussions**: `GitHub Discussions <https://github.com/gkhullar/pykosmospp/discussions>`_
* **Author**: Gourav Khullar
* **FAQ**: :doc:`faq`
