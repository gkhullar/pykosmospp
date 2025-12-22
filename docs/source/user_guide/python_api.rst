.. _python_api:

==================
Python API
==================

The pyKOSMOS++ pipeline can be used programmatically from Python for custom workflows, batch processing, or integration into larger analysis frameworks.

Overview
========

The main entry points for programmatic access are:

* ``PipelineRunner``: High-level pipeline orchestration
* ``PipelineConfig``: Configuration management
* Individual module functions: Calibration, wavelength, extraction, quality

Installation
============

Install the package to make modules available::

    pip install pykosmospp

Or for development::

    git clone https://github.com/gkhullar/pykosmospp.git
    cd pykosmospp
    pip install -e .


Basic Usage
===========

Simple Pipeline Execution
--------------------------

.. code-block:: python

    from pathlib import Path
    from pykosmospp.pipeline import PipelineRunner
    
    # Initialize pipeline
    runner = PipelineRunner(
        input_dir=Path('data/2024-01-15'),
        output_dir=Path('reduced'),
        mode='batch'
    )
    
    # Run full reduction
    reduced_data_list = runner.run()
    
    # Access results
    for reduced_data in reduced_data_list:
        print(f"Processed: {reduced_data.source_frame.file_path}")
        print(f"  Traces: {len(reduced_data.spectra_1d)}")
        print(f"  SNR: {reduced_data.quality_metrics.median_snr:.2f}")
        print(f"  Grade: {reduced_data.quality_metrics.overall_grade}")


Custom Configuration
--------------------

.. code-block:: python

    from pykosmospp.io.config import PipelineConfig
    
    # Load custom config
    config = PipelineConfig.from_yaml('custom_config.yaml')
    
    # Or create programmatically
    config_dict = {
        'detector': {
            'gain': 1.4,
            'readnoise': 3.7,
            'saturate': 58982
        },
        'trace_detection': {
            'expected_fwhm': 4.0,
            'min_snr': 2.5  # Lower for faint sources
        }
    }
    config = PipelineConfig(config_dict)
    
    # Use with pipeline
    runner = PipelineRunner(
        input_dir=Path('data'),
        output_dir=Path('reduced'),
        config=config
    )


PipelineRunner Class
====================

.. class:: PipelineRunner(input_dir, output_dir, mode='batch', config=None, max_traces=5)

   High-level orchestrator for the complete reduction workflow.
   
   :param Path input_dir: Directory containing subdirectories: biases/, flats/, arcs/, science/
   :param Path output_dir: Output directory for reduced products
   :param str mode: Processing mode - 'batch' (automatic) or 'interactive' (manual trace selection)
   :param PipelineConfig config: Configuration object (uses defaults if None)
   :param int max_traces: Maximum traces to extract per science frame
   
   .. method:: run()
   
      Execute the full reduction pipeline.
      
      :returns: List of ReducedData objects, one per science frame
      :rtype: List[ReducedData]
      :raises CriticalPipelineError: If calibrations fail or critical errors occur
      
      **Workflow:**
      
      1. Discover and classify FITS files
      2. Create master bias and flat
      3. Fit wavelength solution from arc frames
      4. Process each science frame:
         
         * Apply calibrations
         * Detect traces
         * Extract 1D spectra
         * Compute quality metrics
      
      5. Generate diagnostic plots and reports


Module-Level API
================

Calibration Module
------------------

.. code-block:: python

    from pykosmospp.calibration.combine import create_master_bias, create_master_flat
    from pykosmospp.models import BiasFrame, FlatFrame
    
    # Load frames
    bias_frames = [BiasFrame.from_fits(f) for f in bias_files]
    flat_frames = [FlatFrame.from_fits(f) for f in flat_files]
    
    # Create master frames
    master_bias = create_master_bias(bias_frames, method='median')
    master_flat = create_master_flat(flat_frames, master_bias, method='median')
    
    # Access properties
    print(f"Bias level: {master_bias.bias_level:.2f} ADU")
    print(f"Bad pixels: {master_flat.bad_pixel_fraction:.4f}")
    
    # Save to FITS
    master_bias.data.write('master_bias.fits', overwrite=True)

**Functions:**

* ``create_master_bias(frames, method='median')``
* ``create_master_flat(frames, master_bias, method='median')``
* ``detect_cosmic_rays(data, sigma_clip=5.0, contrast=3.0)``


Wavelength Module
-----------------

.. code-block:: python

    from pykosmospp.wavelength.identify import detect_arc_lines
    from pykosmospp.wavelength.match import match_lines_to_catalog
    from pykosmospp.wavelength.fit import fit_wavelength_solution
    import numpy as np
    
    # Load arc frame and collapse to 1D
    arc_frame = ArcFrame.from_fits('arc_001.fits')
    arc_spectrum = np.median(arc_frame.data.data, axis=0)
    
    # Detect lines
    detected_lines = detect_arc_lines(
        arc_spectrum,
        detection_threshold=5.0,
        min_separation=5
    )
    
    # Match to catalog
    matched_lines = match_lines_to_catalog(
        detected_lines,
        linelist_file='resources/linelists/apohenear.dat',
        initial_dispersion=1.0,
        wavelength_range=(4000, 7000)
    )
    
    # Fit wavelength solution
    solution = fit_wavelength_solution(
        matched_lines,
        order_range=(3, 7),  # BIC selects best order
        sigma_clip=3.0
    )
    
    # Evaluate wavelength at pixel positions
    pixel = 1024
    wavelength = solution.wavelength(pixel)
    print(f"Pixel {pixel} = {wavelength:.2f} Å")

**Functions:**

* ``detect_arc_lines(spectrum, detection_threshold, min_separation)``
* ``match_lines_to_catalog(lines, linelist_file, initial_dispersion, wavelength_range)``
* ``fit_wavelength_solution(matched_lines, order_range, sigma_clip)``
* ``apply_wavelength_to_spectrum(spectrum, wavelength_solution)``


Extraction Module
-----------------

.. code-block:: python

    from pykosmospp.extraction.trace import detect_traces_cross_correlation
    from pykosmospp.extraction.sky import estimate_sky_background
    from pykosmospp.extraction.extract import extract_optimal
    
    # Detect traces in calibrated 2D spectrum
    traces = detect_traces_cross_correlation(
        calibrated_data,
        expected_fwhm=4.0,
        min_snr=3.0,
        min_separation=20
    )
    
    # Process each trace
    for trace in traces:
        # Estimate and subtract sky
        sky_2d = estimate_sky_background(
            calibrated_data,
            trace,
            sky_buffer=30
        )
        sky_subtracted = calibrated_data - sky_2d
        
        # Optimal extraction
        spectrum_1d = extract_optimal(
            sky_subtracted,
            trace,
            aperture_width=10
        )
        
        # Access spectrum
        print(f"Trace {trace.trace_id}:")
        print(f"  Flux: {spectrum_1d.flux}")
        print(f"  Uncertainty: {spectrum_1d.uncertainty}")

**Functions:**

* ``detect_traces_cross_correlation(data, expected_fwhm, min_snr)``
* ``fit_spatial_profile(data, trace, profile_type='Gaussian')``
* ``estimate_sky_background(data, trace, sky_buffer)``
* ``extract_optimal(data, trace, aperture_width)``


Quality Module
--------------

.. code-block:: python

    from pykosmospp.quality.metrics import compute_quality_metrics
    from pykosmospp.quality.validate import validate_calibrations
    
    # Validate calibrations
    is_valid = validate_calibrations(master_bias, master_flat)
    if not is_valid:
        print("WARNING: Calibration quality issues detected")
    
    # Compute spectrum quality metrics
    metrics = compute_quality_metrics(spectrum_1d)
    print(f"Median SNR: {metrics['median_snr']:.2f}")
    print(f"Overall grade: {metrics['overall_grade']}")

**Functions:**

* ``validate_calibrations(master_bias, master_flat)``
* ``compute_quality_metrics(spectrum)``


Data Models
===========

Raw Frame Classes
-----------------

.. code-block:: python

    from pykosmospp.models import BiasFrame, FlatFrame, ArcFrame, ScienceFrame
    
    # Load different frame types
    bias = BiasFrame.from_fits('bias_001.fits')
    flat = FlatFrame.from_fits('flat_001.fits')
    arc = ArcFrame.from_fits('arc_001.fits')
    science = ScienceFrame.from_fits('science_001.fits')
    
    # Access common properties
    print(f"Exposure time: {science.exposure_time} s")
    print(f"Observation date: {science.observation_date}")
    print(f"Data shape: {science.data.shape}")
    
    # Access frame-specific properties
    print(f"Target: {science.target_name}")
    print(f"Airmass: {science.airmass}")
    print(f"Arc lamp: {arc.lamp_type}")

**Classes:**

* ``RawFrame``: Base class (abstract)
* ``BiasFrame``: Zero-second exposures
* ``FlatFrame``: Uniform illumination frames
* ``ArcFrame``: Wavelength calibration lamps
* ``ScienceFrame``: Target observations


Calibration Classes
-------------------

.. code-block:: python

    from pykosmospp.models import MasterBias, MasterFlat, CalibrationSet
    
    # Create calibration set
    calib_set = CalibrationSet(
        master_bias=master_bias,
        master_flat=master_flat,
        bad_pixel_mask=bad_pixel_mask
    )
    
    # Apply to science frame
    calibrated_frame = calib_set.apply_to_frame(science_frame)
    
    # Validate
    is_valid = calib_set.validate()

**Classes:**

* ``MasterBias``: Combined bias frame with statistics
* ``MasterFlat``: Normalized flat field
* ``CalibrationSet``: Collection of calibration frames


Spectroscopic Data
------------------

.. code-block:: python

    from pykosmospp.models import Spectrum2D, Trace, WavelengthSolution
    
    # 2D spectrum with trace information
    spectrum_2d = Spectrum2D(
        data=calibrated_data,
        variance=variance,
        mask=mask
    )
    
    # Trace object
    trace = Trace(
        trace_id=1,
        spatial_positions=spatial_pos,
        spectral_pixels=spectral_pixels,
        snr_estimate=5.2
    )
    
    # Wavelength solution
    solution = WavelengthSolution(
        coefficients=coeffs,
        order=5,
        rms_residual=0.08
    )
    
    # Evaluate
    wavelength = solution.wavelength(1024)

**Classes:**

* ``Spectrum2D``: 2D spectroscopic data
* ``Trace``: Spectral trace path
* ``SpatialProfile``: Cross-dispersion profile
* ``WavelengthSolution``: Pixel-to-wavelength mapping


Advanced Examples
=================

Custom Processing Workflow
---------------------------

.. code-block:: python

    from pathlib import Path
    from pykosmospp.pipeline import PipelineRunner
    from pykosmospp.io.organize import discover_fits_files
    from pykosmospp.calibration.combine import create_master_bias
    
    # Discover files
    input_dir = Path('data')
    obs_set = discover_fits_files(input_dir)
    
    # Create calibrations with custom parameters
    master_bias = create_master_bias(
        obs_set.bias_frames,
        method='median'
    )
    
    # Process only specific science frames
    for science_frame in obs_set.science_frames[:5]:  # First 5 only
        # Custom processing here
        pass


Batch Processing Multiple Nights
---------------------------------

.. code-block:: python

    import glob
    from pathlib import Path
    from pykosmospp.pipeline import PipelineRunner
    
    # Process all observation nights
    data_dirs = glob.glob('data/2024-*')
    
    for data_dir in data_dirs:
        date = Path(data_dir).name
        print(f"Processing {date}...")
        
        runner = PipelineRunner(
            input_dir=Path(data_dir),
            output_dir=Path(f'reduced/{date}'),
            mode='batch'
        )
        
        try:
            results = runner.run()
            print(f"  ✓ Processed {len(results)} frames")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue


Integration with Astropy
-------------------------

.. code-block:: python

    from pykosmospp.extraction.extract import extract_optimal
    from specutils import Spectrum1D
    from astropy import units as u
    
    # Extract spectrum
    spectrum_1d = extract_optimal(data, trace, aperture_width=10)
    
    # Spectrum1D is astropy Spectrum1D object
    # Can use specutils functions directly
    from specutils.manipulation import FluxConservingResampler
    
    resampler = FluxConservingResampler()
    new_grid = np.arange(4000, 7000, 1) * u.AA
    resampled = resampler(spectrum_1d, new_grid)


Error Handling
==============

.. code-block:: python

    from pykosmospp.pipeline import PipelineRunner
    from pykosmospp.io.logging import CriticalPipelineError, QualityWarning
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        runner = PipelineRunner(
            input_dir=Path('data'),
            output_dir=Path('reduced')
        )
        results = runner.run()
        
    except CriticalPipelineError as e:
        # Unrecoverable errors (missing calibrations, bad FITS files)
        print(f"Pipeline failed: {e}")
        exit(1)
        
    except QualityWarning as w:
        # Quality issues (low SNR, poor wavelength fit) - proceeds with flag
        print(f"Quality warning: {w}")
        # Continue - outputs flagged
        
    except Exception as e:
        # Unexpected errors
        print(f"Unexpected error: {e}")
        raise


Configuration via Python
=========================

.. code-block:: python

    from pykosmospp.io.config import PipelineConfig
    
    # Create custom config
    config_dict = {
        'detector': {
            'gain': 1.4,
            'readnoise': 3.7,
            'saturate': 58982
        },
        'wavelength': {
            'max_order': 7,
            'initial_dispersion': 1.0,
            'rms_threshold': 0.1
        },
        'trace_detection': {
            'expected_fwhm': 4.0,
            'min_snr': 3.0,
            'min_separation': 20
        },
        'extraction': {
            'method': 'optimal',
            'aperture_width': 10,
            'sky_buffer': 30
        },
        'quality': {
            'min_snr': 5.0,
            'max_wavelength_rms': 0.2
        }
    }
    
    config = PipelineConfig(config_dict)
    
    # Validate config
    config.validate()
    
    # Access parameters
    gain = config['detector']['gain']
    
    # Use with pipeline
    runner = PipelineRunner(
        input_dir=Path('data'),
        output_dir=Path('reduced'),
        config=config
    )


See Also
========

* :ref:`cli_reference` - Command-line interface
* :ref:`configuration` - Configuration parameters
* :ref:`output_products` - Output file formats
* Full API documentation: :ref:`api_modules`
