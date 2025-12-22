.. _configuration:

=======================
Configuration Reference
=======================

pyKOSMOS++ uses YAML configuration files to customize pipeline behavior. This page documents all available configuration parameters.

Configuration File Format
=========================

Configuration files use YAML syntax with nested sections::

    # config/custom_config.yaml
    detector:
      gain: 1.4
      readnoise: 3.7
      
    wavelength:
      max_order: 7
      
    # Comments supported

Loading Configuration
=====================

From command line::

    kosmos-reduce --input-dir data/ --output-dir reduced/ --config custom_config.yaml

From Python::

    from pykosmospp.io.config import PipelineConfig
    config = PipelineConfig.from_yaml('custom_config.yaml')

Default Configuration
=====================

If no custom config is provided, the pipeline uses ``config/kosmos_defaults.yaml`` with parameters optimized for the APO-KOSMOS spectrograph.


Parameter Reference
===================

Detector Section
----------------

Physical detector properties and data characteristics.

``detector.gain`` (float)
    CCD gain in electrons per ADU.
    
    * **Default:** ``1.4``
    * **Units:** e⁻/ADU
    * **Range:** 0.5 - 5.0
    * **Notes:** Check instrument manual for your detector

``detector.readnoise`` (float)
    CCD read noise in electrons.
    
    * **Default:** ``3.7``
    * **Units:** e⁻
    * **Range:** 1.0 - 10.0
    * **Notes:** From detector spec sheet

``detector.saturate`` (int)
    Saturation level in ADU.
    
    * **Default:** ``58982``
    * **Units:** ADU
    * **Range:** 10000 - 65535
    * **Notes:** Pixels above this value are flagged

``detector.dark_current`` (float, optional)
    Dark current in electrons per pixel per second.
    
    * **Default:** ``0.0`` (negligible for KOSMOS)
    * **Units:** e⁻/pixel/s


Calibration Section
-------------------

Parameters for master calibration creation.

``calibration.method`` (string)
    Combination method for bias and flat frames.
    
    * **Default:** ``median``
    * **Options:** ``median``, ``mean``
    * **Notes:** Median is robust to outliers, mean is faster

``calibration.sigma_clip`` (float)
    Sigma-clipping threshold for combining frames.
    
    * **Default:** ``3.0``
    * **Units:** sigma
    * **Range:** 2.0 - 5.0

``calibration.min_frames`` (dict)
    Minimum number of frames required for each type.
    
    .. code-block:: yaml
    
        calibration:
          min_frames:
            bias: 3
            flat: 3
            arc: 1
    
    * **Default:** ``{bias: 3, flat: 3, arc: 1}``

``calibration.flat_norm_region`` (list, optional)
    Spatial region [start, end] for flat normalization.
    
    * **Default:** ``null`` (uses central 50%)
    * **Units:** pixels
    * **Example:** ``[200, 1800]``


Wavelength Section
------------------

Wavelength calibration parameters.

``wavelength.max_order`` (int)
    Maximum polynomial order for wavelength fit.
    
    * **Default:** ``7``
    * **Range:** 3 - 10
    * **Notes:** Higher orders risk overfitting

``wavelength.initial_dispersion`` (float)
    Approximate dispersion in Angstroms per pixel.
    
    * **Default:** ``1.0``
    * **Units:** Å/pixel
    * **Range:** 0.1 - 10.0
    * **Notes:** Used for initial line matching

``wavelength.wavelength_range`` (list)
    Wavelength range [min, max] to search for arc lines.
    
    * **Default:** ``[3500, 7500]`` (KOSMOS range)
    * **Units:** Angstroms
    * **Example:** ``[4000, 7000]`` for limited range

``wavelength.rms_threshold`` (float)
    Maximum acceptable RMS residual for wavelength solution.
    
    * **Default:** ``0.1``
    * **Units:** Angstroms
    * **Range:** 0.01 - 0.5
    * **Notes:** Solutions with RMS > threshold are flagged

``wavelength.min_lines`` (int)
    Minimum identified arc lines required for fit.
    
    * **Default:** ``20``
    * **Range:** 10 - 100
    * **Notes:** More lines improve robustness

``wavelength.match_tolerance`` (float)
    Tolerance for matching detected lines to catalog.
    
    * **Default:** ``2.0``
    * **Units:** Angstroms
    * **Range:** 0.5 - 5.0

``wavelength.sigma_clip`` (float)
    Sigma-clipping threshold for iterative wavelength fit.
    
    * **Default:** ``3.0``
    * **Units:** sigma
    * **Range:** 2.0 - 5.0

``wavelength.max_iterations`` (int)
    Maximum sigma-clipping iterations.
    
    * **Default:** ``5``
    * **Range:** 1 - 10


Trace Detection Section
------------------------

Parameters for spectral trace identification.

``trace_detection.expected_fwhm`` (float)
    Expected trace FWHM in spatial direction.
    
    * **Default:** ``4.0``
    * **Units:** pixels
    * **Range:** 2.0 - 20.0
    * **Notes:** Depends on seeing and slit width

``trace_detection.min_snr`` (float)
    Minimum signal-to-noise ratio for trace detection.
    
    * **Default:** ``3.0``
    * **Range:** 1.0 - 10.0
    * **Notes:** Lower values detect fainter traces but increase false positives

``trace_detection.min_separation`` (int)
    Minimum pixel separation between distinct traces.
    
    * **Default:** ``20``
    * **Units:** pixels
    * **Range:** 5 - 100
    * **Notes:** Prevents duplicate detections

``trace_detection.detection_threshold`` (float)
    Threshold for arc line detection in cross-correlation.
    
    * **Default:** ``5.0``
    * **Units:** sigma above background
    * **Range:** 3.0 - 10.0


Extraction Section
------------------

Spectral extraction parameters.

``extraction.method`` (string)
    Extraction method.
    
    * **Default:** ``optimal``
    * **Options:** ``optimal``, ``boxcar``
    * **Notes:** Optimal uses Horne 1986 variance weighting

``extraction.aperture_width`` (int)
    Extraction aperture width in spatial direction.
    
    * **Default:** ``10``
    * **Units:** pixels
    * **Range:** 5 - 50
    * **Notes:** Should encompass trace profile

``extraction.sky_buffer`` (int)
    Distance from trace center to start sky estimation.
    
    * **Default:** ``30``
    * **Units:** pixels
    * **Range:** 10 - 100
    * **Notes:** Larger values avoid contamination from trace wings

``extraction.sky_width`` (int, optional)
    Width of sky region on each side of trace.
    
    * **Default:** ``50`` pixels
    * **Units:** pixels
    * **Range:** 20 - 200

``extraction.profile_type`` (string)
    Spatial profile model.
    
    * **Default:** ``Gaussian``
    * **Options:** ``Gaussian``, ``Moffat``, ``empirical``


Cosmic Ray Section
------------------

Cosmic ray detection parameters (L.A.Cosmic algorithm).

``cosmic_ray.enabled`` (bool)
    Enable cosmic ray detection and cleaning.
    
    * **Default:** ``true``

``cosmic_ray.sigma_clip`` (float)
    Detection threshold in sigma.
    
    * **Default:** ``5.0``
    * **Range:** 3.0 - 10.0

``cosmic_ray.contrast`` (float)
    Contrast threshold for neighbor comparison.
    
    * **Default:** ``3.0``
    * **Range:** 1.0 - 5.0

``cosmic_ray.max_iterations`` (int)
    Maximum cleaning iterations.
    
    * **Default:** ``5``
    * **Range:** 1 - 10


Quality Section
---------------

Quality assessment thresholds.

``quality.min_snr`` (float)
    Minimum acceptable median SNR for "Good" grade.
    
    * **Default:** ``5.0``
    * **Range:** 1.0 - 20.0

``quality.excellent_snr`` (float)
    Minimum SNR for "Excellent" grade.
    
    * **Default:** ``20.0``

``quality.max_wavelength_rms`` (float)
    Maximum wavelength RMS for "Good" grade.
    
    * **Default:** ``0.2``
    * **Units:** Angstroms

``quality.excellent_wavelength_rms`` (float)
    Maximum wavelength RMS for "Excellent" grade.
    
    * **Default:** ``0.1``
    * **Units:** Angstroms

``quality.max_bad_pixel_fraction`` (float)
    Maximum fraction of bad pixels in flat field.
    
    * **Default:** ``0.05`` (5%)
    * **Range:** 0.0 - 0.2

``quality.max_cosmic_ray_fraction`` (float)
    Maximum fraction of pixels flagged as cosmic rays.
    
    * **Default:** ``0.01`` (1%)
    * **Range:** 0.0 - 0.1


Binning Section (Optional)
---------------------------

Spatial and spectral binning options.

``binning.spatial.enabled`` (bool)
    Enable spatial binning before extraction.
    
    * **Default:** ``false``
    * **Notes:** Useful for faint sources to boost SNR

``binning.spatial.factor`` (int)
    Spatial binning factor.
    
    * **Default:** ``2``
    * **Range:** 2 - 8

``binning.spectral.enabled`` (bool)
    Enable spectral binning after extraction.
    
    * **Default:** ``false``

``binning.spectral.width_angstrom`` (float)
    Spectral bin width.
    
    * **Default:** ``2.0``
    * **Units:** Angstroms


AB Subtraction Section (Optional)
----------------------------------

Parameters for nod-and-shuffle observations.

``ab_subtraction.enabled`` (bool)
    Enable AB pair subtraction for sky removal.
    
    * **Default:** ``false``

``ab_subtraction.nod_tolerance`` (float)
    Maximum time difference to match A/B pairs.
    
    * **Default:** ``600.0`` (10 minutes)
    * **Units:** seconds

``ab_subtraction.max_iterations`` (int)
    Maximum iterations for iterative sky estimation.
    
    * **Default:** ``5``


Example Configurations
======================

Faint Galaxy Configuration
---------------------------

Optimized for low surface brightness extended sources::

    # faint_galaxy_config.yaml
    trace_detection:
      expected_fwhm: 5.0      # Slightly extended seeing
      min_snr: 2.0            # Lower threshold
      
    extraction:
      aperture_width: 15      # Wider to capture all flux
      sky_buffer: 40          # Extra buffer from wings
      
    binning:
      spatial:
        enabled: true
        factor: 2             # 2x2 spatial binning
      
    quality:
      min_snr: 3.0            # Accept lower SNR


High-Resolution Wavelength Config
----------------------------------

For precision wavelength calibration::

    # precision_wavelength_config.yaml
    wavelength:
      max_order: 9            # Higher order fit
      rms_threshold: 0.05     # Stricter RMS requirement
      min_lines: 40           # More lines for robustness
      sigma_clip: 2.5         # Tighter outlier rejection
      
    quality:
      excellent_wavelength_rms: 0.05
      max_wavelength_rms: 0.1


Multi-Slit Configuration
-------------------------

For multi-object spectroscopy::

    # multi_slit_config.yaml
    trace_detection:
      min_separation: 50      # Objects well-separated
      min_snr: 3.0
      
    extraction:
      aperture_width: 8       # Narrower for slit width
      sky_buffer: 20          # Sky regions between slits


Configuration Validation
=========================

The pipeline validates configuration on load::

    from pykosmospp.io.config import PipelineConfig
    
    try:
        config = PipelineConfig.from_yaml('config.yaml')
        config.validate()  # Checks for required sections and valid ranges
    except ValueError as e:
        print(f"Invalid configuration: {e}")

**Validation checks:**

* All required sections present
* Numeric values in valid ranges
* String options match allowed values
* No conflicting parameters


See Also
========

* :ref:`cli_reference` - Using config with CLI
* :ref:`python_api` - Programmatic config access
* :ref:`quickstart` - Example workflows with custom configs
