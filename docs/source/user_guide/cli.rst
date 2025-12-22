.. _cli_reference:

==================
CLI Reference
==================

The pyKOSMOS++ command-line interface provides a complete spectroscopic reduction workflow through the ``kosmos-reduce`` command and its subcommands.

Main Command
============

kosmos-reduce
-------------

Process KOSMOS longslit spectroscopy observations from raw FITS files to wavelength-calibrated 1D spectra.

**Synopsis**::

    kosmos-reduce [OPTIONS] --input-dir INPUT --output-dir OUTPUT

**Required Arguments:**

``--input-dir PATH``
    Input directory containing FITS files organized in subdirectories:
    
    * ``biases/``: Bias frames (≥3 required)
    * ``flats/``: Flat field frames (≥3 required)
    * ``arcs/``: Arc lamp frames (≥1 required)
    * ``science/``: Science target frames

``--output-dir PATH``
    Output directory for reduced data products. Will be created if it doesn't exist.

**Optional Arguments:**

``--config PATH``
    Path to custom YAML configuration file. If not provided, uses default KOSMOS parameters.

``--mode {batch,interactive}``
    Processing mode:
    
    * ``batch``: Automatic processing of all detected traces (default)
    * ``interactive``: Visual trace selection with matplotlib GUI

``--max-traces N``
    Maximum number of traces to extract per science frame (default: 5)

``--verbose``
    Enable verbose logging output

``--quiet``
    Suppress non-essential output

``--log-file PATH``
    Write detailed logs to specified file

``--validate-only``
    Validate calibrations and inputs without processing science frames

``--overwrite``
    Overwrite existing output files without prompting

**Examples:**

Basic reduction with default settings::

    kosmos-reduce --input-dir data/2024-01-15 --output-dir reduced/

Custom configuration with verbose output::

    kosmos-reduce --input-dir data/faint_galaxy \\
                  --output-dir reduced/ \\
                  --config custom_config.yaml \\
                  --verbose

Interactive trace selection::

    kosmos-reduce --input-dir data/multi_slit \\
                  --output-dir reduced/ \\
                  --mode interactive

Validation only (check calibrations)::

    kosmos-reduce --input-dir data/ --output-dir tmp/ --validate-only


Subcommands
===========

calibrate
---------

Generate master bias and flat calibration frames only, without processing science data.

**Synopsis**::

    kosmos-reduce calibrate [OPTIONS] --input-dir INPUT --output-dir OUTPUT

**Arguments:**

``--input-dir PATH``
    Directory containing ``biases/`` and ``flats/`` subdirectories

``--output-dir PATH``
    Directory for calibration outputs (``calibrations/`` subdirectory created)

``--method {median,mean}``
    Combination method (default: median)

**Example**::

    kosmos-reduce calibrate --input-dir data/ --output-dir calibrations/


wavelength
----------

Perform wavelength calibration on arc lamp frames independently.

**Synopsis**::

    kosmos-reduce wavelength [OPTIONS] --input-arc ARC --lamp-type LAMP

**Required Arguments:**

``--input-arc PATH``
    Path to arc lamp FITS file

``--lamp-type {HeNeAr,Ar,Kr,Ne}``
    Arc lamp type for linelist selection

**Optional Arguments:**

``--output PATH``
    Output FITS table for wavelength solution (default: wavelength_solution.fits)

``--poly-order N``
    Polynomial order (3-7, default: auto-select with BIC)

``--rms-threshold FLOAT``
    Maximum acceptable RMS residual in Angstroms (default: 0.1)

``--min-lines N``
    Minimum number of identified lines required (default: 20)

``--plot``
    Generate diagnostic plot (wavelength_fit.pdf)

**Example**::

    kosmos-reduce wavelength --input-arc arc_001.fits \\
                             --lamp-type HeNeAr \\
                             --poly-order 5 \\
                             --plot


combine
-------

Combine multiple reduced 1D spectra onto a common wavelength grid.

**Synopsis**::

    kosmos-reduce combine [OPTIONS] --input-spectra PATTERN --output OUTPUT

**Required Arguments:**

``--input-spectra PATTERN``
    Glob pattern for input spectrum FITS files (e.g., ``"spectra_1d/*.fits"``)

``--output PATH``
    Output combined spectrum FITS file

**Optional Arguments:**

``--method {median,mean,weighted}``
    Combination method:
    
    * ``median``: Robust to outliers
    * ``mean``: Simple average
    * ``weighted``: Weighted by inverse variance

``--wavelength-grid START STOP STEP``
    Common wavelength grid in Angstroms (default: auto-determine from inputs)

``--resolution FLOAT``
    Target spectral resolution in Angstroms (default: preserve native)

**Example**::

    kosmos-reduce combine --input-spectra "reduced/spectra_1d/*.fits" \\
                         --output combined_spectrum.fits \\
                         --method weighted


Exit Codes
==========

The pipeline uses standard exit codes to indicate processing status:

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Code
     - Meaning
   * - 0
     - Success - all processing completed without errors
   * - 1
     - Missing or insufficient calibration frames
   * - 2
     - Invalid FITS files or configuration
   * - 3
     - Wavelength solution failed to converge
   * - 4
     - No traces detected in science frames
   * - 5
     - User canceled interactive mode

**Example usage in scripts**::

    #!/bin/bash
    kosmos-reduce --input-dir data/ --output-dir reduced/
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Reduction successful!"
    elif [ $EXIT_CODE -eq 1 ]; then
        echo "Error: Missing calibration frames"
        exit 1
    else
        echo "Pipeline error (code $EXIT_CODE)"
        exit 1
    fi


Output Products
===============

The pipeline creates a standardized output directory structure:

.. code-block:: text

    output_dir/
    ├── calibrations/
    │   ├── master_bias.fits
    │   ├── master_flat.fits
    │   └── bad_pixel_mask.fits
    ├── reduced_2d/
    │   ├── science_001_2d.fits
    │   └── science_002_2d.fits
    ├── spectra_1d/
    │   ├── science_001_trace1.fits
    │   ├── science_001_trace2.fits
    │   └── science_002_trace1.fits
    ├── wavelength_solutions/
    │   └── wavelength_solution_arc001.fits
    ├── quality_reports/
    │   ├── science_001_quality.yaml
    │   └── summary_report.txt
    ├── diagnostic_plots/
    │   ├── wavelength_solution.png
    │   ├── science_001_2d.png
    │   └── science_001_trace1_profile.png
    └── logs/
        └── reduction_log.txt

See :ref:`output_products` for detailed format specifications.


Configuration Files
===================

Custom configuration files use YAML format with nested sections:

.. code-block:: yaml

    # Detector parameters
    detector:
      gain: 1.4           # e-/ADU
      readnoise: 3.7      # e-
      saturate: 58982     # ADU
    
    # Wavelength calibration
    wavelength:
      max_order: 7
      initial_dispersion: 1.0
      rms_threshold: 0.1
    
    # Trace detection
    trace_detection:
      expected_fwhm: 4.0
      min_snr: 3.0
      min_separation: 20
    
    # Extraction
    extraction:
      method: optimal     # or 'boxcar'
      aperture_width: 10
      sky_buffer: 30
    
    # Quality thresholds
    quality:
      min_snr: 5.0
      max_wavelength_rms: 0.2

See :ref:`configuration` for complete parameter documentation.


Performance Tips
================

**Speed up calibration creation:**

* Use ``--method mean`` instead of median for faster (but less robust) combination
* Process only necessary calibration frames::

    # Only combine first 5 bias/flat frames
    ls biases/*.fits | head -5 | xargs -I {} cp {} subset_biases/

**Parallel processing:**

* Process multiple nights independently in separate terminals
* Use ``--max-traces`` to limit extraction overhead for multi-slit observations

**Memory optimization:**

* Process large datasets in batches rather than all at once
* Use ``--quiet`` mode to reduce I/O overhead from logging

**Disk space:**

* Diagnostic plots can be large; use ``--no-plots`` (if implemented) for production runs
* Remove intermediate ``reduced_2d/`` files after successful 1D extraction


Troubleshooting
===============

**"No FITS files found"**

* Check directory structure matches expected layout
* Ensure files have ``.fits`` extension (not ``.fit`` or ``.FITS``)
* Verify FITS headers contain ``IMAGETYP`` keyword

**"Insufficient calibration frames"**

* Need ≥3 bias, ≥3 flat, ≥1 arc minimum
* Check subdirectory names match exactly: ``biases/``, ``flats/``, ``arcs/``

**"Wavelength solution failed"**

* Arc lamp exposure may be too short (need SNR >10 for line detection)
* Try ``--lamp-type`` matching actual lamp used
* Use ``wavelength`` subcommand with ``--plot`` to diagnose

**"No traces detected"**

* Lower ``min_snr`` in config for faint sources
* Use ``--mode interactive`` to visually verify 2D spectrum
* Check flat field quality (illumination pattern)

For more issues, see :ref:`troubleshooting`.


See Also
========

* :ref:`python_api` - Programmatic access
* :ref:`configuration` - Configuration file reference
* :ref:`output_products` - Output format specifications
* :ref:`quickstart` - Quick start guide
