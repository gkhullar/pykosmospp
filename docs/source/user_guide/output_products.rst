.. _output_products:

=================
Output Products
=================

pyKOSMOS++ generates a standardized set of output products organized in a structured directory hierarchy. This page documents the format and content of all output files.

Directory Structure
===================

The pipeline creates the following output directory structure::

    output_dir/
    ├── calibrations/
    │   ├── master_bias.fits
    │   ├── master_flat.fits
    │   └── bad_pixel_mask.fits
    ├── reduced_2d/
    │   ├── science_001_2d.fits
    │   ├── science_002_2d.fits
    │   └── ...
    ├── spectra_1d/
    │   ├── science_001_trace1.fits
    │   ├── science_001_trace2.fits
    │   ├── science_002_trace1.fits
    │   └── ...
    ├── wavelength_solutions/
    │   ├── wavelength_solution_arc001.fits
    │   └── ...
    ├── quality_reports/
    │   ├── science_001_quality.yaml
    │   ├── science_002_quality.yaml
    │   └── summary_report.txt
    ├── diagnostic_plots/
    │   ├── wavelength_solution.png
    │   ├── science_001_2d.png
    │   ├── science_001_trace1_profile.png
    │   └── ...
    └── logs/
        └── reduction_log.txt


Calibration Products
====================

master_bias.fits
----------------

Combined master bias frame.

**Format:** FITS image (Primary HDU)

**Header Keywords:**

.. code-block:: text

    NAXIS   = 2
    NAXIS1  = 2048          / Spectral axis
    NAXIS2  = 515           / Spatial axis
    BUNIT   = 'ADU'
    
    # Calibration metadata
    NCOMBINE= 5             / Number of bias frames combined
    COMBMETH= 'median'      / Combination method
    BIASLVL = 389.2         / Median bias level (ADU)
    BIASSTD = 3.5           / Bias standard deviation (ADU)
    
    # Provenance
    PIPELINE= 'pyKOSMOS++ v0.1.0'
    DATE    = '2025-01-15T10:23:45'
    PROCSTEP= 'BIAS_COMBINE'

**Data:** 2D float32 array, bias level in ADU

**Usage:**

.. code-block:: python

    from astropy.io import fits
    with fits.open('master_bias.fits') as hdul:
        bias_data = hdul[0].data
        bias_level = hdul[0].header['BIASLVL']


master_flat.fits
----------------

Normalized master flat field.

**Format:** FITS image (Primary HDU)

**Header Keywords:**

.. code-block:: text

    NAXIS   = 2
    BUNIT   = 'normalized'
    
    NCOMBINE= 3             / Number of flat frames combined
    COMBMETH= 'median'
    NORMLVL = 1.0           / Normalization level
    BADPIXFR= 0.0023        / Bad pixel fraction
    
    PIPELINE= 'pyKOSMOS++ v0.1.0'
    PROCSTEP= 'FLAT_COMBINE'

**Data:** 2D float32 array, normalized response (median = 1.0)

**Bad Pixel Mask:** Pixels with response <0.5 or >1.5 are flagged


bad_pixel_mask.fits
-------------------

Boolean mask of bad pixels.

**Format:** FITS image (Primary HDU)

**Data:** 2D boolean array (0=good, 1=bad)

**Bad pixel criteria:**

* Flat response <0.5 or >1.5 (vignetting or hot pixels)
* Saturated pixels in calibration frames
* Cosmic ray hits in master flat


Reduced 2D Spectra
==================

science_NNN_2d.fits
-------------------

Calibrated 2D spectrum with trace information.

**Format:** Multi-extension FITS

**Extension 0 (Primary):** Calibrated 2D spectrum

.. code-block:: text

    NAXIS   = 2
    BUNIT   = 'ADU'
    
    # Original frame metadata
    OBJECT  = 'NGC1234'
    EXPTIME = 1200.0        / seconds
    AIRMASS = 1.15
    DATEOBS = '2024-01-15T05:30:00'
    
    # Calibration applied
    BIASFILE= 'master_bias.fits'
    FLATFILE= 'master_flat.fits'
    BIASCORR= T
    FLATCORR= T
    COSMCORR= T             / Cosmic ray cleaning applied
    
    PIPELINE= 'pyKOSMOS++ v0.1.0'
    PROCSTEP= '2D_CALIBRATION'

**Extension 1 (VARIANCE):** Variance array

* Propagated from read noise and Poisson statistics
* Same shape as primary data

**Extension 2 (MASK):** Pixel mask

* 0 = good pixel
* 1 = bad pixel from flat
* 2 = cosmic ray detected
* 4 = saturation
* 8 = other (can combine bits)

**Extension 3 (TRACES):** Binary table of detected traces

Columns:

* ``TRACE_ID`` (int): Trace identifier
* ``SPATIAL_POS`` (float array): Spatial center positions [pixels]
* ``SPECTRAL_PIX`` (float array): Spectral pixel coordinates
* ``SNR_EST`` (float): Estimated signal-to-noise ratio

**Usage:**

.. code-block:: python

    with fits.open('science_001_2d.fits') as hdul:
        data = hdul[0].data
        variance = hdul[1].data
        mask = hdul[2].data
        traces = hdul[3].data


Extracted 1D Spectra
====================

science_NNN_traceM.fits
-----------------------

Extracted and wavelength-calibrated 1D spectrum.

**Format:** Multi-extension FITS (compatible with Astropy Spectrum1D)

**Extension 0 (Primary):** Header only

.. code-block:: text

    # Source information
    OBJECT  = 'NGC1234'
    TRACEID = 1
    EXPTIME = 1200.0
    AIRMASS = 1.15
    
    # Extraction metadata
    EXTMETH = 'optimal'     / Extraction method
    APWIDTH = 10            / Aperture width (pixels)
    SKYBUFFE= 30            / Sky buffer distance (pixels)
    
    # Quality metrics
    MED_SNR = 12.5          / Median signal-to-noise ratio
    WLRMS   = 0.08          / Wavelength RMS residual (Angstroms)
    GRADE   = 'Good'        / Quality grade
    
    PIPELINE= 'pyKOSMOS++ v0.1.0'
    PROCSTEP= '1D_EXTRACTION'

**Extension 1 (FLUX):** 1D flux array

* **NAXIS:** 1
* **NAXIS1:** Number of wavelength points
* **BUNIT:** 'ADU' or 'erg/s/cm^2/Angstrom' (if flux calibrated)

**Extension 2 (WAVELENGTH):** Wavelength array

* **BUNIT:** 'Angstrom'
* **WAVEMIN, WAVEMAX:** Wavelength range

**Extension 3 (UNCERTAINTY):** 1D uncertainty array

* Propagated from variance in 2D spectrum
* Same units as flux

**Extension 4 (MASK):** 1D mask array

* 0 = good pixel
* Nonzero = various flags (cosmic ray, saturation, etc.)

**Usage:**

.. code-block:: python

    from specutils import Spectrum1D
    from astropy import units as u
    
    # Load as Spectrum1D
    spectrum = Spectrum1D.read('science_001_trace1.fits')
    
    # Access data
    flux = spectrum.flux            # with units
    wavelength = spectrum.spectral_axis
    uncertainty = spectrum.uncertainty
    
    # Or use FITS directly
    with fits.open('science_001_trace1.fits') as hdul:
        flux = hdul['FLUX'].data
        wavelength = hdul['WAVELENGTH'].data
        uncertainty = hdul['UNCERTAINTY'].data


Wavelength Solutions
====================

wavelength_solution_arcNNN.fits
-------------------------------

Wavelength calibration solution for arc frame.

**Format:** FITS binary table (Primary HDU)

**Columns:**

* ``PIXEL`` (float): Pixel position
* ``WAVELENGTH`` (float): Wavelength in Angstroms
* ``RESIDUAL`` (float): Fit residual in Angstroms
* ``INTENSITY`` (float): Line intensity
* ``USED`` (bool): Line included in fit (not clipped)

**Header Keywords:**

.. code-block:: text

    LAMPTYPE= 'HeNeAr'     / Arc lamp type
    POLYORD = 5            / Polynomial order
    NLINES  = 45           / Total lines identified
    NUSED   = 42           / Lines used in fit
    RMS     = 0.078        / RMS residual (Angstroms)
    WAVEMIN = 3650.0       / Minimum wavelength (Angstroms)
    WAVEMAX = 7200.0       / Maximum wavelength (Angstroms)
    
    # Polynomial coefficients
    COEFF0  = 3645.123
    COEFF1  = 1.0234
    COEFF2  = -0.000123
    ...

**Usage:**

.. code-block:: python

    with fits.open('wavelength_solution_arc001.fits') as hdul:
        table = hdul[1].data
        header = hdul[0].header
        
        # Extract coefficients
        order = header['POLYORD']
        coeffs = [header[f'COEFF{i}'] for i in range(order+1)]
        
        # Reconstruct solution
        from numpy.polynomial.chebyshev import chebval
        wavelength = chebval(pixel, coeffs)


Quality Reports
===============

science_NNN_quality.yaml
------------------------

Detailed quality metrics for each science frame.

**Format:** YAML

**Structure:**

.. code-block:: yaml

    # Source metadata
    source_file: science_001.fits
    object_name: NGC1234
    exposure_time: 1200.0
    observation_date: '2024-01-15T05:30:00'
    
    # Calibration quality
    calibration:
      bias_level: 389.2
      bias_stdev: 3.5
      flat_bad_pixel_fraction: 0.0023
      cosmic_ray_fraction: 0.0087
    
    # Wavelength solution
    wavelength:
      polynomial_order: 5
      rms_residual: 0.078
      n_lines_identified: 45
      n_lines_used: 42
      wavelength_range: [3650.0, 7200.0]
    
    # Extraction metrics (per trace)
    traces:
      - trace_id: 1
        median_snr: 12.5
        peak_snr: 45.2
        spatial_center: 257.5
        profile_consistency: 0.92
        flux_conservation: 0.98
        
      - trace_id: 2
        median_snr: 8.3
        ...
    
    # Overall assessment
    overall_grade: Good
    quality_flags: []
    warnings:
      - 'Trace 2 has lower SNR (8.3 < 10.0 recommended)'
    
    # Processing metadata
    pipeline_version: '0.1.0'
    processing_date: '2025-01-15T10:45:00'
    processing_time_seconds: 125.3

**Usage:**

.. code-block:: python

    import yaml
    with open('science_001_quality.yaml') as f:
        quality = yaml.safe_load(f)
    
    print(f"Overall grade: {quality['overall_grade']}")
    for trace in quality['traces']:
        print(f"Trace {trace['trace_id']}: SNR = {trace['median_snr']:.1f}")


summary_report.txt
------------------

Human-readable summary of all processed frames.

**Format:** Plain text

**Example:**

.. code-block:: text

    pyKOSMOS++ Pipeline Summary Report
    ==================================
    
    Processing Date: 2025-01-15 10:45:00
    Pipeline Version: 0.1.0
    
    Input Directory: data/2024-01-15
    Output Directory: reduced/2024-01-15
    
    Calibration Summary:
    -------------------
    Master Bias:  5 frames combined, level=389.2±3.5 ADU
    Master Flat:  3 frames combined, bad pixels=0.23%
    Arc Lamp:     HeNeAr, 45 lines identified, RMS=0.078 Å
    
    Science Frames Processed: 10
    ----------------------------
    
    Frame                    Traces  Grade      Median SNR  Wavelength RMS
    science_001.fits         2       Good       12.5        0.078 Å
    science_002.fits         2       Excellent  23.1        0.065 Å
    science_003.fits         1       Fair       7.2         0.092 Å
    ...
    
    Overall Statistics:
    ------------------
    Total spectra extracted: 15
    Mean SNR: 14.8
    Grade distribution:
      Excellent: 3 (20%)
      Good: 10 (67%)
      Fair: 2 (13%)
      Poor: 0 (0%)
    
    Processing Time: 2.3 minutes (13.8 seconds/frame)


Diagnostic Plots
================

wavelength_solution.png
-----------------------

Wavelength calibration diagnostic.

**Contents:**

* Top panel: Wavelength vs pixel with Chebyshev fit
* Bottom panel: Fit residuals with RMS threshold lines

**Format:** PNG, 14x8 inches, 300 DPI


science_NNN_2d.png
------------------

Calibrated 2D spectrum with detected traces.

**Contents:**

* 2D spectrum with log-scale colormap
* Overlaid trace positions (red lines)
* Wavelength and spatial axes labeled

**Format:** PNG, 14x6 inches, 300 DPI


science_NNN_traceM_profile.png
------------------------------

Spatial profile fit diagnostic.

**Contents:**

* Top panel: Data vs fitted profile (Gaussian/Moffat)
* Bottom panel: Fit residuals

**Format:** PNG, 10x6 inches, 300 DPI


science_NNN_traceM_1d.png
-------------------------

Extracted 1D spectrum.

**Contents:**

* Flux vs wavelength
* Major spectral features annotated (if known)

**Format:** PNG, 14x4 inches, 300 DPI


Logs
====

reduction_log.txt
-----------------

Detailed processing log.

**Format:** Plain text with timestamps

**Example:**

.. code-block:: text

    2025-01-15 10:30:00 INFO: pyKOSMOS++ v0.1.0 starting
    2025-01-15 10:30:00 INFO: Input directory: data/2024-01-15
    2025-01-15 10:30:01 INFO: Discovered 5 bias, 3 flat, 1 arc, 10 science frames
    2025-01-15 10:30:02 INFO: Creating master bias from 5 frames
    2025-01-15 10:30:05 INFO: Master bias: level=389.2±3.5 ADU
    2025-01-15 10:30:05 INFO: Creating master flat from 3 frames
    2025-01-15 10:30:12 INFO: Master flat: bad pixels=0.23%
    2025-01-15 10:30:13 INFO: Fitting wavelength solution
    2025-01-15 10:30:15 INFO: Wavelength solution: order=5, RMS=0.078 Å, 45 lines
    2025-01-15 10:30:16 INFO: Processing science_001.fits
    2025-01-15 10:30:18 INFO:   Detected 2 traces
    2025-01-15 10:30:25 INFO:   Trace 1: SNR=12.5, extracted
    2025-01-15 10:30:30 INFO:   Trace 2: SNR=8.3, extracted
    2025-01-15 10:30:30 INFO:   Overall grade: Good
    ...
    2025-01-15 10:32:45 INFO: Pipeline completed successfully
    2025-01-15 10:32:45 INFO: Total time: 2.75 minutes


Reading Output Products
=======================

Python Example
--------------

Complete workflow to read and analyze outputs:

.. code-block:: python

    from pathlib import Path
    from astropy.io import fits
    from specutils import Spectrum1D
    import yaml
    import matplotlib.pyplot as plt
    
    output_dir = Path('reduced/2024-01-15')
    
    # Read quality report
    with open(output_dir / 'quality_reports/science_001_quality.yaml') as f:
        quality = yaml.safe_load(f)
    
    print(f"Overall grade: {quality['overall_grade']}")
    print(f"Median SNR: {quality['traces'][0]['median_snr']:.1f}")
    
    # Load 1D spectrum
    spectrum_file = output_dir / 'spectra_1d/science_001_trace1.fits'
    spectrum = Spectrum1D.read(spectrum_file)
    
    # Plot spectrum
    plt.figure(figsize=(12, 4))
    plt.plot(spectrum.spectral_axis, spectrum.flux)
    plt.xlabel(f'Wavelength ({spectrum.spectral_axis.unit})')
    plt.ylabel(f'Flux ({spectrum.flux.unit})')
    plt.title(f"{quality['object_name']} - Grade: {quality['overall_grade']}")
    plt.show()
    
    # Load 2D spectrum
    with fits.open(output_dir / 'reduced_2d/science_001_2d.fits') as hdul:
        data_2d = hdul[0].data
        variance = hdul[1].data
        mask = hdul[2].data
        traces = hdul[3].data
        
        print(f"Detected {len(traces)} traces")
        for trace in traces:
            print(f"  Trace {trace['TRACE_ID']}: SNR={trace['SNR_EST']:.1f}")


See Also
========

* :ref:`cli_reference` - Generating output products
* :ref:`python_api` - Programmatic access to outputs
* :ref:`quickstart` - Example workflows
