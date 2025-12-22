.. _troubleshooting:

===============
Troubleshooting
===============

Common issues and solutions for pyKOSMOS++ pipeline problems.

Installation Issues
===================

ImportError: No module named 'pykosmospp'
------------------------------------------

**Problem:** Package not installed or not in Python path.

**Solution:**

.. code-block:: bash

    # Verify installation
    pip list | grep pykosm
    
    # If not found, install
    pip install pykosmospp
    
    # Or for development
    pip install -e .


ModuleNotFoundError: No module named 'astropy'
-----------------------------------------------

**Problem:** Missing dependencies.

**Solution:**

.. code-block:: bash

    # Install all dependencies
    pip install -e .
    
    # Or manually
    pip install astropy scipy numpy matplotlib specutils pyyaml


Version conflicts with existing packages
-----------------------------------------

**Problem:** Dependency version incompatibility.

**Solution:**

.. code-block:: bash

    # Create isolated environment
    conda create -n pykosmospp python=3.10
    conda activate pykosmospp
    pip install pykosmospp
    
    # Or with venv
    python -m venv venv_pykosmospp
    source venv_pykosmospp/bin/activate  # Linux/Mac
    pip install pykosmospp


Data Loading Issues
===================

"No FITS files found"
---------------------

**Problem:** Pipeline cannot locate input files.

**Diagnosis:**

.. code-block:: bash

    # Check directory structure
    ls -R data/
    
    # Should show:
    # data/
    #   biases/
    #     *.fits
    #   flats/
    #     *.fits
    #   arcs/
    #     *.fits
    #   science/
    #     *.fits

**Solutions:**

1. **Wrong directory structure:**

   .. code-block:: bash
   
       # Create correct structure
       mkdir -p data/{biases,flats,arcs,science}
       
       # Move files
       mv bias*.fits data/biases/
       mv flat*.fits data/flats/

2. **Wrong file extensions:**

   .. code-block:: bash
   
       # Rename .fit to .fits
       rename 's/\.fit$/.fits/' *.fit
       
       # Or lowercase .FITS
       rename 'y/A-Z/a-z/' *.FITS

3. **Files in wrong subdirectory:**
   
   Check FITS headers to verify frame type::
   
       from astropy.io import fits
       with fits.open('mystery_file.fits') as hdul:
           print(hdul[0].header['IMAGETYP'])


"IMAGETYP keyword not found"
-----------------------------

**Problem:** FITS headers missing required keywords.

**Solution:**

Add missing keywords manually:

.. code-block:: python

    from astropy.io import fits
    
    # Read file
    with fits.open('file.fits', mode='update') as hdul:
        header = hdul[0].header
        
        # Add IMAGETYP
        header['IMAGETYP'] = 'Bias'  # or 'Flat', 'Arc', 'Object'
        header['OBJECT'] = 'bias'
        header['EXPTIME'] = 0.0

Or use FITS editor::

    fitsheader -u IMAGETYP 'Bias' bias_001.fits


Calibration Errors
==================

"Insufficient calibration frames"
----------------------------------

**Problem:** Not enough bias/flat/arc frames.

**Requirements:**

* Minimum 3 bias frames
* Minimum 3 flat frames  
* Minimum 1 arc frame

**Solution:**

.. code-block:: bash

    # Check frame counts
    ls data/biases/*.fits | wc -l
    ls data/flats/*.fits | wc -l
    ls data/arcs/*.fits | wc -l
    
    # If insufficient, acquire more frames or adjust minimum in config:
    
    # config.yaml
    calibration:
      min_frames:
        bias: 2   # Reduce minimum (not recommended)
        flat: 2


"Bias level variation too high"
--------------------------------

**Problem:** Bias frames have inconsistent levels.

**Diagnosis:**

.. code-block:: python

    from pykosmospp.models import BiasFrame
    import numpy as np
    
    levels = []
    for f in bias_files:
        bias = BiasFrame.from_fits(f)
        levels.append(np.median(bias.data.data))
    
    print(f"Bias levels: {levels}")
    print(f"Std dev: {np.std(levels):.2f} ADU")

**Solutions:**

1. **Temperature variation:** Ensure CCD temperature stable during calibrations

2. **Different readout modes:** Check all bias frames use same binning/readout

3. **Outlier frames:** Remove problematic bias with different level::

       # Remove outlier
       rm data/biases/bias_003.fits  # If level differs by >20 ADU


"Flat field saturation"
-----------------------

**Problem:** Flat frames over-exposed.

**Diagnosis:**

.. code-block:: python

    from astropy.io import fits
    
    with fits.open('flat_001.fits') as hdul:
        data = hdul[0].data
        saturate = hdul[0].header['SATURATE']
        n_saturated = np.sum(data >= saturate * 0.9)
        
        print(f"Saturated pixels: {n_saturated} ({100*n_saturated/data.size:.2f}%)")

**Solution:**

* Re-observe flats with shorter exposure time
* Target median counts 10k-50k ADU (well below saturation)
* Or remove saturated frames from calibration set


Wavelength Calibration Issues
==============================

"Wavelength solution failed to converge"
-----------------------------------------

**Problem:** Cannot fit wavelength solution to arc lines.

**Common Causes:**

1. **Too few lines identified:** Need ≥20 lines

   .. code-block:: python
   
       # Check line detection
       from pykosmospp.wavelength.identify import detect_arc_lines
       import numpy as np
       
       arc_spectrum = np.median(arc_frame.data.data, axis=0)
       lines = detect_arc_lines(arc_spectrum, detection_threshold=5.0)
       print(f"Detected {len(lines)} lines")
       
       # If <20 lines:
       # - Lower detection_threshold to 3.0
       # - Check arc exposure time (may be too short)
       # - Verify correct lamp type selected

2. **Wrong arc lamp type:**

   .. code-block:: bash
   
       # Try different lamp types
       kosmos-reduce wavelength --input-arc arc.fits --lamp-type HeNeAr --plot
       kosmos-reduce wavelength --input-arc arc.fits --lamp-type Ar --plot
       kosmos-reduce wavelength --input-arc arc.fits --lamp-type Kr --plot

3. **Arc lamp under-exposed:**
   
   * Arc lines should have SNR >10
   * Increase exposure time for next observation
   * Can try spatial binning to boost SNR


"Wavelength RMS too high"
--------------------------

**Problem:** Wavelength fit residuals exceed threshold.

**Diagnosis:**

.. code-block:: bash

    # Generate diagnostic plot
    kosmos-reduce wavelength --input-arc arc.fits \\
                             --lamp-type HeNeAr \\
                             --plot
    
    # Check wavelength_fit.pdf for:
    # - Systematic residuals (wrong polynomial order)
    # - Outlier lines (misidentifications)
    # - Overall RMS value

**Solutions:**

1. **Adjust polynomial order:**

   .. code-block:: yaml
   
       # config.yaml
       wavelength:
         max_order: 5  # Try lower order if overfitting
         # or
         max_order: 9  # Try higher order if underfitting

2. **Tighten sigma-clipping:**

   .. code-block:: yaml
   
       wavelength:
         sigma_clip: 2.5  # More aggressive outlier rejection

3. **Use more arc lines:**
   
   * Increase arc exposure time for better S/N
   * Expand wavelength_range to detect more lines


Trace Detection Issues
======================

"No traces detected"
--------------------

**Problem:** Pipeline finds no spectral traces.

**Diagnosis:**

.. code-block:: python

    # Visualize 2D spectrum
    from astropy.io import fits
    import matplotlib.pyplot as plt
    
    with fits.open('reduced_2d/science_001_2d.fits') as hdul:
        data = hdul[0].data
    
    plt.imshow(data, cmap='viridis', origin='lower',
              vmin=np.percentile(data, 1),
              vmax=np.percentile(data, 99))
    plt.colorbar()
    plt.xlabel('Spectral Direction')
    plt.ylabel('Spatial Direction')
    plt.show()
    
    # Visual inspection:
    # - Is there a visible horizontal trace?
    # - Is the spectrum too faint?
    # - Is the slit illuminated?

**Solutions:**

1. **Lower detection threshold:**

   .. code-block:: yaml
   
       # config.yaml
       trace_detection:
         min_snr: 2.0  # Default is 3.0

2. **Adjust expected FWHM:**

   .. code-block:: yaml
   
       trace_detection:
         expected_fwhm: 5.0  # Try wider if seeing poor
         # or
         expected_fwhm: 3.0  # Try narrower if sharp

3. **Enable spatial binning:**

   .. code-block:: yaml
   
       binning:
         spatial:
           enabled: true
           factor: 2

4. **Use interactive mode:**

   .. code-block:: bash
   
       kosmos-reduce --input-dir data/ --output-dir reduced/ --mode interactive
   
   Manually select trace position in GUI.


"Too many false trace detections"
----------------------------------

**Problem:** Pipeline detects spurious traces.

**Solution:**

Increase detection threshold and minimum separation:

.. code-block:: yaml

    # config.yaml
    trace_detection:
      min_snr: 4.0        # Stricter threshold
      min_separation: 30  # Wider spacing


Extraction Issues
=================

"Negative flux values in extracted spectrum"
---------------------------------------------

**Problem:** Sky over-subtracted or poor profile fit.

**Diagnosis:**

Check sky estimation:

.. code-block:: python

    from pykosmospp.extraction.sky import estimate_sky_background
    
    sky_2d = estimate_sky_background(data, trace, sky_buffer=30)
    
    # Visualize
    plt.plot(np.median(sky_2d, axis=0))
    plt.title('Estimated Sky Background')
    plt.show()

**Solutions:**

1. **Increase sky buffer:**

   .. code-block:: yaml
   
       extraction:
         sky_buffer: 50  # Move sky regions away from trace wings

2. **Use boxcar extraction:**

   .. code-block:: yaml
   
       extraction:
         method: boxcar  # Simpler, less sensitive to profile errors


"Low flux conservation"
-----------------------

**Problem:** Extracted 1D flux doesn't match 2D aperture sum.

**Solution:**

Increase aperture width to capture all flux:

.. code-block:: yaml

    extraction:
      aperture_width: 15  # Default is 10


Quality Issues
==============

"Low SNR warning"
-----------------

**Problem:** Signal-to-noise ratio below threshold.

**This is not an error**, just a quality flag. Options:

1. **Accept lower quality:** Spectrum still usable for some science

2. **Re-observe with longer integration:**
   
   * SNR improves as √(exposure time)
   * To double SNR, need 4× exposure time

3. **Spatial binning:**

   .. code-block:: yaml
   
       binning:
         spatial:
           enabled: true
           factor: 2  # Increases SNR by √2


"Poor wavelength calibration grade"
------------------------------------

**Problem:** Wavelength RMS above threshold.

**Impact:**

* Redshift measurements less accurate
* Line identifications may be uncertain

**Solutions:**

See "Wavelength Calibration Issues" above.


Performance Issues
==================

"Pipeline runs very slowly"
---------------------------

**Expected Performance:** ~15 seconds per science frame on modern hardware.

**Optimizations:**

1. **Use batch mode:**

   .. code-block:: bash
   
       kosmos-reduce --mode batch  # Faster than interactive

2. **Limit trace extraction:**

   .. code-block:: bash
   
       kosmos-reduce --max-traces 2  # Process only first 2 traces

3. **Disable diagnostic plots:**
   
   (Feature not yet implemented, but in roadmap)

4. **Use SSD for I/O:**
   
   * Move data to fast storage
   * Output to SSD, copy to network storage later


"Out of memory errors"
----------------------

**Problem:** Large FITS files exhaust RAM.

**Solutions:**

1. **Process in batches:**

   .. code-block:: bash
   
       # Process 5 frames at a time
       kosmos-reduce --input-dir data/subset1/ --output-dir reduced/
       kosmos-reduce --input-dir data/subset2/ --output-dir reduced/

2. **Close Python sessions:**
   
   Large arrays may not be garbage collected immediately

3. **Increase system RAM or swap:**
   
   Minimum 8 GB RAM recommended for KOSMOS data


Debugging Tips
==============

Enable Verbose Logging
-----------------------

.. code-block:: bash

    kosmos-reduce --input-dir data/ --output-dir reduced/ --verbose

Logs show detailed processing steps and intermediate values.


Check Intermediate Products
----------------------------

.. code-block:: bash

    # Verify calibrations created
    ls -lh reduced/calibrations/
    
    # Check if 2D spectra processed
    ls reduced/reduced_2d/
    
    # Verify 1D extraction
    ls reduced/spectra_1d/


Use Python Interactively
-------------------------

.. code-block:: python

    # Load modules and test step-by-step
    from pykosmospp.pipeline import PipelineRunner
    from pathlib import Path
    
    runner = PipelineRunner(
        input_dir=Path('data'),
        output_dir=Path('reduced')
    )
    
    # Run individual stages
    # (requires modifying PipelineRunner to expose internal methods)


Examine FITS Headers
---------------------

.. code-block:: bash

    # List all header keywords
    fitsheader science_001.fits
    
    # Check specific keyword
    fitsheader -k IMAGETYP -k EXPTIME science_001.fits


Validate FITS Files
-------------------

.. code-block:: bash

    # Check FITS integrity
    fitsverify science_001.fits


Getting Help
============

**GitHub Issues:** https://github.com/gkhullar/pykosmospp/issues

When reporting issues, include:

1. **Pipeline version:**

   .. code-block:: bash
   
       kosmos-reduce --version
       # or
       pip show pykosmospp

2. **Full error message:**
   
   Copy complete traceback from terminal

3. **Input file info:**

   .. code-block:: bash
   
       ls -lh data/
       fitsheader science_001.fits | head -20

4. **Configuration file:**
   
   Attach your custom config.yaml (if used)

5. **Log file:**
   
   Attach reduction_log.txt from output directory


**Email:** gk@astro.washington.edu

For sensitive data or private discussions.


**Documentation:** https://pykosmospp.readthedocs.io

Search docs for specific error messages or concepts.


See Also
========

* :ref:`quickstart` - Getting started guide
* :ref:`cli_reference` - Command-line options
* :ref:`configuration` - Configuration parameters
* :ref:`output_products` - Understanding outputs
