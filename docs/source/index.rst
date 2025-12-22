.. pyKOSMOS++ documentation master file

pyKOSMOS++ Documentation
========================

**pyKOSMOS++** is an AI-assisted spectroscopic reduction pipeline for APO-KOSMOS longslit observations.

**Author:** Gourav Khullar

**pyKOSMOS++** is an AI-assisted spectroscopic reduction pipeline for APO-KOSMOS longslit observations built with modern spec-driven development and LLM assistance. 
It automates the workflow from raw CCD images to wavelength-calibrated 1D spectra with quality assessment.

.. image:: https://img.shields.io/badge/Python-3.10+-blue.svg
   :alt: Python Version
   :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/badge/License-MIT-green.svg
   :alt: License
   :target: https://opensource.org/licenses/MIT

Key Features
------------

✨ **Automated Calibration**: Master bias and flat creation with validation

🌈 **Wavelength Calibration**: Arc line detection, catalog matching, and polynomial fitting (RMS <0.2Å)

📊 **Trace Detection**: Cross-correlation with Gaussian templates for robust trace identification

🔬 **Optimal Extraction**: Variance-weighted extraction with cosmic ray rejection

📈 **Quality Assessment**: SNR computation, profile consistency, and grading (Excellent/Good/Fair/Poor)

🚀 **Batch Processing**: Automated pipeline for multiple observations

🧪 **Interactive Mode**: Visual trace selection and parameter tuning

📝 **Comprehensive Documentation**: Tutorial notebook, user guides, and API reference


Quick Links
-----------

* :doc:`installation` - Get started in 5 minutes
* :doc:`quickstart` - Run your first reduction
* :doc:`tutorial` - Complete workflow walkthrough
* :doc:`api/index` - Full API reference
* `GitHub Repository <https://github.com/gkhullar/pykosmospp>`_

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   installation
   quickstart
   tutorial
   user_guide/overview
   user_guide/configuration
   user_guide/calibration
   user_guide/wavelength
   user_guide/extraction
   user_guide/quality
   user_guide/batch_processing

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/index
   api/pipeline
   api/models
   api/calibration
   api/wavelength
   api/extraction
   api/quality
   api/io

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide
   :hidden:

   developer/contributing
   developer/architecture
   developer/testing
   developer/algorithms

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources
   :hidden:

   changelog
   faq
   troubleshooting
   references


Getting Started
---------------

Installation
^^^^^^^^^^^^

Install via pip from the repository root::

    pip install -e .

Or install with optional development dependencies::

    pip install -e ".[dev,docs]"

See :doc:`installation` for detailed instructions including conda environments.

Basic Usage
^^^^^^^^^^^

Reduce a single observation directory::

    from pykosmos_spec_ai.pipeline import PipelineRunner
    from pathlib import Path

    runner = PipelineRunner(
        input_dir=Path("data/2024-01-15/galaxy_NGC1234"),
        output_dir=Path("reduced_output"),
        mode="batch"
    )
    
    reduced_data_list = runner.run()

See :doc:`quickstart` for a 5-minute introduction.


Pipeline Workflow
-----------------

The pyKOSMOS++ pipeline follows standard spectroscopic reduction steps:

1. **Calibration Creation**
   
   * Combine bias frames → Master Bias
   * Combine flat frames → Master Flat (normalized)
   * Validate calibration quality

2. **Wavelength Calibration**
   
   * Detect arc emission lines
   * Match lines to catalog (He-Ne-Ar)
   * Fit Chebyshev polynomial (BIC order selection)
   * Validate RMS residual <0.2Å

3. **Trace Detection**
   
   * Cross-correlate with Gaussian templates
   * Identify spectral traces (SNR ≥ 3σ)
   * Fit spatial trace positions

4. **Sky Subtraction**
   
   * Estimate background from sky regions (±30px buffer)
   * Apply sigma-clipping (3σ)
   * Subtract 2D sky model

5. **Optimal Extraction**
   
   * Compute spatial profile
   * Apply variance-weighted extraction
   * Reject cosmic rays

6. **Quality Assessment**
   
   * Compute SNR, wavelength RMS, profile consistency
   * Assign grade: Excellent / Good / Fair / Poor
   * Generate diagnostic plots


System Requirements
-------------------

* **Python**: ≥ 3.10
* **Memory**: 4 GB minimum (8 GB recommended for large datasets)
* **Storage**: 100 MB installation + observation data (typically 1-10 GB per night)
* **OS**: macOS, Linux, Windows (WSL recommended)


Dependencies
------------

Core scientific libraries:

* **astropy** ≥ 5.3 - FITS I/O, units, coordinates
* **specutils** ≥ 1.10 - Spectrum manipulation
* **scipy** ≥ 1.10 - Signal processing, optimization
* **numpy** ≥ 1.23 - Array operations
* **matplotlib** ≥ 3.6 - Visualization
* **pyyaml** ≥ 6.0 - Configuration parsing

See ``pyproject.toml`` for complete dependency list.


Citation
--------

If you use pyKOSMOS++ in your research, please cite::

    @software{pykosmospp2025,
      author = {Gourav Khullar},
      title = {pyKOSMOS++: AI-Assisted Spectroscopic Reduction Pipeline for APO-KOSMOS},
      year = {2025},
      publisher = {GitHub},
      url = {https://github.com/gkhullar/pykosmospp}
    }


Support
-------

* **Issues**: `GitHub Issues <https://github.com/gkhullar/pykosmospp/issues>`_
* **Discussions**: `GitHub Discussions <https://github.com/gkhullar/pykosmospp/discussions>`_
* **Author**: Gourav Khullar


License
-------

This project is licensed under the MIT License - see the `LICENSE <https://github.com/gkhullar/pykosmospp/blob/main/LICENSE>`_ file for details.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
