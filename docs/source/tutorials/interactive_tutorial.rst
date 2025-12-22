.. _tutorial_notebook:

=============================
Interactive Tutorial Notebook
=============================

This comprehensive tutorial demonstrates the complete pyKOSMOS++ spectroscopic reduction workflow through an interactive Jupyter notebook.

The tutorial covers:

* **Introduction & Setup** - Installation verification and configuration loading
* **Data Exploration** - Understanding KOSMOS FITS file structure
* **Calibration Creation** - Master bias and flat frame generation with validation
* **Wavelength Calibration** - Arc line detection, catalog matching, and polynomial fitting
* **Trace Detection & Extraction** - Cross-correlation trace detection and optimal extraction
* **Quality Assessment** - SNR computation and quality grading
* **Advanced Parameters** - Customizing reduction parameters for specific observations
* **Batch Processing** - Automated pipeline for multiple observations

Tutorial Notebook
=================

.. nbsphinx:: ../../../examples/tutorial.ipynb

Additional Resources
====================

* :ref:`quickstart` - Quick 5-minute reduction guide
* :ref:`user_guide_cli` - Command-line interface reference
* :ref:`user_guide_python_api` - Python API documentation
* :ref:`troubleshooting` - Common issues and solutions

Download
========

Download the tutorial notebook: `tutorial.ipynb <https://github.com/gkhullar/pykosmospp/blob/main/examples/tutorial.ipynb>`_

To run the tutorial locally:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/gkhullar/pykosmospp.git
   cd pykosmospp
   
   # Install dependencies
   pip install -e ".[dev]"
   
   # Launch Jupyter
   jupyter notebook examples/tutorial.ipynb

Requirements
============

* Python ≥3.10
* pyKOSMOS++ installed with all dependencies
* KOSMOS FITS data (or use generated test data)
* Jupyter notebook or JupyterLab

**Estimated Time:** 15-20 minutes for interactive execution
