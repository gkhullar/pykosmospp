Installation
============

This guide covers multiple installation methods for pyKOSMOS++.

**Author:** Gourav Khullar

**Note:** This pipeline extends `pyKOSMOS <https://github.com/jradavenport/pykosmos>`_ by James R. A. Davenport (University of Washington). 
Reference data (arc lamp linelists, extinction curves) is included from the pyKOSMOS resource directory.

Requirements
------------

**System Requirements**

* Python ≥ 3.10
* 4 GB RAM minimum (8 GB recommended)
* 100 MB disk space for installation
* Git (for development installation)

**Operating Systems**

* macOS (Intel & Apple Silicon)
* Linux (Ubuntu 20.04+, CentOS 8+, Debian 11+)
* Windows via WSL2 (recommended) or native Python


Quick Install (pip)
-------------------

For standard use, clone the repository and install with pip:

.. code-block:: bash

    # Clone repository
    git clone https://github.com/gkhullar/pykosmospp.git
    cd pykosmospp

    # Install in editable mode
    pip install -e .

This installs pyKOSMOS++ and all required dependencies.


Development Install
-------------------

For development work including testing and documentation:

.. code-block:: bash

    # Install with development dependencies
    pip install -e ".[dev,docs]"

This adds:

* **dev**: pytest, pytest-cov, black, ruff
* **docs**: sphinx, sphinx-rtd-theme, sphinx-autodoc-typehints


Conda Environment (Recommended)
--------------------------------

Using conda provides better dependency management, especially for scientific packages:

.. code-block:: bash

    # Create conda environment with Python 3.10
    conda create -n pykosmos python=3.10
    conda activate pykosmos

    # Install core scientific dependencies via conda
    conda install -c conda-forge astropy scipy numpy matplotlib pyyaml

    # Install pyKOSMOS and remaining dependencies
    cd pykosmos_spec_ai
    pip install -e .

**Advantages of conda:**

* Pre-compiled binaries for numpy, scipy (faster)
* Better compatibility on Apple Silicon Macs
* Isolated environment avoids conflicts


Virtual Environment (Alternative)
----------------------------------

For lightweight isolation without conda:

.. code-block:: bash

    # Create virtual environment
    python3.10 -m venv venv_pykosmos
    source venv_pykosmos/bin/activate  # On Windows: venv_pykosmos\Scripts\activate

    # Install pyKOSMOS
    cd pykosmos_spec_ai
    pip install -e .


Dependencies
------------

**Core Dependencies** (automatically installed)

.. code-block:: text

    astropy >= 5.3      # FITS I/O, units, coordinates
    specutils >= 1.10   # Spectrum manipulation
    scipy >= 1.10       # Signal processing, optimization
    numpy >= 1.23       # Array operations
    matplotlib >= 3.6   # Visualization
    pyyaml >= 6.0       # Configuration parsing

**Development Dependencies** (optional)

.. code-block:: text

    pytest >= 7.0           # Testing framework
    pytest-cov >= 4.0       # Coverage reporting
    black >= 23.0           # Code formatting
    ruff >= 0.0.270         # Linting

**Documentation Dependencies** (optional)

.. code-block:: text

    sphinx >= 5.0                    # Documentation generator
    sphinx-rtd-theme >= 1.2          # Read the Docs theme
    sphinx-autodoc-typehints >= 1.23 # Type hint support


Verify Installation
-------------------

Check that pyKOSMOS is correctly installed:

.. code-block:: bash

    # Test import
    python -c "import pykosmos_spec_ai; print('Installation successful!')"

    # Run test suite (if dev dependencies installed)
    pytest tests/

Expected output for tests::

    ============================= test session starts ==============================
    collected 43 items

    tests/test_bias.py ....                                                  [ 9%]
    tests/test_cosmic.py .                                                   [ 11%]
    tests/test_flat.py .....                                                 [ 23%]
    tests/test_wavelength.py ...........                                     [ 48%]
    tests/test_extraction.py ............                                    [ 76%]
    tests/test_quality.py ..........                                         [100%]

    ========================== 37 passed, 6 skipped in 2.34s =======================


Platform-Specific Notes
-----------------------

macOS
^^^^^

**Apple Silicon (M1/M2/M3)**

Conda is strongly recommended for Apple Silicon. Some dependencies require Rosetta 2 or native ARM builds:

.. code-block:: bash

    # Install Rosetta 2 if not already installed
    softwareupdate --install-rosetta

    # Use conda with osx-arm64 platform
    conda create -n pykosmos python=3.10
    conda activate pykosmos
    conda install -c conda-forge astropy scipy numpy matplotlib
    pip install -e .

**Intel Macs**

Standard pip or conda installation works without issues.


Linux
^^^^^

**Ubuntu/Debian**

Install system dependencies for matplotlib:

.. code-block:: bash

    sudo apt-get update
    sudo apt-get install python3.10 python3.10-dev python3-pip
    sudo apt-get install libfreetype6-dev libpng-dev  # For matplotlib

    pip install -e .

**CentOS/RHEL**

.. code-block:: bash

    sudo yum install python3.10 python3.10-devel
    sudo yum install freetype-devel libpng-devel

    pip install -e .


Windows
^^^^^^^

**WSL2 (Recommended)**

Install Ubuntu via WSL2, then follow Linux instructions:

.. code-block:: bash

    # In WSL2 Ubuntu terminal
    sudo apt-get update
    sudo apt-get install python3.10 python3.10-dev python3-pip
    pip install -e .

**Native Windows**

Use Anaconda or Miniconda for best compatibility:

1. Download `Miniconda for Windows <https://docs.conda.io/en/latest/miniconda.html>`_
2. Open Anaconda Prompt
3. Follow conda installation instructions above


Troubleshooting
---------------

**Import Error: No module named 'pykosmos_spec_ai'**

Ensure you're in the correct Python environment:

.. code-block:: bash

    which python  # Should point to venv or conda environment
    pip list | grep pykosmos  # Should show pykosmospp

**Compilation Errors for scipy/numpy**

Use conda instead of pip for these packages:

.. code-block:: bash

    conda install -c conda-forge scipy numpy

**matplotlib Backend Issues**

If plots don't display, set a non-interactive backend:

.. code-block:: python

    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

**Permission Errors on macOS**

Use virtual environment instead of system Python:

.. code-block:: bash

    python3.10 -m venv venv_pykosmos
    source venv_pykosmos/bin/activate
    pip install -e .


Upgrading
---------

To upgrade to the latest version:

.. code-block:: bash

    cd pykosmos_spec_ai
    git pull origin main
    pip install -e . --upgrade

For major version upgrades, recreate the environment:

.. code-block:: bash

    # Conda
    conda deactivate
    conda env remove -n pykosmos
    conda create -n pykosmos python=3.10
    conda activate pykosmos
    pip install -e .

    # Venv
    deactivate
    rm -rf venv_pykosmos
    python3.10 -m venv venv_pykosmos
    source venv_pykosmos/bin/activate
    pip install -e .


Uninstalling
------------

.. code-block:: bash

    # Remove package
    pip uninstall pykosmospp

    # Remove conda environment (if used)
    conda deactivate
    conda env remove -n pykosmos

    # Remove repository (optional)
    cd ..
    rm -rf pykosmos_spec_ai


Next Steps
----------

* :doc:`quickstart` - Run your first reduction in 5 minutes
* :doc:`tutorial` - Complete tutorial notebook
* :doc:`user_guide/configuration` - Configure pipeline parameters
