.. _contributing:

======================
Contributing Guide
======================

Thank you for your interest in contributing to pyKOSMOS++! This guide will help you get started with development, testing, and submitting contributions.

Development Philosophy
======================

pyKOSMOS++ follows these core principles:

1. **Constitution-Driven Development**: All code must align with established astronomical standards and best practices documented in the project constitution

2. **AI-Augmented Workflow**: We embrace AI tools (LLMs) to accelerate development while maintaining rigor through comprehensive testing and code review

3. **Transparency & Education**: Code should be self-documenting, with clear rationale for algorithmic choices and extensive inline comments

4. **Test-Driven Development**: New features require tests before merge; aim for >80% code coverage


Getting Started
===============

Fork and Clone
--------------

.. code-block:: bash

    # Fork on GitHub first, then:
    git clone https://github.com/YOUR_USERNAME/pykosmospp.git
    cd pykosmospp
    
    # Add upstream remote
    git remote add upstream https://github.com/gkhullar/pykosmospp.git


Development Environment
-----------------------

.. code-block:: bash

    # Create isolated environment
    conda create -n pykosmospp-dev python=3.10
    conda activate pykosmospp-dev
    
    # Install in editable mode with dev dependencies
    pip install -e ".[dev,docs,test]"
    
    # Verify installation
    pytest tests/
    sphinx-build docs/source docs/build


Development Workflow
====================

Feature Branch Workflow
-----------------------

1. **Create feature branch from main:**

   .. code-block:: bash
   
       git checkout main
       git pull upstream main
       git checkout -b feature/my-new-feature

2. **Make changes with incremental commits:**

   .. code-block:: bash
   
       # Edit files
       git add src/module.py tests/test_module.py
       git commit -m "feat: Add optimal extraction variance weighting"
       
       # Use conventional commits:
       # feat: New feature
       # fix: Bug fix
       # docs: Documentation
       # test: Tests
       # refactor: Code restructuring
       # perf: Performance improvement

3. **Keep branch updated:**

   .. code-block:: bash
   
       git fetch upstream
       git rebase upstream/main

4. **Push and open pull request:**

   .. code-block:: bash
   
       git push origin feature/my-new-feature
   
   Then open PR on GitHub with description of changes.


Code Standards
==============

Style Guide
-----------

* **PEP 8** for Python code style
* **Maximum line length:** 100 characters (not 79)
* **Docstrings:** NumPy style

**Example:**

.. code-block:: python

    def fit_wavelength_solution(
        matched_lines: List[Dict[str, float]],
        order_range: Tuple[int, int] = (3, 7),
        sigma_clip: float = 3.0,
        max_iterations: int = 5
    ) -> WavelengthSolution:
        """
        Fit Chebyshev polynomial wavelength solution with BIC order selection.
        
        Uses iterative sigma-clipping to reject outlier lines and Bayesian
        Information Criterion to select optimal polynomial order.
        
        Parameters
        ----------
        matched_lines : List[Dict[str, float]]
            List of matched arc lines with keys 'pixel', 'wavelength', 'intensity'
        order_range : Tuple[int, int], optional
            Range of polynomial orders to test (min, max), default (3, 7)
        sigma_clip : float, optional
            Sigma threshold for outlier rejection, default 3.0
        max_iterations : int, optional
            Maximum iterations for sigma-clipping, default 5
        
        Returns
        -------
        WavelengthSolution
            Fitted wavelength solution with coefficients, order, RMS, etc.
        
        Raises
        ------
        ValueError
            If fewer than 10 matched lines provided
        CriticalPipelineError
            If fit fails to converge
        
        Examples
        --------
        >>> matched_lines = match_lines_to_catalog(detected_lines, 'HeNeAr')
        >>> solution = fit_wavelength_solution(matched_lines, order_range=(3, 7))
        >>> print(f"RMS: {solution.rms_residual:.4f} Å")
        
        Notes
        -----
        Implementation follows Chebyshev polynomial approach from
        pyKOSMOS (Davenport et al. 2023) with BIC model selection
        added for automatic order determination.
        
        References
        ----------
        .. [1] Davenport et al. (2023), pyKOSMOS, DOI:10.5281/zenodo.10152905
        """
        # Implementation...


Type Hints
----------

Use type hints for all function signatures:

.. code-block:: python

    from typing import List, Dict, Optional, Tuple, Union
    from pathlib import Path
    import numpy as np
    from numpy.typing import NDArray
    
    def process_frame(
        data: NDArray[np.float32],
        config: PipelineConfig,
        output_path: Optional[Path] = None
    ) -> Tuple[NDArray[np.float32], Dict[str, float]]:
        """Process single frame."""
        ...


Imports
-------

Organize imports in this order:

.. code-block:: python

    # Standard library
    import os
    import sys
    from pathlib import Path
    from typing import List, Dict
    
    # Third-party
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from scipy.signal import find_peaks
    
    # Local modules
    from pykosmospp.models import RawFrame, WavelengthSolution
    from pykosmospp.io.config import PipelineConfig
    from pykosmospp.calibration.combine import sigma_clipped_median


Testing
=======

Test Structure
--------------

Tests are organized in ``tests/`` directory mirroring ``src/`` structure:

.. code-block:: text

    tests/
    ├── test_calibration.py
    ├── test_wavelength.py
    ├── test_extraction.py
    ├── test_quality.py
    ├── test_integration.py
    └── fixtures/
        └── synthetic_data.py


Running Tests
-------------

.. code-block:: bash

    # Run all tests
    pytest tests/
    
    # Run specific test file
    pytest tests/test_wavelength.py
    
    # Run specific test
    pytest tests/test_wavelength.py::TestWavelengthFitting::test_fit_wavelength_solution
    
    # With coverage
    pytest --cov=pykosmospp --cov-report=html tests/
    
    # View coverage report
    open htmlcov/index.html


Writing Tests
-------------

Use pytest with fixtures:

.. code-block:: python

    import pytest
    import numpy as np
    from pykosmospp.wavelength.fit import fit_wavelength_solution
    
    
    @pytest.fixture
    def matched_lines():
        """Generate synthetic matched arc lines."""
        pixels = np.array([100, 200, 300, 400, 500])
        wavelengths = 4000 + pixels * 1.0  # 1 Å/pixel dispersion
        return [
            {'pixel': p, 'wavelength': w, 'intensity': 1000}
            for p, w in zip(pixels, wavelengths)
        ]
    
    
    def test_fit_wavelength_solution_linear(matched_lines):
        """Test wavelength solution with linear dispersion."""
        solution = fit_wavelength_solution(
            matched_lines,
            order_range=(1, 1)  # Force linear fit
        )
        
        assert solution.order == 1
        assert solution.rms_residual < 0.01  # Should fit perfectly
        assert len(solution.coefficients) == 2
        
        # Check wavelength evaluation
        wavelength = solution.wavelength(100)
        np.testing.assert_allclose(wavelength, 4100, rtol=1e-3)
    
    
    def test_fit_wavelength_solution_outlier_rejection(matched_lines):
        """Test sigma-clipping removes outliers."""
        # Add outlier
        matched_lines.append({
            'pixel': 250,
            'wavelength': 5000,  # Way off
            'intensity': 500
        })
        
        solution = fit_wavelength_solution(
            matched_lines,
            sigma_clip=3.0
        )
        
        # Outlier should be rejected
        assert solution.n_lines_identified < len(matched_lines)
        assert solution.rms_residual < 0.1


Test Data
---------

Use synthetic data for tests (avoid large FITS files in repo):

.. code-block:: python

    # tests/fixtures/synthetic_data.py
    import numpy as np
    from astropy.nddata import CCDData
    from astropy import units as u
    
    
    def generate_synthetic_bias(shape=(515, 2048), bias_level=400, readnoise=3.7):
        """Generate synthetic bias frame."""
        data = np.random.normal(bias_level, readnoise, shape)
        return CCDData(data.astype(np.float32), unit=u.adu)
    
    
    def generate_synthetic_spectrum_2d(
        shape=(515, 2048),
        trace_center=257,
        trace_fwhm=4.0,
        snr=10
    ):
        """Generate synthetic 2D spectrum with Gaussian trace."""
        data = np.zeros(shape)
        spatial = np.arange(shape[0])
        
        # Create Gaussian spatial profile
        for i in range(shape[1]):
            profile = np.exp(-0.5 * ((spatial - trace_center) / trace_fwhm)**2)
            data[:, i] = 1000 * profile
        
        # Add noise
        noise = data / snr
        data += np.random.normal(0, noise)
        
        return data


Documentation
=============

Building Documentation
----------------------

.. code-block:: bash

    # Build HTML docs
    cd docs/
    make html
    
    # View locally
    open build/html/index.html
    
    # Clean and rebuild
    make clean html


Documentation Standards
-----------------------

* All public functions/classes require docstrings
* Use NumPy docstring format
* Include examples in docstrings
* Add new modules to ``docs/source/api/``
* Update user guides for new features


API Documentation
-----------------

Add new modules to API reference:

.. code-block:: rst

    .. _api_wavelength:
    
    ==================
    Wavelength Module
    ==================
    
    .. automodule:: pykosmospp.wavelength.fit
       :members:
       :undoc-members:
       :show-inheritance:


Pull Request Process
====================

Before Submitting
-----------------

1. **Run tests:**

   .. code-block:: bash
   
       pytest tests/
       # All tests must pass

2. **Check code style:**

   .. code-block:: bash
   
       flake8 src/ tests/
       black --check src/ tests/
       
       # Auto-format
       black src/ tests/

3. **Update documentation:**
   
   * Add docstrings to new functions
   * Update user guides if behavior changes
   * Add entry to CHANGELOG.md

4. **Update tests:**
   
   * New features require new tests
   * Aim for >80% coverage of new code


PR Template
-----------

When opening a pull request, use this template:

.. code-block:: markdown

    ## Description
    Brief description of changes.
    
    ## Motivation
    Why is this change needed? What problem does it solve?
    
    ## Changes
    - [ ] Added new feature X
    - [ ] Fixed bug in module Y
    - [ ] Updated documentation
    
    ## Tests
    - [ ] All existing tests pass
    - [ ] Added new tests for new features
    - [ ] Coverage maintained/improved
    
    ## Documentation
    - [ ] Docstrings added/updated
    - [ ] User guide updated (if needed)
    - [ ] CHANGELOG.md updated
    
    ## Related Issues
    Closes #123


Review Process
--------------

* Maintainer will review within 1 week
* Address reviewer comments with new commits
* Once approved, maintainer will merge
* Delete feature branch after merge


Issue Guidelines
================

Reporting Bugs
--------------

Use this template:

.. code-block:: markdown

    **Description:**
    Clear description of the bug.
    
    **To Reproduce:**
    1. Step 1
    2. Step 2
    3. See error
    
    **Expected Behavior:**
    What should happen instead.
    
    **Environment:**
    - OS: macOS 13.2
    - Python: 3.10.8
    - pykosmospp: 0.1.0
    
    **Error Message:**
    ```
    Full traceback here
    ```
    
    **Additional Context:**
    Logs, config files, FITS headers, etc.


Feature Requests
----------------

.. code-block:: markdown

    **Feature Description:**
    What feature would you like?
    
    **Use Case:**
    How would this feature be used? What problem does it solve?
    
    **Proposed Implementation:**
    (Optional) Ideas for how to implement.
    
    **Alternatives:**
    Other approaches you've considered.


Project Structure
=================

.. code-block:: text

    pykosmospp/
    ├── src/
    │   └── pykosmospp/
    │       ├── __init__.py
    │       ├── calibration/      # Bias/flat/cosmic
    │       ├── wavelength/       # Arc line fitting
    │       ├── extraction/       # Trace detection, extraction
    │       ├── quality/          # Metrics, validation
    │       ├── io/               # FITS I/O, config
    │       ├── models.py         # Data classes
    │       ├── pipeline.py       # PipelineRunner
    │       └── cli.py            # Command-line interface
    ├── tests/
    │   ├── test_*.py            # Unit tests
    │   └── fixtures/            # Test data generation
    ├── docs/
    │   └── source/
    │       ├── user_guide/      # CLI, API, config
    │       ├── tutorials/       # Walkthroughs
    │       ├── api/             # Auto-generated API docs
    │       └── algorithms/      # Algorithm descriptions
    ├── examples/
    │   ├── tutorial.ipynb       # Jupyter tutorial
    │   └── data/                # Example data
    ├── config/
    │   └── kosmos_defaults.yaml # Default configuration
    ├── resources/
    │   └── pykosmos_reference/  # Linelists, etc.
    ├── pyproject.toml           # Package metadata
    ├── README.md
    ├── CHANGELOG.md
    └── LICENSE


Code of Conduct
===============

We follow the `Astropy Code of Conduct <https://www.astropy.org/code_of_conduct.html>`_.

Summary:

* Be respectful and inclusive
* Harassment and discrimination not tolerated
* Assume good faith
* Focus on what is best for the community


License
=======

By contributing, you agree that your contributions will be licensed under the MIT License.


Questions?
==========

* **GitHub Discussions:** https://github.com/gkhullar/pykosmospp/discussions
* **Email:** gk@astro.washington.edu

We're happy to help new contributors get started!


See Also
========

* :ref:`quickstart` - Getting started with pyKOSMOS++
* :ref:`python_api` - Python API reference
* :ref:`troubleshooting` - Common issues
