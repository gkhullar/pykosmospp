# pyKOSMOS++ Documentation

This directory contains the Sphinx documentation source for pyKOSMOS++.

**Author:** Gourav Khullar

## Building Documentation Locally

### Prerequisites

Install documentation dependencies:

```bash
pip install -r requirements.txt
```

Or install with the project:

```bash
pip install -e ".[docs]"
```

### Build HTML Documentation

```bash
# From the docs/ directory
make html

# Output will be in build/html/
# Open build/html/index.html in a browser
```

### Build PDF Documentation

```bash
make latexpdf

# Output will be in build/latex/pyKOSMOS.pdf
```

### Clean Build Artifacts

```bash
make clean
```

## Documentation Structure

```
docs/
├── source/
│   ├── conf.py                 # Sphinx configuration
│   ├── index.rst               # Main documentation page
│   ├── installation.rst        # Installation guide
│   ├── quickstart.rst          # 5-minute quick start
│   ├── tutorial.rst            # Complete tutorial (links to Jupyter notebook)
│   ├── user_guide/             # User guides
│   │   ├── overview.rst
│   │   ├── configuration.rst
│   │   ├── calibration.rst
│   │   ├── wavelength.rst
│   │   ├── extraction.rst
│   │   ├── quality.rst
│   │   └── batch_processing.rst
│   ├── api/                    # API reference
│   │   ├── index.rst
│   │   ├── pipeline.rst
│   │   ├── models.rst
│   │   ├── calibration.rst
│   │   ├── wavelength.rst
│   │   ├── extraction.rst
│   │   ├── quality.rst
│   │   └── io.rst
│   ├── developer/              # Developer documentation
│   │   ├── contributing.rst
│   │   ├── architecture.rst
│   │   ├── testing.rst
│   │   └── algorithms.rst
│   ├── changelog.rst           # Version history
│   ├── faq.rst                 # Frequently asked questions
│   ├── troubleshooting.rst     # Common issues and solutions
│   └── references.rst          # Citations and references
├── build/                      # Generated documentation (not version controlled)
├── requirements.txt            # Documentation dependencies
├── Makefile                    # Build commands (Unix/Linux/macOS)
└── make.bat                    # Build commands (Windows)
```

## Read the Docs Integration

This project is configured for automatic documentation builds on [Read the Docs](https://readthedocs.org/).

### Setup Steps

1. **Create Read the Docs Account**: Sign up at https://readthedocs.org/
2. **Import Project**:
   - Click "Import a Project"
   - Select the GitHub repository: `pykosmos_specllm`
   - RTD will automatically detect `.readthedocs.yaml` configuration
3. **Configure Webhook** (automatic):
   - RTD creates a webhook in your GitHub repo
   - Docs rebuild automatically on every push to `main`
4. **Access Documentation**:
   - Live docs at: `https://pykosmos-specllm.readthedocs.io/`
   - Version-specific: `https://pykosmos-specllm.readthedocs.io/en/latest/`

### Configuration File

See `.readthedocs.yaml` in the repository root for RTD configuration:

- **Python version**: 3.10
- **Build dependencies**: Installs from `docs/requirements.txt` and package
- **Output formats**: HTML, PDF, EPUB
- **Sphinx version**: Specified in `docs/requirements.txt`

## Writing Documentation

### reStructuredText (reST) Syntax

Sphinx uses reStructuredText markup. Key syntax:

```rst
Section Title
=============

Subsection
----------

**Bold text**
*Italic text*

- Bullet list
- Another item

1. Numbered list
2. Another item

Code block:

.. code-block:: python

    from pykosmos_spec_ai.pipeline import PipelineRunner
    runner = PipelineRunner(...)

External link: `GitHub <https://github.com/>`_
Internal link: :doc:`installation`
API reference: :func:`pykosmos_spec_ai.pipeline.run`
```

### Auto-Documentation

Sphinx automatically generates API documentation from docstrings:

```python
def my_function(param1: int, param2: str) -> bool:
    """Short description of function.
    
    Longer description with more details about the function's
    purpose and behavior.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When parameter is invalid
        
    Example:
        >>> my_function(42, "test")
        True
    """
    return True
```

### Building Incrementally

For fast iteration during documentation writing:

```bash
# Watch for changes and rebuild automatically
sphinx-autobuild source build/html

# Open browser to http://127.0.0.1:8000/
# Docs auto-refresh when you save changes
```

Install `sphinx-autobuild`:

```bash
pip install sphinx-autobuild
```

## Troubleshooting

**Build Warnings**

Most warnings are harmless, but fix them to ensure RTD builds succeed:

```bash
# Show all warnings
make html

# Treat warnings as errors (for CI)
SPHINXOPTS="-W" make html
```

**Missing Dependencies**

If Sphinx can't import modules:

```bash
# Ensure package is installed in editable mode
pip install -e .

# Verify imports work
python -c "import pykosmos_spec_ai; print('OK')"
```

**Theme Not Found**

Install the Read the Docs theme:

```bash
pip install sphinx-rtd-theme
```

## Contributing

When adding new modules or features:

1. **Write docstrings**: Use Google or NumPy style
2. **Add user guide**: Create `.rst` file in `source/user_guide/`
3. **Update API reference**: Add module to `source/api/`
4. **Build locally**: Test with `make html`
5. **Submit PR**: Documentation updates are reviewed alongside code

See `developer/contributing.rst` for full guidelines.
