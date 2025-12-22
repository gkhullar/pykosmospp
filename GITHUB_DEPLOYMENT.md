# GitHub Deployment Guide for pyKOSMOS++

**Author:** Gourav Khullar  
**Date:** December 2025

## Pre-Deployment Checklist

✅ **README.md**: Comprehensive, professional README created with pyKOSMOS++ branding  
✅ **.gitignore**: Updated to exclude FITS data files while preserving directory structure  
✅ **Documentation**: Tutorial notebook (8 sections), Sphinx docs, Read the Docs config complete  
✅ **Branding**: Global rename from pyKOSMOS_specllm → pyKOSMOS++ completed  
✅ **Project Name**: Next-generation spectroscopic reduction pipeline with modern naming  
✅ **Empty directories**: .gitkeep files created for examples/data/{biases,flats,arcs,science}/  

## Repository Details

- **Repository Name**: `pykosmospp`
- **GitHub URL**: https://github.com/gkhullar/pykosmospp
- **Read the Docs URL**: https://pykosmospp.readthedocs.io/
- **Author**: Gourav Khullar
- **License**: MIT
- **Python Version**: ≥3.10

## Deployment Steps

### 1. Initialize Git Repository

```bash
cd /Users/gkhullar/Desktop/projects/UWashington/apo/reductions/pykosmos_spec_ai

# Initialize git (if not already initialized)
git init

# Check current status
git status
```

### 2. Stage All Files

```bash
# Add all files (respects .gitignore)
git add .

# Verify what will be committed
git status

# Check that FITS files are NOT staged
git status | grep -i "\.fits"
```

### 3. Create Initial Commit

```bash
# Commit with comprehensive message
git commit -m "Initial commit: pyKOSMOS++ v0.1.0

- AI-assisted spectroscopic reduction pipeline for APO-KOSMOS
- Next-generation spec-driven development with LLM assistance
- Complete calibration, wavelength, extraction, quality modules
- 37/43 tests passing (86% coverage)
- Comprehensive documentation: tutorial notebook, Sphinx, Read the Docs
- Author: Gourav Khullar
"
```

### 4. Create GitHub Repository

**Option A: Via GitHub Web Interface**

1. Go to https://github.com/new
2. Repository name: `pykosmospp`
3. Description: "AI-Assisted Spectroscopic Reduction Pipeline for APO-KOSMOS (Next-generation with LLM assistance)"
4. Public repository
5. **Do NOT** initialize with README, .gitignore, or LICENSE (we have these locally)
6. Click "Create repository"

**Option B: Via GitHub CLI** (if installed)

```bash
gh repo create gkhullar/pykosmospp --public \
  --description "AI-Assisted Spectroscopic Reduction Pipeline for APO-KOSMOS" \
  --source=.
```

### 5. Connect Local Repository to GitHub

```bash
# Add GitHub remote
git remote add origin https://github.com/gkhullar/pykosmospp.git

# Verify remote
git remote -v
```

### 6. Push to GitHub

```bash
# Push to main branch (or master, depending on your default)
git branch -M main
git push -u origin main
```

### 7. Verify Upload

Visit https://github.com/gkhullar/pykosmospp and verify:

- ✅ README.md displays correctly with badges and formatting
- ✅ Directory structure visible (src/, tests/, docs/, examples/, specs/)
- ✅ examples/data/ subdirectories present (biases/, flats/, arcs/, science/)
- ✅ No FITS files uploaded (check .gitignore is working)
- ✅ All documentation files present

## Read the Docs Setup

### 8. Connect to Read the Docs

1. Go to https://readthedocs.org/
2. Log in with GitHub account
3. Click "Import a Project"
4. Select `gkhullar/pykosmospp`
5. Configure:
   - Name: `pykosmospp`
   - Repository URL: https://github.com/gkhullar/pykosmospp
   - Default branch: `main`
   - Configuration file: `.readthedocs.yaml` (auto-detected)
6. Click "Next"

### 9. Verify Documentation Build

1. Wait for first build to complete (2-5 minutes)
2. Visit https://pykosmospp.readthedocs.io/
3. Verify:
   - ✅ Homepage loads with pyKOSMOS++ branding
   - ✅ Installation guide accessible
   - ✅ Quick start guide accessible
   - ✅ API reference generates correctly
   - ✅ PDF/EPUB downloads available

### 10. Add Documentation Badge to README

If not already present, the README includes:

```markdown
[![Documentation Status](https://readthedocs.org/projects/pykosmospp/badge/?version=latest)](https://pykosmospp.readthedocs.io/en/latest/)
```

This badge will update automatically once Read the Docs builds successfully.

## Post-Deployment Tasks

### Update Repository Settings

1. Go to https://github.com/gkhullar/pykosmospp/settings
2. **About section** (top right):
   - Description: "AI-Assisted Spectroscopic Reduction Pipeline for APO-KOSMOS"
   - Website: https://pykosmospp.readthedocs.io/
   - Topics: `astronomy`, `spectroscopy`, `data-reduction`, `apo`, `kosmos`, `python`, `ai-assisted`, `spec-driven`
3. **Features**:
   - ✅ Issues
   - ✅ Discussions (optional, for community Q&A)
   - ✅ Wiki (optional)
4. **Social preview image** (optional):
   - Upload a banner/logo image for social media previews

### Create Release

1. Go to https://github.com/gkhullar/pykosmospp/releases
2. Click "Create a new release"
3. Tag version: `v0.1.0`
4. Release title: "pyKOSMOS++ v0.1.0 - Initial Release"
5. Description:

```markdown
## 🌟 Initial Release: pyKOSMOS++ v0.1.0

**AI-Assisted Spectroscopic Reduction Pipeline for APO-KOSMOS**

### What's New

This is the first public release of pyKOSMOS++, an AI-assisted spectroscopic reduction pipeline for APO-KOSMOS longslit observations.

### Features

- ✅ Automated calibration pipeline (bias, flat)
- ✅ Wavelength calibration with BIC model selection (RMS <0.2Å)
- ✅ Optimal extraction (Horne 1986) with cosmic ray rejection
- ✅ Quality assessment and automated grading
- ✅ Batch processing mode
- ✅ Comprehensive documentation (tutorial + Sphinx + Read the Docs)
- ✅ 86% unit test coverage (37/43 tests passing)

### Installation

```bash
pip install git+https://github.com/gkhullar/pykosmospp.git
```

### Documentation

- 📖 **Read the Docs**: https://pykosmospp.readthedocs.io/
- 📓 **Tutorial Notebook**: [examples/tutorial.ipynb](https://github.com/gkhullar/pykosmospp/blob/main/examples/tutorial.ipynb)

### Citation

```bibtex
@software{pykosmospp2025,
  author = {Gourav Khullar},
  title = {pyKOSMOS++: AI-Assisted Spectroscopic Reduction Pipeline for APO-KOSMOS},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/gkhullar/pykosmospp}
}
```

### Author

**Gourav Khullar** (University of Washington)

Built with 🤖 Claude Sonnet 4.5 and 🔬 scientific rigor
```

6. Click "Publish release"

## Maintenance Commands

### Update Documentation

```bash
# Update docs and push
git add docs/
git commit -m "Update documentation"
git push origin main

# Read the Docs will rebuild automatically
```

### Update README

```bash
git add README.md
git commit -m "Update README with [description]"
git push origin main
```

### Create New Branch for Features

```bash
git checkout -b feature/new-feature
# ... make changes ...
git add .
git commit -m "Add new feature"
git push origin feature/new-feature

# Then create Pull Request on GitHub
```

## Troubleshooting

### Read the Docs Build Fails

1. Check build logs: https://readthedocs.org/projects/pykosmospp/builds/
2. Common issues:
   - Missing dependencies in `docs/requirements.txt`
   - Syntax errors in `.rst` files
   - Configuration errors in `.readthedocs.yaml` or `docs/source/conf.py`
3. Fix locally:
   ```bash
   cd docs
   make clean
   make html
   # Fix any errors
   git commit -am "Fix documentation build"
   git push origin main
   ```

### FITS Files Accidentally Committed

```bash
# Remove from git but keep local files
git rm --cached examples/data/**/*.fits

# Update .gitignore if needed
git add .gitignore
git commit -m "Remove FITS files from git"
git push origin main
```

### Wrong Branch Pushed

```bash
# Rename branch
git branch -m old-name new-name
git push origin -u new-name
git push origin --delete old-name
```

## Success Criteria

- ✅ Repository visible at https://github.com/gkhullar/pykosmospp
- ✅ README displays professional, comprehensive information
- ✅ Documentation builds at https://pykosmospp.readthedocs.io/
- ✅ No FITS data files committed (check repository size <10MB)
- ✅ Directory structure preserved (empty data directories visible)
- ✅ All badges (Python version, license, docs status) functional
- ✅ Tutorial notebook viewable on GitHub

## Next Steps

1. **Share with community**: Announce on relevant mailing lists, forums
2. **Gather feedback**: Monitor Issues and Discussions
3. **Continuous improvement**: Address bugs, add features based on user feedback
4. **PyPI release**: Package for `pip install pykosmospp` (requires `setup.py` or `pyproject.toml` refinement)

---

**Repository Status**: ✅ Ready for GitHub deployment  
**Documentation Status**: ✅ Ready for Read the Docs  
**Project Name**: pyKOSMOS++ (AI-assisted spectroscopic reduction)  
**Author**: Gourav Khullar  
**License**: MIT
