"""
Unit tests for wavelength calibration module.

Per T117: Tests arc line identification, line matching,
polynomial fitting, wavelength application with pyKOSMOS linelists.
"""

import pytest
import numpy as np
from astropy import units as u
from specutils import Spectrum1D

from src.wavelength.identify import detect_arc_lines
from src.wavelength.match import match_lines_to_catalog, load_linelist
from src.wavelength.fit import fit_wavelength_solution
from src.wavelength.apply import apply_wavelength_to_spectrum, bin_spectral


class TestArcLineIdentification:
    """Test arc line detection."""
    
    def test_detect_arc_lines_basic(self):
        """Test basic arc line detection."""
        # Create synthetic arc spectrum with known lines
        data = np.ones((1000, 50)) * 100  # Background
        
        # Add emission lines at known positions
        line_positions = [200, 400, 600, 800]
        for pos in line_positions:
            y = np.arange(50)
            for i in range(max(0, pos-10), min(1000, pos+10)):
                spatial_profile = np.exp(-((y - 25) / 2)**2)
                spectral_profile = np.exp(-((i - pos) / 1.5)**2)
                data[i, :] += spatial_profile * spectral_profile * 5000
        
        lines = detect_arc_lines(data, min_snr=5.0)
        
        assert len(lines) >= 3, "Should detect multiple arc lines"
        # Check that detected lines are near expected positions
        detected_positions = [line['pixel_position'] for line in lines]
        for expected in line_positions:
            assert any(np.abs(pos - expected) < 20 for pos in detected_positions)


class TestLineMatching:
    """Test line matching to catalog."""
    
    def test_match_lines_to_catalog(self):
        """Test matching detected lines to catalog."""
        # Simulated detected lines (pixel positions)
        detected_lines = [
            {'pixel_position': 100, 'intensity': 1000},
            {'pixel_position': 500, 'intensity': 2000},
            {'pixel_position': 900, 'intensity': 1500}
        ]
        
        # Simulated catalog (wavelengths in Å)
        catalog = {
            'wavelengths': np.array([4000.0, 5000.0, 6000.0]),
            'intensities': np.array([1.0, 2.0, 1.5])
        }
        
        matched = match_lines_to_catalog(
            detected_lines,
            catalog,
            initial_dispersion=1.0  # ~1 Å/pixel
        )
        
        assert len(matched) > 0, "Should match at least some lines"
        assert 'pixel_position' in matched[0]
        assert 'wavelength' in matched[0]


class TestWavelengthSolutionFitting:
    """Test wavelength solution fitting."""
    
    def test_fit_wavelength_solution_linear(self):
        """Test linear wavelength solution fit."""
        # Create perfect linear relationship
        pixels = np.array([100, 300, 500, 700, 900])
        wavelengths = 4000 + pixels * 1.0  # 1 Å/pixel
        
        matched_lines = [
            {'pixel_position': p, 'wavelength': w}
            for p, w in zip(pixels, wavelengths)
        ]
        
        from src.models import ArcFrame
        mock_arc = ArcFrame(Path("mock.fits"))
        
        solution = fit_wavelength_solution(matched_lines, arc_frame=mock_arc, max_order=1)
        
        assert solution.order >= 1
        assert solution.rms_residual < 0.1  # Should fit perfectly
        
        # Test prediction
        predicted = solution.pixel_to_wavelength(500)
        assert np.abs(predicted - 4500) < 1
    
    def test_fit_wavelength_solution_polynomial(self):
        """Test polynomial wavelength solution."""
        # Create nonlinear relationship
        pixels = np.linspace(0, 1000, 20)
        wavelengths = 4000 + pixels * 1.0 + 0.0001 * pixels**2
        
        matched_lines = [
            {'pixel_position': p, 'wavelength': w}
            for p, w in zip(pixels, wavelengths)
        ]
        
        from src.models import ArcFrame
        from pathlib import Path
        mock_arc = ArcFrame(Path("mock.fits"))
        
        solution = fit_wavelength_solution(matched_lines, arc_frame=mock_arc, max_order=3)
        
        assert solution.order >= 2  # Should use polynomial
        assert solution.rms_residual < 1.0


class TestWavelengthApplication:
    """Test applying wavelength solution to spectrum."""
    
    def test_apply_wavelength_to_spectrum(self):
        """Test wavelength application."""
        # Create mock 1D spectrum
        flux = np.random.normal(1000, 100, 500)
        wavelength = np.arange(500) * u.Angstrom  # Placeholder
        
        spectrum = Spectrum1D(
            flux=flux * u.electron,
            spectral_axis=wavelength
        )
        
        # Create mock wavelength solution
        from src.models import WavelengthSolution, ArcFrame
        from pathlib import Path
        
        mock_arc = ArcFrame(Path("mock.fits"))
        coefficients = np.array([4000.0, 1.0, 0.0001])  # λ = 4000 + 1*x + 0.0001*x²
        
        solution = WavelengthSolution(
            coefficients=coefficients,
            order=2,
            rms_residual=0.05,
            n_lines_matched=20,
            arc_frame=mock_arc
        )
        
        calibrated = apply_wavelength_to_spectrum(spectrum, solution)
        
        assert len(calibrated.wavelength) == 500
        assert calibrated.wavelength[0] > 4000  # Should start around 4000 Å


class TestSpectralBinning:
    """Test spectral binning."""
    
    def test_bin_spectral(self):
        """Test spectral binning."""
        # Create high-res spectrum
        wavelength = np.linspace(4000, 6000, 2000) * u.Angstrom
        flux = np.random.normal(1000, 100, 2000) * u.electron
        
        spectrum = Spectrum1D(flux=flux, spectral_axis=wavelength)
        
        # Bin to lower resolution
        binned = bin_spectral(spectrum, bin_width_angstrom=5.0)
        
        # Should have fewer pixels
        assert len(binned.flux) < len(spectrum.flux)
        # Wavelength range should be preserved
        assert np.abs(binned.wavelength[0].value - 4000) < 10
        assert np.abs(binned.wavelength[-1].value - 6000) < 10
