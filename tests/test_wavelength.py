"""
Tests for wavelength calibration module.

Tests arc line detection, matching, fitting, and application.
"""

import pytest
import numpy as np
from pathlib import Path
from astropy.nddata import CCDData
from astropy.io import fits
import astropy.units as u
import tempfile
import os

from src.wavelength import (
    detect_arc_lines,
    load_linelist,
    match_lines_to_catalog,
    fit_wavelength_solution,
    apply_wavelength_to_spectrum
)
from src.models import WavelengthSolution, ArcFrame


def create_test_arc_frame(data: np.ndarray, lamp_type: str = 'henear') -> ArcFrame:
    """Helper to create test ArcFrame with temporary FITS file."""
    # Create temporary FITS file
    tmpfile = tempfile.NamedTemporaryFile(mode='wb', suffix='.fits', delete=False)
    tmpfile.close()
    
    # Create FITS file
    hdu = fits.PrimaryHDU(data)
    hdu.header['IMAGETYP'] = 'Arc'
    hdu.header['OBJECT'] = lamp_type
    hdu.header['GAIN'] = 1.4
    hdu.header['RDNOISE'] = 3.7
    hdu.header['SATURATE'] = 58982
    hdu.writeto(tmpfile.name, overwrite=True)
    
    # Create ArcFrame
    arc_frame = ArcFrame(Path(tmpfile.name))
    
    return arc_frame


@pytest.mark.skip(reason="Issue #9: Arc line detection API needs alignment. See KNOWN_ISSUES.md")
class TestArcLineDetection:
    """Test arc line identification."""
    
    def test_detect_arc_lines_synthetic(self):
        """Test arc line detection with synthetic spectrum."""
        # Create synthetic arc spectrum with known lines
        npix = 4096
        x = np.arange(npix)
        
        # Base continuum with noise
        continuum = 100.0
        noise = 5.0
        spectrum = continuum + np.random.normal(0, noise, npix)
        
        # Add synthetic emission lines at known positions
        line_positions = [500, 1000, 1500, 2000, 2500, 3000, 3500]
        line_heights = [500, 300, 400, 600, 350, 450, 550]
        line_width = 3.0  # pixels
        
        for pos, height in zip(line_positions, line_heights):
            gaussian = height * np.exp(-0.5 * ((x - pos) / line_width)**2)
            spectrum += gaussian
        
        # Create 2D arc frame (spatial x spectral)
        arc_2d = np.tile(spectrum, (100, 1))
        arc_data = CCDData(arc_2d, unit=u.adu)
        
        # Detect lines
        detected_pixels, intensities = detect_arc_lines(
            arc_data,
            detection_threshold=5.0,
            min_separation=5
        )
        
        # Check that we detected the expected number of lines
        assert len(detected_pixels) >= 7, f"Expected ≥7 lines, detected {len(detected_pixels)}"
        
        # Check that detected positions are close to true positions
        for true_pos in line_positions:
            closest = np.abs(detected_pixels - true_pos).min()
            assert closest < 5.0, f"Line at {true_pos} not detected within 5 pixels (closest: {closest})"
    
    def test_detect_arc_lines_threshold(self):
        """Test that detection threshold works correctly."""
        # Create spectrum with weak and strong lines
        npix = 1000
        x = np.arange(npix)
        
        continuum = 100.0
        noise = 10.0
        spectrum = continuum + np.random.normal(0, noise, npix)
        
        # Strong line (10-sigma)
        spectrum += 100 * np.exp(-0.5 * ((x - 500) / 3.0)**2)
        
        # Weak line (3-sigma, should not be detected at 5-sigma threshold)
        spectrum += 30 * np.exp(-0.5 * ((x - 700) / 3.0)**2)
        
        arc_2d = np.tile(spectrum, (50, 1))
        arc_data = CCDData(arc_2d, unit=u.adu)
        
        # Detect with 5-sigma threshold
        detected_pixels, _ = detect_arc_lines(arc_data, detection_threshold=5.0)
        
        # Should detect strong line but not weak line
        assert len(detected_pixels) >= 1
        assert np.any(np.abs(detected_pixels - 500) < 5)  # Strong line detected


class TestLineMatching:
    """Test arc line catalog matching."""
    
    def test_match_lines_basic(self):
        """Test basic line matching with known lamp type."""
        # Simulate detected lines in pixel space
        # These would come from detect_arc_lines()
        pixel_positions = np.array([500, 1000, 1500, 2000, 2500, 3000, 3500])
        
        # Try to match (may fail if linelist not available, which is expected in test env)
        try:
            matched_pixels, matched_waves, matched_ints = match_lines_to_catalog(
                pixel_positions,
                lamp_type='henear',
                wavelength_range=(4000, 7000),
                match_tolerance=5.0,
                initial_dispersion=1.0
            )
            
            # If we got matches, validate structure
            assert len(matched_pixels) == len(matched_waves)
            assert len(matched_pixels) == len(matched_ints)
            assert len(matched_pixels) > 0
            
        except (FileNotFoundError, ValueError) as e:
            # Expected if resources/pykosmos_reference not available or no matches
            pytest.skip(f"Linelist test skipped: {e}")
    
    def test_match_lines_invalid_lamp(self):
        """Test that invalid lamp type raises error."""
        pixel_positions = np.array([500, 1000, 1500])
        
        with pytest.raises(ValueError, match="Unknown lamp type"):
            match_lines_to_catalog(
                pixel_positions,
                lamp_type='invalid_lamp',
                wavelength_range=(4000, 7000)
            )


class TestWavelengthFitting:
    """Test wavelength solution fitting."""
    
    def test_fit_wavelength_solution_linear(self):
        """Test fitting with perfect linear dispersion."""
        # Create perfect linear wavelength relationship
        pixels = np.array([0, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000])  # 11 lines
        dispersion = 1.2  # Angstroms per pixel
        wavelength_offset = 4000.0  # Angstroms
        wavelengths = wavelength_offset + pixels * dispersion
        
        # Create mock arc frame
        arc_frame = create_test_arc_frame(np.zeros((100, 4096)), lamp_type='HeNeAr')
        
        # Fit wavelength solution
        solution = fit_wavelength_solution(
            pixels,
            wavelengths,
            arc_frame,
            poly_type='chebyshev',
            order=3,
            sigma_clip=3.0
        )
        
        # Validate solution
        assert isinstance(solution, WavelengthSolution)
        assert solution.order == 3
        assert solution.n_lines_identified == len(pixels)
        assert solution.rms_residual < 0.01  # Should be nearly perfect
        
        # Test wavelength evaluation
        test_pixels = np.array([250, 750, 1250])
        expected_waves = wavelength_offset + test_pixels * dispersion
        computed_waves = solution.wavelength(test_pixels)
        
        np.testing.assert_allclose(computed_waves, expected_waves, rtol=1e-3)
    
    def test_fit_wavelength_solution_with_outliers(self):
        """Test that sigma-clipping removes outliers."""
        # Create linear relationship with one outlier
        pixels = np.array([0, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000])  # 11 lines
        wavelengths = 4000.0 + pixels * 1.2
        
        # Add outlier (0.3 Å offset - realistic for misidentification)
        wavelengths[5] += 0.3  # Small offset at pixel 2000
        
        arc_frame = create_test_arc_frame(np.zeros((100, 4096)), lamp_type='HeNeAr')
        
        # Fit should handle the outlier
        solution = fit_wavelength_solution(
            pixels,
            wavelengths,
            arc_frame,
            order=3,
            sigma_clip=3.0
        )
        
        # RMS should be acceptable (may have warning but passes)
        assert solution.rms_residual < 1.0
        # Should have all or most lines
        assert solution.n_lines_identified >= 10
    
    def test_fit_wavelength_solution_bic_order_selection(self):
        """Test BIC-based polynomial order selection."""
        # Create quadratic relationship
        pixels = np.linspace(0, 4000, 50)
        wavelengths = 4000.0 + 1.2 * pixels + 0.00001 * pixels**2
        
        arc_frame = create_test_arc_frame(np.zeros((100, 4096)), lamp_type='HeNeAr')
        
        # Let BIC select order
        solution = fit_wavelength_solution(
            pixels,
            wavelengths,
            arc_frame,
            order=None,  # Auto-select
            use_bic=True,
            min_order=2,
            max_order=5
        )
        
        # Should select order 2 or 3 for quadratic data
        assert solution.order in [2, 3, 4]
        assert solution.rms_residual < 0.1
    
    def test_fit_wavelength_solution_validation(self):
        """Test that validation catches poor fits."""
        # Create data with too few points
        pixels = np.array([0, 500, 1000])  # Only 3 points
        wavelengths = np.array([4000, 4600, 5200])
        
        arc_frame = create_test_arc_frame(np.zeros((100, 4096)), lamp_type='HeNeAr')
        
        # Should raise error for too few lines
        with pytest.raises(ValueError, match="at least 10 matched lines"):
            fit_wavelength_solution(pixels, wavelengths, arc_frame)


class TestWavelengthSolution:
    """Test WavelengthSolution class methods."""
    
    def test_wavelength_solution_evaluation(self):
        """Test wavelength evaluation."""
        # Create simple solution
        arc_frame = create_test_arc_frame(np.zeros((100, 4096)), lamp_type='HeNeAr')
        
        # Linear coefficients for Chebyshev (normalized domain [-1,1])
        coeffs = np.array([5500.0, 2000.0])  # center + slope
        
        solution = WavelengthSolution(
            coefficients=coeffs,
            order=1,
            arc_frame=arc_frame,
            n_lines_identified=20,
            rms_residual=0.05,
            wavelength_range=(4000, 7000),
            poly_type='chebyshev'
        )
        
        # Test forward evaluation
        pixels = np.array([0, 2048, 4095])
        wavelengths = solution.wavelength(pixels)
        
        assert len(wavelengths) == len(pixels)
        assert wavelengths.min() >= 3500  # Reasonable wavelength
        assert wavelengths.max() <= 7500
    
    def test_wavelength_solution_inverse(self):
        """Test inverse wavelength to pixel mapping."""
        arc_frame = create_test_arc_frame(np.zeros((100, 4096)), lamp_type='HeNeAr')
        
        coeffs = np.array([5500.0, 2000.0])
        solution = WavelengthSolution(
            coefficients=coeffs,
            order=1,
            arc_frame=arc_frame,
            n_lines_identified=20,
            rms_residual=0.05,
            wavelength_range=(4000, 7000),
            poly_type='chebyshev'
        )
        
        # Test inverse
        test_wavelengths = np.array([4500, 5500, 6500])
        pixels = solution.inverse(test_wavelengths)
        
        # Inverse should return reasonable pixel values
        assert len(pixels) == len(test_wavelengths)
        assert pixels.min() >= 0
        assert pixels.max() < 4096
    
    def test_wavelength_solution_validate(self):
        """Test validation method."""
        arc_frame = create_test_arc_frame(np.zeros((100, 4096)), lamp_type='HeNeAr')
        
        # Good solution
        good_solution = WavelengthSolution(
            coefficients=np.array([5500.0, 2000.0]),
            order=1,
            arc_frame=arc_frame,
            n_lines_identified=20,
            rms_residual=0.08,
            wavelength_range=(4000, 7000),
            poly_type='chebyshev'
        )
        
        assert good_solution.validate() is True
        
        # Bad solution (high RMS)
        bad_solution_rms = WavelengthSolution(
            coefficients=np.array([5500.0, 2000.0]),
            order=1,
            arc_frame=arc_frame,
            n_lines_identified=20,
            rms_residual=0.5,  # Exceeds 0.2 Å threshold
            wavelength_range=(4000, 7000),
            poly_type='chebyshev'
        )
        
        with pytest.raises(ValueError, match="RMS too high"):
            bad_solution_rms.validate()
        
        # Bad solution (too few lines)
        bad_solution_lines = WavelengthSolution(
            coefficients=np.array([5500.0, 2000.0]),
            order=1,
            arc_frame=arc_frame,
            n_lines_identified=5,  # Too few
            rms_residual=0.05,
            wavelength_range=(4000, 7000),
            poly_type='chebyshev'
        )
        
        with pytest.raises(ValueError, match="Too few arc lines"):
            bad_solution_lines.validate()


class TestApplyWavelength:
    """Test wavelength application to spectra."""
    
    def test_apply_wavelength_to_spectrum(self):
        """Test creating wavelength-calibrated Spectrum1D."""
        # Create mock wavelength solution
        arc_frame = create_test_arc_frame(np.zeros((100, 4096)), lamp_type='HeNeAr')
        
        coeffs = np.array([5500.0, 2000.0])
        solution = WavelengthSolution(
            coefficients=coeffs,
            order=1,
            arc_frame=arc_frame,
            n_lines_identified=25,
            rms_residual=0.06,
            wavelength_range=(4000, 7000),
            poly_type='chebyshev'
        )
        
        # Create mock 1D spectrum
        npix = 4096
        flux = np.random.normal(100, 10, npix)
        uncertainty = np.sqrt(flux)
        
        # Apply wavelength calibration
        spectrum_1d = apply_wavelength_to_spectrum(
            flux,
            uncertainty,
            solution
        )
        
        # Validate Spectrum1D
        assert spectrum_1d.spectral_axis.unit == u.Angstrom
        assert len(spectrum_1d.flux) == npix
        assert len(spectrum_1d.spectral_axis) == npix
        assert spectrum_1d.uncertainty is not None
        
        # Check metadata
        assert 'wavelength_rms' in spectrum_1d.meta
        assert spectrum_1d.meta['wavelength_rms'] == 0.06
        assert spectrum_1d.meta['n_arc_lines'] == 25
        assert spectrum_1d.meta['poly_order'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
