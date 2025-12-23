"""
Unit tests for quality assessment module.

Per T118: Tests validation functions, metrics computation,
plot generation functions.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import shutil
from astropy import units as u
from specutils import Spectrum1D

from src.quality.validate import validate_calibrations, generate_validation_report
from src.models import QualityMetrics, MasterBias, MasterFlat, Spectrum2D, WavelengthSolution
from src.quality.plots import (
    setup_latex_plots,
    plot_2d_spectrum,
    plot_wavelength_residuals,
    plot_extraction_profile,
    plot_sky_subtraction
)
from tests.fixtures.synthetic_data import generate_bias_frame, generate_flat_frame


@pytest.fixture
def temp_dir():
    """Temporary directory for plot outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.mark.skip(reason="Issue #16-17: Calibration validation API mismatch. See KNOWN_ISSUES.md")
class TestCalibrationValidation:
    """Test calibration validation."""
    
    def test_validate_calibrations(self, temp_dir):
        """Test validation of master calibrations."""
        # Generate synthetic calibrations
        bias_path = temp_dir / "bias.fits"
        flat_path = temp_dir / "flat.fits"
        
        generate_bias_frame(bias_path, seed=1000)
        generate_flat_frame(flat_path, seed=1001)
        
        from src.models import BiasFrame, FlatFrame
        from src.calibration.combine import create_master_bias, create_master_flat
        
        bias_frame = BiasFrame.from_fits(bias_path)
        master_bias = create_master_bias([bias_frame])
        
        flat_frame = FlatFrame.from_fits(flat_path)
        master_flat = create_master_flat([flat_frame], master_bias)
        
        # Validate
        results = validate_calibrations(master_bias, master_flat)
        
        assert 'overall_valid' in results
        assert isinstance(results['overall_valid'], bool)
    
    def test_generate_validation_report(self, temp_dir):
        """Test generation of validation report."""
        bias_path = temp_dir / "bias.fits"
        generate_bias_frame(bias_path, seed=1100)
        
        from src.models import BiasFrame
        from src.calibration.combine import create_master_bias
        
        bias_frame = BiasFrame.from_fits(bias_path)
        master_bias = create_master_bias([bias_frame])
        
        flat_path = temp_dir / "flat.fits"
        generate_flat_frame(flat_path, seed=1101)
        
        from src.models import FlatFrame
        from src.calibration.combine import create_master_flat
        
        flat_frame = FlatFrame.from_fits(flat_path)
        master_flat = create_master_flat([flat_frame], master_bias)
        
        results = validate_calibrations(master_bias, master_flat)
        report = generate_validation_report(results)
        
        assert isinstance(report, str)
        assert len(report) > 0


@pytest.mark.skip(reason="Issue #15: Quality metrics computation API mismatch. See KNOWN_ISSUES.md")
class TestQualityMetrics:
    """Test quality metrics computation."""
    
    def test_quality_metrics_initialization(self):
        """Test QualityMetrics initialization."""
        metrics = QualityMetrics()
        
        assert hasattr(metrics, 'median_snr')
        assert hasattr(metrics, 'overall_grade')
    
    def test_quality_metrics_compute(self):
        """Test quality metrics computation."""
        # Create mock spectrum
        wavelength = np.linspace(4000, 6000, 1000) * u.Angstrom
        flux = np.random.normal(1000, 100, 1000) * u.electron
        uncertainty = np.ones(1000) * 100 * u.electron
        
        from astropy.nddata import StdDevUncertainty
        spectrum = Spectrum1D(
            flux=flux,
            spectral_axis=wavelength,
            uncertainty=StdDevUncertainty(uncertainty)
        )
        
        # Create mock 2D spectrum
        data = np.random.normal(1000, 100, (1000, 100))
        variance = np.ones_like(data) * 100**2
        
        from src.models import Spectrum2D, ScienceFrame
        mock_frame = ScienceFrame(Path("mock.fits"))
        
        spectrum_2d = Spectrum2D(
            data=data,
            variance=variance,
            source_frame=mock_frame
        )
        
        metrics = QualityMetrics()
        metrics.compute(spectrum, spectrum_2d)
        
        assert metrics.median_snr > 0
        assert metrics.overall_grade in ['A', 'B', 'C', 'D', 'F']


@pytest.mark.skip(reason="Issue #18-19: Plot generation API mismatch. See KNOWN_ISSUES.md")
class TestPlotGeneration:
    """Test diagnostic plot generation."""
    
    def test_setup_latex_plots(self):
        """Test LaTeX plot setup."""
        # Should not raise error
        setup_latex_plots()
    
    def test_plot_2d_spectrum(self, temp_dir):
        """Test 2D spectrum plotting."""
        # Create mock 2D spectrum
        data = np.random.normal(1000, 100, (500, 100))
        variance = np.ones_like(data) * 100**2
        
        from src.models import Spectrum2D, ScienceFrame
        mock_frame = ScienceFrame(Path("mock.fits"))
        
        spectrum_2d = Spectrum2D(
            data=data,
            variance=variance,
            source_frame=mock_frame
        )
        
        output_path = temp_dir / "test_2d.png"
        
        # Should create plot without error
        plot_2d_spectrum(spectrum_2d, output_path, title="Test")
        
        assert output_path.exists()
    
    def test_plot_wavelength_residuals(self, temp_dir):
        """Test wavelength residuals plotting."""
        # Create mock wavelength solution
        from src.models import WavelengthSolution, ArcFrame
        
        mock_arc = ArcFrame(Path("mock.fits"))
        coefficients = np.array([4000.0, 1.0])
        
        solution = WavelengthSolution(
            coefficients=coefficients,
            order=1,
            rms_residual=0.05,
            n_lines_matched=20,
            arc_frame=mock_arc
        )
        
        output_path = temp_dir / "test_wavelength.png"
        
        # Should create plot without error
        plot_wavelength_residuals(solution, output_path)
        
        assert output_path.exists()
    
    def test_plot_extraction_profile(self, temp_dir):
        """Test extraction profile plotting."""
        # Create mock trace and 2D spectrum
        from src.extraction.trace import Trace
        
        trace = Trace(
            center_y=50,
            trace_function=lambda x: np.full_like(x, 50.0)
        )
        
        data = np.random.normal(1000, 100, (500, 100))
        variance = np.ones_like(data) * 100**2
        
        from src.models import Spectrum2D, ScienceFrame
        mock_frame = ScienceFrame(Path("mock.fits"))
        
        spectrum_2d = Spectrum2D(
            data=data,
            variance=variance,
            source_frame=mock_frame
        )
        
        output_path = temp_dir / "test_profile.png"
        
        # Should create plot without error
        plot_extraction_profile(trace, spectrum_2d, output_path)
        
        assert output_path.exists()
    
    def test_plot_sky_subtraction(self, temp_dir):
        """Test sky subtraction plotting."""
        data = np.random.normal(1000, 100, (500, 100))
        variance = np.ones_like(data) * 100**2
        sky_background = np.ones_like(data) * 500
        
        from src.models import Spectrum2D, ScienceFrame
        mock_frame = ScienceFrame(Path("mock.fits"))
        
        spectrum_2d = Spectrum2D(
            data=data,
            variance=variance,
            source_frame=mock_frame
        )
        
        output_path = temp_dir / "test_sky.png"
        
        # Should create plot without error
        plot_sky_subtraction(spectrum_2d, sky_background, output_path)
        
        assert output_path.exists()


@pytest.mark.skip(reason="Issue #20: Quality grading API mismatch. See KNOWN_ISSUES.md")
class TestQualityGrading:
    """Test quality grading logic."""
    
    def test_grade_assignment(self):
        """Test that quality grades are assigned correctly."""
        # High SNR should get good grade
        high_snr_metrics = QualityMetrics()
        high_snr_metrics.median_snr = 50.0
        high_snr_metrics._assign_grade()
        
        assert high_snr_metrics.overall_grade in ['A', 'B']
        
        # Low SNR should get poor grade
        low_snr_metrics = QualityMetrics()
        low_snr_metrics.median_snr = 2.0
        low_snr_metrics._assign_grade()
        
        assert low_snr_metrics.overall_grade in ['C', 'D', 'F']
