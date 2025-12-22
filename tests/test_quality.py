"""
Unit tests for quality assessment module.

Tests T057-T059: QualityMetrics, validate_calibrations, compute_quality_metrics
"""

import pytest
import numpy as np
from astropy.nddata import CCDData, StdDevUncertainty
import astropy.units as u
from specutils import Spectrum1D

from src.models import (
    QualityMetrics, MasterBias, MasterFlat, BiasFrame, FlatFrame,
    Spectrum2D, ScienceFrame
)
from src.quality.validate import validate_calibrations
from src.quality.metrics import compute_quality_metrics, _assign_grade
from pathlib import Path


class TestQualityMetrics:
    """Test QualityMetrics class (T057)"""
    
    def test_quality_metrics_initialization(self):
        """Test QualityMetrics initialization"""
        qm = QualityMetrics()
        
        assert qm.median_snr is None
        assert qm.wavelength_rms is None
        assert qm.sky_residual_rms is None
        assert qm.cosmic_ray_fraction == 0.0
        assert qm.saturation_flag is False
        assert qm.overall_grade == 'Unknown'
    
    def test_quality_metrics_compute(self):
        """Test QualityMetrics.compute() method"""
        # Create synthetic spectrum with good SNR
        wavelength = np.linspace(4000, 7000, 1000) * u.AA
        flux = np.ones(1000) * 100 + np.random.normal(0, 5, 1000)
        uncertainty = StdDevUncertainty(np.ones(1000) * 5)
        
        spectrum_1d = Spectrum1D(
            spectral_axis=wavelength,
            flux=flux * u.electron,
            uncertainty=uncertainty,
            meta={'wavelength_rms': 0.08}
        )
        
        qm = QualityMetrics()
        qm.compute(spectrum_1d)
        
        assert qm.median_snr is not None
        assert qm.median_snr > 10  # Should have good SNR
        assert qm.wavelength_rms == 0.08
        assert qm.overall_grade in ['Excellent', 'Good', 'Fair', 'Poor']
    
    def test_quality_metrics_generate_report(self):
        """Test QualityMetrics.generate_report() method"""
        qm = QualityMetrics()
        qm.median_snr = 25.3
        qm.wavelength_rms = 0.075
        qm.sky_residual_rms = 5.2
        qm.cosmic_ray_fraction = 0.002
        qm.saturation_flag = False
        qm.overall_grade = 'Excellent'
        
        report = qm.generate_report()
        
        assert 'Quality Assessment Report' in report
        assert 'Excellent' in report
        assert '25.3' in report
        assert '0.075' in report
        assert 'No' in report  # Saturation


class TestValidateCalibrations:
    """Test validate_calibrations function (T058)"""
    
    def test_validate_good_calibrations(self):
        """Test validation of good quality calibrations"""
        # Create good master bias (low variation)
        bias_data = np.random.normal(100, 2, (100, 100))
        bias_ccd = CCDData(bias_data, unit=u.electron)
        master_bias = MasterBias(
            data=bias_ccd,
            n_combined=3,
            bias_level=100.0,
            bias_stdev=2.0
        )
        
        # Create good master flat (proper normalization)
        flat_data = np.random.normal(30000, 1000, (100, 100))
        flat_ccd = CCDData(flat_data, unit=u.electron)
        master_flat = MasterFlat(
            data=flat_ccd,
            n_combined=3,
            normalization_region=(10, 90, 10, 90),
            bad_pixel_fraction=0.001
        )
        
        results = validate_calibrations(master_bias, master_flat)
        
        assert results['bias_valid'] is True
        assert results['flat_valid'] is True
        assert results['overall_valid'] is True
        assert results['bias_variation'] < 10.0
    
    def test_validate_bad_bias(self):
        """Test validation fails for high bias variation"""
        # Create bad master bias (high variation)
        bias_data = np.random.normal(100, 15, (100, 100))
        bias_ccd = CCDData(bias_data, unit=u.electron)
        master_bias = MasterBias(
            data=bias_ccd,
            n_combined=3,
            bias_level=100.0,
            bias_stdev=15.0
        )
        
        # Create good flat
        flat_data = np.random.normal(30000, 1000, (100, 100))
        flat_ccd = CCDData(flat_data, unit=u.electron)
        master_flat = MasterFlat(
            data=flat_ccd,
            n_combined=3,
            normalization_region=(10, 90, 10, 90),
            bad_pixel_fraction=0.001
        )
        
        results = validate_calibrations(master_bias, master_flat)
        
        assert results['bias_valid'] is False
        assert results['overall_valid'] is False
    
    def test_validate_saturated_flat(self):
        """Test validation fails for saturated flat"""
        # Create good bias
        bias_data = np.random.normal(100, 2, (100, 100))
        bias_ccd = CCDData(bias_data, unit=u.electron)
        master_bias = MasterBias(
            data=bias_ccd,
            n_combined=3,
            bias_level=100.0,
            bias_stdev=2.0
        )
        
        # Create saturated flat
        flat_data = np.random.normal(30000, 1000, (100, 100))
        flat_data[0:20, 0:20] = 65535  # Saturate 4% of pixels
        flat_ccd = CCDData(flat_data, unit=u.electron)
        master_flat = MasterFlat(
            data=flat_ccd,
            n_combined=3,
            normalization_region=(10, 90, 10, 90),
            bad_pixel_fraction=0.001
        )
        
        results = validate_calibrations(master_bias, master_flat)
        
        assert results['flat_valid'] is False
        assert results['flat_saturation_fraction'] > 0.01
        assert results['overall_valid'] is False


class TestComputeQualityMetrics:
    """Test compute_quality_metrics function (T059)"""
    
    def test_compute_quality_metrics_excellent(self):
        """Test metrics computation for excellent spectrum"""
        # Create high SNR spectrum with good wavelength solution
        wavelength = np.linspace(4000, 7000, 1000) * u.AA
        flux = np.ones(1000) * 500 + np.random.normal(0, 10, 1000)
        uncertainty = StdDevUncertainty(np.ones(1000) * 10)
        
        spectrum_1d = Spectrum1D(
            spectral_axis=wavelength,
            flux=flux * u.electron,
            uncertainty=uncertainty,
            meta={'wavelength_rms': 0.05}
        )
        
        metrics = compute_quality_metrics(spectrum_1d)
        
        assert metrics['median_snr'] > 20
        assert metrics['wavelength_rms'] == 0.05
        assert metrics['overall_grade'] == 'Excellent'
        assert metrics['saturation_flag'] is False
    
    def test_compute_quality_metrics_poor(self):
        """Test metrics computation for poor spectrum"""
        # Create low SNR spectrum with poor wavelength solution
        wavelength = np.linspace(4000, 7000, 1000) * u.AA
        flux = np.ones(1000) * 10 + np.random.normal(0, 5, 1000)
        uncertainty = StdDevUncertainty(np.ones(1000) * 5)
        
        spectrum_1d = Spectrum1D(
            spectral_axis=wavelength,
            flux=flux * u.electron,
            uncertainty=uncertainty,
            meta={'wavelength_rms': 0.35}
        )
        
        metrics = compute_quality_metrics(spectrum_1d)
        
        assert metrics['median_snr'] < 10
        assert metrics['wavelength_rms'] == 0.35
        assert metrics['overall_grade'] == 'Poor'
    
    def test_compute_quality_metrics_with_2d_spectrum(self):
        """Test metrics computation with 2D spectrum for cosmic ray fraction"""
        # Create 1D spectrum
        wavelength = np.linspace(4000, 7000, 1000) * u.AA
        flux = np.ones(1000) * 100 + np.random.normal(0, 5, 1000)
        
        spectrum_1d = Spectrum1D(
            spectral_axis=wavelength,
            flux=flux * u.electron
        )
        
        # Create 2D spectrum with cosmic ray mask
        data_2d = np.random.normal(100, 10, (50, 1000))
        cr_mask = np.zeros((50, 1000), dtype=bool)
        cr_mask[10:15, 100:105] = True  # Mark some cosmic rays
        
        # Create mock source frame
        mock_file = Path("/tmp/mock_science.fits")
        
        spectrum_2d = Spectrum2D(
            data=data_2d,
            variance=data_2d.copy(),
            source_frame=None,  # Mock - not needed for test
            cosmic_ray_mask=cr_mask
        )
        
        metrics = compute_quality_metrics(spectrum_1d, spectrum_2d)
        
        assert 'cosmic_ray_fraction' in metrics
        assert metrics['cosmic_ray_fraction'] > 0
        assert metrics['cosmic_ray_fraction'] < 0.01
    
    def test_assign_grade_thresholds(self):
        """Test grade assignment thresholds"""
        # Excellent
        metrics_excellent = {
            'median_snr': 25.0,
            'wavelength_rms': 0.08,
            'saturation_flag': False
        }
        assert _assign_grade(metrics_excellent) == 'Excellent'
        
        # Good
        metrics_good = {
            'median_snr': 15.0,
            'wavelength_rms': 0.15,
            'saturation_flag': False
        }
        assert _assign_grade(metrics_good) == 'Good'
        
        # Fair
        metrics_fair = {
            'median_snr': 7.0,
            'wavelength_rms': 0.25,
            'saturation_flag': False
        }
        assert _assign_grade(metrics_fair) == 'Fair'
        
        # Poor
        metrics_poor = {
            'median_snr': 3.0,
            'wavelength_rms': 0.40,
            'saturation_flag': False
        }
        assert _assign_grade(metrics_poor) == 'Poor'
        
        # Saturation should prevent Excellent
        metrics_saturated = {
            'median_snr': 30.0,
            'wavelength_rms': 0.05,
            'saturation_flag': True
        }
        assert _assign_grade(metrics_saturated) != 'Excellent'
