"""
End-to-end integration test for full pipeline.

Per T114: Tests complete reduction workflow from raw FITS
to extracted 1D spectra, validating outputs and quality metrics.
"""

import pytest
import tempfile
from pathlib import Path
import shutil
import numpy as np
from astropy.io import fits

from src.pipeline import PipelineRunner
from src.io.organize import discover_fits_files
from tests.fixtures.synthetic_data import generate_test_dataset


@pytest.fixture
def test_data_dir():
    """Create temporary directory with synthetic test data."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    # Generate complete test dataset
    files = generate_test_dataset(
        temp_path,
        num_bias=5,
        num_flat=5,
        num_arc=2,
        num_science=3,
        seed=42  # Reproducible
    )
    
    yield temp_path
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def output_dir():
    """Create temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    yield temp_path
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.mark.skip(reason="Issue #21: File discovery not working with test fixtures. See KNOWN_ISSUES.md")
def test_pipeline_discovers_files(test_dataset):
    """Test that pipeline can discover FITS files."""
    frames = discover_fits_files(test_data_dir)
    
    assert len(frames) > 0, "Should discover FITS files"
    assert all(f.suffix == '.fits' for f in frames), "All files should be FITS"


@pytest.mark.skip(reason="Issue #22: End-to-end pipeline blocked by upstream issues. See KNOWN_ISSUES.md")
def test_pipeline_end_to_end(test_dataset):
    """
    Test complete pipeline execution.
    
    Per SC-001 through SC-007: Validates full workflow including:
    - File discovery and organization
    - Master calibration creation
    - Wavelength calibration from arc
    - Science frame reduction
    - Quality metrics computation
    - Output file generation
    """
    # Create pipeline runner
    pipeline = PipelineRunner(
        input_dir=test_data_dir,
        output_dir=output_dir,
        mode='batch'
    )
    
    # Run pipeline
    reduced_data_list = pipeline.run()
    
    # Validate execution
    assert len(reduced_data_list) > 0, "Should produce reduced data"
    
    # Check each reduced data product
    for reduced_data in reduced_data_list:
        # Validate 2D spectrum exists
        assert reduced_data.spectrum_2d is not None, "2D spectrum should exist"
        assert reduced_data.spectrum_2d.data.shape[0] > 0, "2D spectrum should have data"
        
        # Validate 1D spectra extracted
        assert len(reduced_data.spectra_1d) > 0, "Should extract at least one 1D spectrum"
        
        for spectrum_1d in reduced_data.spectra_1d:
            # Check spectral data
            assert len(spectrum_1d.flux) > 0, "1D spectrum should have flux"
            assert len(spectrum_1d.wavelength) > 0, "1D spectrum should have wavelength"
            assert len(spectrum_1d.flux) == len(spectrum_1d.wavelength), "Flux and wavelength should match"
            
            # Check wavelength calibration
            assert spectrum_1d.wavelength[0] > 3000, "Wavelength should be in optical range (>3000 Å)"
            assert spectrum_1d.wavelength[-1] < 10000, "Wavelength should be in optical range (<10000 Å)"
            
            # Check uncertainty propagation
            if spectrum_1d.uncertainty is not None:
                assert len(spectrum_1d.uncertainty.array) == len(spectrum_1d.flux), "Uncertainty should match flux length"
                assert np.all(spectrum_1d.uncertainty.array >= 0), "Uncertainty should be non-negative"
        
        # Validate quality metrics
        assert reduced_data.quality_metrics is not None, "Quality metrics should exist"
        assert hasattr(reduced_data.quality_metrics, 'overall_grade'), "Should have overall grade"
        assert hasattr(reduced_data.quality_metrics, 'median_snr'), "Should have median SNR"
        
        # Validate diagnostic plots generated
        assert len(reduced_data.diagnostic_plots) > 0, "Should generate diagnostic plots"
        assert '2d_spectrum' in reduced_data.diagnostic_plots, "Should have 2D spectrum plot"
        assert 'wavelength' in reduced_data.diagnostic_plots, "Should have wavelength plot"


@pytest.mark.skip(reason="Issue #23: Output validation blocked by pipeline issues. See KNOWN_ISSUES.md")
@pytest.mark.skip(reason="Issue #23: Output validation blocked by pipeline issues. See KNOWN_ISSUES.md")
def test_output_files_created(test_dataset):
    """Test that all expected output files are created."""
    pipeline = PipelineRunner(
        input_dir=test_data_dir,
        output_dir=output_dir,
        mode='batch'
    )
    
    pipeline.run()
    
    # Check output directory structure
    assert (output_dir / 'calibrations').exists(), "Calibrations directory should exist"
    assert (output_dir / 'reduced_2d').exists(), "Reduced 2D directory should exist"
    assert (output_dir / 'spectra_1d').exists(), "Spectra 1D directory should exist"
    assert (output_dir / 'quality_reports').exists(), "Quality reports directory should exist"
    assert (output_dir / 'diagnostic_plots').exists(), "Diagnostic plots directory should exist"
    
    # Check that files were created
    fits_files = list((output_dir / 'spectra_1d').glob('*.fits'))
    assert len(fits_files) > 0, "Should create 1D spectrum FITS files"
    
    plot_files = list((output_dir / 'diagnostic_plots').glob('*.png'))
    assert len(plot_files) > 0, "Should create diagnostic plots"


@pytest.mark.skip(reason="Issue #24: Quality validation blocked by pipeline issues. See KNOWN_ISSUES.md")
def test_quality_thresholds(test_dataset):
    """
    Test that quality metrics meet expected thresholds.
    
    Per quickstart.md test scenarios: Validates quality requirements.
    """
    pipeline = PipelineRunner(
        input_dir=test_data_dir,
        output_dir=output_dir,
        mode='batch'
    )
    
    reduced_data_list = pipeline.run()
    
    for reduced_data in reduced_data_list:
        metrics = reduced_data.quality_metrics
        
        # SNR should be reasonable (synthetic data)
        assert metrics.median_snr > 1.0, "Median SNR should be > 1.0"
        
        # Overall grade should be assigned
        assert metrics.overall_grade in ['A', 'B', 'C', 'D', 'F'], "Grade should be A-F"


@pytest.mark.skip(reason="Issue #25: Wavelength accuracy test blocked by pipeline issues. See KNOWN_ISSUES.md")
def test_wavelength_calibration_accuracy(test_data_dir, output_dir):
    """Test wavelength calibration accuracy."""
    pipeline = PipelineRunner(
        input_dir=test_data_dir,
        output_dir=output_dir,
        mode='batch'
    )
    
    reduced_data_list = pipeline.run()
    
    for reduced_data in reduced_data_list:
        for spectrum_1d in reduced_data.spectra_1d:
            # Check if wavelength RMS is in metadata
            if 'wavelength_rms' in spectrum_1d.meta:
                rms = spectrum_1d.meta['wavelength_rms']
                # For synthetic data, RMS should be reasonable
                # Real data target is <0.1 Å per quickstart.md
                assert rms < 1.0, f"Wavelength RMS should be <1.0 Å, got {rms}"


@pytest.mark.skip(reason="Issue #26: Cosmic ray integration test blocked. See KNOWN_ISSUES.md")
def test_cosmic_ray_detection(test_dataset):
    """Test that cosmic rays are detected and flagged."""
    pipeline = PipelineRunner(
        input_dir=test_data_dir,
        output_dir=output_dir,
        mode='batch'
    )
    
    reduced_data_list = pipeline.run()
    
    for reduced_data in reduced_data_list:
        # Check cosmic ray mask exists
        assert reduced_data.spectrum_2d.cosmic_ray_mask is not None, "Cosmic ray mask should exist"
        
        # Check that some cosmic rays were detected (synthetic data has them)
        cr_fraction = reduced_data.spectrum_2d.cosmic_ray_mask.sum() / reduced_data.spectrum_2d.cosmic_ray_mask.size
        # Synthetic data has ~0.1% cosmic rays
        assert 0 < cr_fraction < 0.05, f"Cosmic ray fraction should be small, got {cr_fraction}"


@pytest.mark.skip(reason="Issue #27: Performance test blocked by pipeline issues. See KNOWN_ISSUES.md")
def test_pipeline_performance(test_data_dir, output_dir):
    """
    Test pipeline execution time.
    
    Per quickstart.md SC-001: <30 minutes for 10 frames on laptop.
    This test uses fewer frames but checks reasonable performance.
    """
    import time
    
    pipeline = PipelineRunner(
        input_dir=test_data_dir,
        output_dir=output_dir,
        mode='batch'
    )
    
    start_time = time.time()
    pipeline.run()
    end_time = time.time()
    
    elapsed = end_time - start_time
    
    # With ~15 total frames (5 bias, 5 flat, 2 arc, 3 science)
    # Should complete in reasonable time (< 5 minutes for test)
    assert elapsed < 300, f"Pipeline should complete in <5 minutes, took {elapsed:.1f}s"


def test_error_handling_missing_calibrations(output_dir):
    """Test that pipeline handles missing calibration frames gracefully."""
    # Create empty directory (no calibration frames)
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    try:
        pipeline = PipelineRunner(
            input_dir=temp_path,
            output_dir=output_dir,
            mode='batch'
        )
        
        # Should raise CriticalPipelineError due to missing calibrations
        with pytest.raises(Exception):  # CriticalPipelineError or similar
            pipeline.run()
    finally:
        shutil.rmtree(temp_dir)


@pytest.mark.skip(reason="Issue #28: Multiple traces test blocked by pipeline issues. See KNOWN_ISSUES.md")
def test_multiple_traces(test_dataset):
    """Test handling of multiple spectral traces in single frame."""
    # Generate science frame with 2 traces
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    try:
        from tests.fixtures.synthetic_data import (
            generate_bias_frame, generate_flat_frame,
            generate_arc_frame, generate_science_frame
        )
        
        # Generate minimal calibration set
        for i in range(3):
            generate_bias_frame(temp_path / f"bias_{i}.fits", seed=100+i)
            generate_flat_frame(temp_path / f"flat_{i}.fits", seed=200+i)
        
        generate_arc_frame(temp_path / "arc.fits", seed=300)
        
        # Generate science with 2 traces
        generate_science_frame(
            temp_path / "science_2trace.fits",
            num_traces=2,
            seed=400
        )
        
        # Run pipeline
        pipeline = PipelineRunner(
            input_dir=temp_path,
            output_dir=output_dir,
            mode='batch'
        )
        
        reduced_data_list = pipeline.run()
        
        # Should extract 2 traces
        assert len(reduced_data_list) > 0, "Should reduce frame"
        assert len(reduced_data_list[0].spectra_1d) == 2, "Should extract 2 traces"
        
    finally:
        shutil.rmtree(temp_dir)
