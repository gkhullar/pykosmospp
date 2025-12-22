"""
Unit tests for calibration module.

Per T115: Tests frame combination, bias/flat creation,
cosmic ray detection independently with synthetic data.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import shutil
from astropy.nddata import CCDData
import astropy.units as u

from src.calibration.combine import (
    sigma_clipped_median_combine,
    create_master_bias,
    create_master_flat
)
from src.calibration.cosmic import detect_cosmic_rays
from src.models import BiasFrame, FlatFrame
from tests.fixtures.synthetic_data import (
    generate_bias_frame,
    generate_flat_frame,
    KOSMOS_READNOISE,
    KOSMOS_GAIN
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestFrameCombination:
    """Test sigma-clipped median combination."""
    
    def test_median_combine_basic(self):
        """Test basic median combination."""
        # Create 5 frames with known values
        data1 = np.ones((100, 100)) * 100
        data2 = np.ones((100, 100)) * 102
        data3 = np.ones((100, 100)) * 98
        data4 = np.ones((100, 100)) * 101
        data5 = np.ones((100, 100)) * 99
        
        frames = [
            CCDData(data1, unit=u.electron),
            CCDData(data2, unit=u.electron),
            CCDData(data3, unit=u.electron),
            CCDData(data4, unit=u.electron),
            CCDData(data5, unit=u.electron)
        ]
        
        combined = sigma_clipped_median_combine(frames)
        
        # Median should be 100
        assert combined.data.shape == (100, 100)
        assert np.abs(np.median(combined.data) - 100) < 1
        assert 'NCOMBINE' in combined.header
        assert combined.header['NCOMBINE'] == 5
    
    def test_median_combine_rejects_outliers(self):
        """Test that sigma clipping rejects outliers."""
        # Create frames with one outlier
        data_clean = [np.ones((50, 50)) * 100 + np.random.normal(0, 1, (50, 50)) for _ in range(4)]
        data_outlier = np.ones((50, 50)) * 1000  # Extreme outlier
        
        frames = [CCDData(d, unit=u.electron) for d in data_clean] + [CCDData(data_outlier, unit=u.electron)]
        
        combined = sigma_clipped_median_combine(frames, sigma=3.0)
        
        # Should reject outlier and return ~100
        assert np.abs(np.median(combined.data) - 100) < 5
    
    def test_combine_inconsistent_shapes_raises_error(self):
        """Test that combining frames with different shapes raises error."""
        frame1 = CCDData(np.ones((100, 100)), unit=u.electron)
        frame2 = CCDData(np.ones((50, 50)), unit=u.electron)
        
        with pytest.raises(ValueError, match="shape"):
            sigma_clipped_median_combine([frame1, frame2])
    
    def test_combine_empty_list_raises_error(self):
        """Test that combining empty list raises error."""
        with pytest.raises(ValueError, match="empty"):
            sigma_clipped_median_combine([])


class TestMasterBias:
    """Test master bias creation."""
    
    def test_create_master_bias(self, temp_dir):
        """Test master bias creation from synthetic frames."""
        # Generate synthetic bias frames
        bias_files = []
        for i in range(5):
            path = temp_dir / f"bias_{i}.fits"
            generate_bias_frame(path, seed=100+i)
            bias_files.append(path)
        
        # Load as BiasFrame objects
        bias_frames = [BiasFrame.from_fits(f) for f in bias_files]
        
        # Create master bias
        master_bias = create_master_bias(bias_frames, method='median')
        
        # Validate
        assert master_bias.data.shape[0] > 0
        assert master_bias.data.shape[1] > 0
        assert len(master_bias.source_frames) == 5
        assert master_bias.combination_method == 'median'
    
    def test_master_bias_reduces_noise(self, temp_dir):
        """Test that combining reduces noise by sqrt(N)."""
        # Generate bias frames with known noise
        bias_files = []
        for i in range(10):
            path = temp_dir / f"bias_{i}.fits"
            generate_bias_frame(path, seed=200+i)
            bias_files.append(path)
        
        bias_frames = [BiasFrame.from_fits(f) for f in bias_files]
        
        # Measure noise in individual frame
        individual_noise = np.std(bias_frames[0].data.data)
        
        # Create master bias
        master_bias = create_master_bias(bias_frames)
        master_noise = np.std(master_bias.data.data)
        
        # Master should have lower noise (approximately sqrt(N) reduction)
        # For N=10, expect ~3x reduction
        assert master_noise < individual_noise / 2


class TestMasterFlat:
    """Test master flat creation."""
    
    def test_create_master_flat(self, temp_dir):
        """Test master flat creation from synthetic frames."""
        # Generate bias frames first
        bias_files = []
        for i in range(3):
            path = temp_dir / f"bias_{i}.fits"
            generate_bias_frame(path, seed=300+i)
            bias_files.append(path)
        
        bias_frames = [BiasFrame.from_fits(f) for f in bias_files]
        master_bias = create_master_bias(bias_frames)
        
        # Generate flat frames
        flat_files = []
        for i in range(5):
            path = temp_dir / f"flat_{i}.fits"
            generate_flat_frame(path, mean_counts=30000, seed=400+i)
            flat_files.append(path)
        
        flat_frames = [FlatFrame.from_fits(f) for f in flat_files]
        
        # Create master flat
        master_flat = create_master_flat(flat_frames, master_bias, method='median')
        
        # Validate
        assert master_flat.data.shape[0] > 0
        assert len(master_flat.source_frames) == 5
        
        # Flat should be normalized (median ~1.0)
        # Note: May vary based on implementation
        assert 0.5 < np.median(master_flat.data.data) < 2.0
    
    def test_master_flat_removes_bias(self, temp_dir):
        """Test that master flat properly subtracts bias."""
        # Create bias
        bias_path = temp_dir / "bias.fits"
        generate_bias_frame(bias_path, seed=500)
        bias_frame = BiasFrame.from_fits(bias_path)
        master_bias = create_master_bias([bias_frame])
        
        # Create flat with known pedestal
        flat_path = temp_dir / "flat.fits"
        generate_flat_frame(flat_path, mean_counts=20000, seed=600)
        flat_frame = FlatFrame.from_fits(flat_path)
        
        # Create master flat
        master_flat = create_master_flat([flat_frame], master_bias)
        
        # After bias subtraction, mean should be significantly higher than bias level
        assert np.mean(master_flat.data.data) > 0.5


class TestCosmicRayDetection:
    """Test cosmic ray detection algorithm."""
    
    def test_detect_cosmic_rays_basic(self):
        """Test basic cosmic ray detection."""
        # Create synthetic data with artificial cosmic rays
        data = np.ones((100, 100)) * 1000  # Background
        
        # Add cosmic rays
        data[25, 25] = 50000  # Strong CR
        data[50, 50] = 20000  # Moderate CR
        data[75, 75] = 15000  # Weak CR
        
        ccd = CCDData(data, unit=u.electron)
        
        # Detect cosmic rays
        cr_mask = detect_cosmic_rays(
            ccd,
            sigma_clip=5.0,
            readnoise=KOSMOS_READNOISE,
            gain=KOSMOS_GAIN
        )
        
        # Should detect cosmic rays
        assert cr_mask[25, 25], "Should detect strong CR"
        assert cr_mask[50, 50], "Should detect moderate CR"
    
    def test_cosmic_ray_mask_shape(self):
        """Test that cosmic ray mask has correct shape."""
        data = np.random.normal(1000, 10, (200, 150))
        ccd = CCDData(data, unit=u.electron)
        
        cr_mask = detect_cosmic_rays(ccd)
        
        assert cr_mask.shape == data.shape
        assert cr_mask.dtype == bool
    
    def test_no_false_positives_on_clean_data(self):
        """Test that clean data doesn't trigger false CR detections."""
        # Smooth data with no cosmic rays
        data = np.ones((100, 100)) * 1000
        data += np.random.normal(0, 50, (100, 100))  # Modest noise
        
        ccd = CCDData(data, unit=u.electron)
        
        cr_mask = detect_cosmic_rays(ccd, sigma_clip=5.0)
        
        # Should have very few false positives (<1%)
        false_positive_rate = cr_mask.sum() / cr_mask.size
        assert false_positive_rate < 0.01
    
    def test_cosmic_ray_parameters(self):
        """Test different cosmic ray detection parameters."""
        data = np.ones((100, 100)) * 1000
        data[50, 50] = 10000  # Moderate cosmic ray
        
        ccd = CCDData(data, unit=u.electron)
        
        # Strict threshold (high sigma) - might miss CR
        cr_mask_strict = detect_cosmic_rays(ccd, sigma_clip=10.0)
        
        # Lenient threshold (low sigma) - should detect CR
        cr_mask_lenient = detect_cosmic_rays(ccd, sigma_clip=3.0)
        
        # Lenient should detect more (or equal) CRs
        assert cr_mask_lenient.sum() >= cr_mask_strict.sum()
    
    def test_cosmic_ray_iteration_convergence(self):
        """Test that cosmic ray detection converges."""
        data = np.ones((100, 100)) * 1000
        
        # Add multiple cosmic rays
        for i in range(10):
            x, y = np.random.randint(10, 90, 2)
            data[x, y] = np.random.uniform(10000, 50000)
        
        ccd = CCDData(data, unit=u.electron)
        
        # With max_iterations, should converge
        cr_mask = detect_cosmic_rays(ccd, max_iterations=5)
        
        # Should detect cosmic rays
        assert cr_mask.sum() > 0
        assert cr_mask.sum() < 100  # But not flag everything


class TestCalibrationIntegration:
    """Test integration of calibration steps."""
    
    def test_full_calibration_workflow(self, temp_dir):
        """Test complete calibration workflow."""
        # Generate calibration frames
        bias_files = [temp_dir / f"bias_{i}.fits" for i in range(5)]
        flat_files = [temp_dir / f"flat_{i}.fits" for i in range(5)]
        
        for i, path in enumerate(bias_files):
            generate_bias_frame(path, seed=700+i)
        
        for i, path in enumerate(flat_files):
            generate_flat_frame(path, seed=800+i)
        
        # Create master calibrations
        bias_frames = [BiasFrame.from_fits(f) for f in bias_files]
        master_bias = create_master_bias(bias_frames)
        
        flat_frames = [FlatFrame.from_fits(f) for f in flat_files]
        master_flat = create_master_flat(flat_frames, master_bias)
        
        # Validate both calibrations
        assert master_bias.validate()
        assert master_flat.validate()
    
    def test_calibration_with_cosmic_rays(self, temp_dir):
        """Test that calibration handles cosmic rays properly."""
        # Flats can have cosmic rays - should be rejected by combination
        flat_path = temp_dir / "flat_with_cr.fits"
        generate_flat_frame(flat_path, seed=900)
        
        # Manually add cosmic ray to FITS data
        from astropy.io import fits
        with fits.open(flat_path, mode='update') as hdul:
            hdul[0].data[500, 500] = 60000  # CR
            hdul.flush()
        
        # Load and create master flat
        bias_path = temp_dir / "bias.fits"
        generate_bias_frame(bias_path, seed=901)
        
        bias_frame = BiasFrame.from_fits(bias_path)
        master_bias = create_master_bias([bias_frame])
        
        flat_frame = FlatFrame.from_fits(flat_path)
        master_flat = create_master_flat([flat_frame], master_bias)
        
        # Master flat should be created despite CR
        assert master_flat.data.shape[0] > 0
