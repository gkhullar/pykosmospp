"""
Pipeline orchestration for automated spectral reduction.

Per data-model.md §18: Coordinates end-to-end workflow from
raw FITS files to wavelength-calibrated 1D spectra.
"""

from pathlib import Path
from typing import Optional, Dict, List
import logging

from .models import (
    ObservationSet,
    MasterBias,
    MasterFlat,
    CalibrationSet,
    ReducedData,
    QualityMetrics
)
from .io.organize import discover_fits_files
from .io.config import load_config
from .calibration.combine import create_master_bias, create_master_flat
from .calibration.cosmic import detect_cosmic_rays
from .wavelength.identify import detect_arc_lines
from .wavelength.match import match_lines_to_catalog, load_linelist
from .wavelength.fit import fit_wavelength_solution
from .extraction.trace import detect_traces_cross_correlation
from .extraction.sky import estimate_sky_background
from .extraction.extract import extract_optimal
from .quality.validate import validate_calibrations, generate_validation_report
from .quality.plots import (
    setup_latex_plots,
    plot_2d_spectrum,
    plot_wavelength_residuals,
    plot_extraction_profile,
    plot_sky_subtraction
)


logger = logging.getLogger(__name__)


class CriticalPipelineError(Exception):
    """
    Critical error that halts pipeline execution.
    
    Per FR-018: Raised for missing calibrations, invalid FITS headers,
    or other unrecoverable errors.
    """
    pass


class QualityWarning(Warning):
    """
    Quality warning that allows pipeline to continue.
    
    Per FR-018: Issued for poor wavelength RMS, low SNR,
    or other quality concerns.
    """
    pass


class PipelineRunner:
    """
    Orchestrates end-to-end spectral reduction workflow.
    
    Per data-model.md §18: Coordinates calibration, wavelength
    calibration, spectral extraction, and quality assessment.
    
    Attributes
    ----------
    input_dir : Path
        Input directory containing raw FITS files
    output_dir : Path
        Output directory for reduced data products
    config : Dict
        Pipeline configuration parameters
    mode : str
        Execution mode ('batch' or 'interactive')
    """
    
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        config: Optional[Dict] = None,
        mode: str = 'batch'
    ):
        """
        Initialize pipeline runner.
        
        Parameters
        ----------
        input_dir : Path
            Input directory
        output_dir : Path
            Output directory
        config : Dict, optional
            Configuration (loads from default if None)
        mode : str
            'batch' or 'interactive'
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.mode = mode
        
        # Load configuration
        if config is None:
            config_path = Path(__file__).parent.parent / 'config' / 'kosmos_defaults.yaml'
            self.config = load_config(config_path)
        else:
            self.config = config
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True, parents=True)
        (self.output_dir / 'calibrations').mkdir(exist_ok=True)
        (self.output_dir / 'reduced_2d').mkdir(exist_ok=True)
        (self.output_dir / 'spectra_1d').mkdir(exist_ok=True)
        (self.output_dir / 'quality_reports').mkdir(exist_ok=True)
        (self.output_dir / 'diagnostic_plots').mkdir(exist_ok=True)
        
        # Configure plotting
        setup_latex_plots()
        
        logger.info(f"Pipeline initialized: {input_dir} -> {output_dir}")
        logger.info(f"Mode: {mode}")
    
    def run(self) -> List[ReducedData]:
        """
        Execute complete reduction workflow.
        
        Per tasks.md T051: Orchestrates:
        1. FITS file discovery
        2. Observation set creation
        3. Master bias/flat creation
        4. Arc lamp wavelength calibration
        5. Science frame processing
        6. Quality assessment
        7. Diagnostic plotting
        
        Returns
        -------
        List[ReducedData]
            List of reduced data products
        
        Raises
        ------
        CriticalPipelineError
            If calibrations missing or invalid
        """
        logger.info("=" * 60)
        logger.info("STARTING PIPELINE EXECUTION")
        logger.info("=" * 60)
        
        reduced_data_list = []
        
        try:
            # Step 1: Discover FITS files
            logger.info("Step 1: Discovering FITS files...")
            frames = discover_fits_files(self.input_dir)
            logger.info(f"  Found {len(frames)} FITS files")
            
            # Step 2: Create observation set
            logger.info("Step 2: Creating observation set...")
            obs_set = ObservationSet.from_directory(self.input_dir)
            logger.info(f"  Bias: {len(obs_set.bias_frames)}")
            logger.info(f"  Flat: {len(obs_set.flat_frames)}")
            logger.info(f"  Arc: {len(obs_set.arc_frames)}")
            logger.info(f"  Science: {len(obs_set.science_frames)}")
            
            # Step 3: Create master bias
            logger.info("Step 3: Creating master bias...")
            if len(obs_set.bias_frames) == 0:
                raise CriticalPipelineError("No bias frames found")
            
            master_bias = create_master_bias(
                obs_set.bias_frames,
                method='median'
            )
            logger.info(f"  Master bias created from {len(obs_set.bias_frames)} frames")
            
            # Step 4: Create master flat
            logger.info("Step 4: Creating master flat...")
            if len(obs_set.flat_frames) == 0:
                raise CriticalPipelineError("No flat frames found")
            
            master_flat = create_master_flat(
                obs_set.flat_frames,
                master_bias,
                method='median'
            )
            logger.info(f"  Master flat created from {len(obs_set.flat_frames)} frames")
            
            # Step 4a: Validate calibrations
            logger.info("Step 4a: Validating calibrations...")
            validation_results = validate_calibrations(master_bias, master_flat)
            
            if not validation_results['overall_valid']:
                logger.warning("Calibration validation failed:")
                logger.warning(generate_validation_report(validation_results))
                raise CriticalPipelineError("Calibration validation failed")
            
            logger.info("  Calibrations validated successfully")
            
            # Create calibration set
            calib_set = CalibrationSet(master_bias=master_bias, master_flat=master_flat)
            
            # Step 5: Process arc frames for wavelength calibration
            logger.info("Step 5: Processing arc frames...")
            wavelength_solutions = {}
            
            for arc_frame in obs_set.arc_frames:
                logger.info(f"  Processing {arc_frame.file_path.name}...")
                
                # Apply calibrations
                calibrated_arc = calib_set.apply_to_frame(arc_frame)
                
                # Detect arc lines
                arc_lines = detect_arc_lines(
                    calibrated_arc.data,
                    min_snr=self.config.get('arc_line_min_snr', 5.0)
                )
                logger.info(f"    Detected {len(arc_lines)} arc lines")
                
                # Load line list
                linelist_path = Path(__file__).parent.parent / 'resources' / 'pykosmos_reference' / 'HeNeAr_linelist.txt'
                catalog = load_linelist(linelist_path)
                
                # Match lines
                matched_lines = match_lines_to_catalog(
                    arc_lines,
                    catalog,
                    initial_dispersion=self.config.get('initial_dispersion', 1.0)
                )
                logger.info(f"    Matched {len(matched_lines)} lines")
                
                # Fit wavelength solution
                solution = fit_wavelength_solution(
                    matched_lines,
                    arc_frame=arc_frame,
                    max_order=self.config.get('wavelength_max_order', 7)
                )
                logger.info(f"    Wavelength solution: order={solution.order}, RMS={solution.rms_residual:.4f} Å")
                
                # Validate solution
                if not solution.validate():
                    logger.warning(f"    Wavelength solution failed validation")
                    import warnings
                    warnings.warn(
                        f"Poor wavelength solution: RMS={solution.rms_residual:.4f}",
                        QualityWarning
                    )
                
                wavelength_solutions[arc_frame.file_path.stem] = solution
            
            if len(wavelength_solutions) == 0:
                raise CriticalPipelineError("No valid wavelength solutions")
            
            # Use first solution for all science frames (could be more sophisticated)
            default_solution = list(wavelength_solutions.values())[0]
            
            # Step 6: Process science frames
            logger.info("Step 6: Processing science frames...")
            
            for science_frame in obs_set.science_frames:
                logger.info(f"  Processing {science_frame.file_path.name}...")
                
                processing_log = []
                processing_log.append(f"Source: {science_frame.file_path.name}")
                
                # Apply calibrations with uncertainty propagation
                calibrated_science = calib_set.apply_to_frame(science_frame, propagate_uncertainty=True)
                processing_log.append("Applied bias and flat calibrations with uncertainty propagation")
                
                # Detect cosmic rays
                from astropy.nddata import CCDData
                cr_mask = detect_cosmic_rays(
                    CCDData(calibrated_science.data, unit=calibrated_science.unit),
                    readnoise=self.config['detector']['readnoise'],
                    gain=self.config['detector']['gain']
                )
                cosmic_ray_fraction = cr_mask.sum() / cr_mask.size
                logger.info(f"    Cosmic rays: {cosmic_ray_fraction:.3f}")
                processing_log.append(f"Detected cosmic rays: {cosmic_ray_fraction:.3f}")
                
                # Create Spectrum2D with proper variance from calibrated data
                from .models import Spectrum2D
                
                # Extract variance from calibrated CCDData
                if calibrated_science.uncertainty is not None:
                    variance = calibrated_science.uncertainty.array ** 2
                else:
                    # Fallback to Poisson approximation if uncertainty not computed
                    variance = np.maximum(calibrated_science.data, 0) * science_frame.gain / science_frame.gain**2
                
                spectrum_2d = Spectrum2D(
                    data=calibrated_science.data,
                    variance=variance,
                    source_frame=science_frame,
                    cosmic_ray_mask=cr_mask
                )
                
                # Detect traces
                traces = detect_traces_cross_correlation(
                    spectrum_2d.data,
                    expected_fwhm=self.config.get('expected_fwhm', 4.0),
                    max_traces=self.config.get('max_traces', 5)
                )
                logger.info(f"    Detected {len(traces)} traces")
                processing_log.append(f"Detected {len(traces)} traces")
                
                if len(traces) == 0:
                    logger.warning(f"    No traces detected, skipping")
                    continue
                
                # Estimate and subtract sky
                sky_background = estimate_sky_background(
                    spectrum_2d.data,
                    traces,
                    sky_buffer=self.config.get('sky_buffer', 30)
                )
                spectrum_2d.data -= sky_background
                processing_log.append("Subtracted sky background")
                
                # Extract spectra
                spectra_1d = []
                for i, trace in enumerate(traces):
                    logger.info(f"    Extracting trace {i+1}/{len(traces)}...")
                    
                    # Fit spatial profile
                    trace.fit_profile(spectrum_2d.data)
                    
                    # Extract optimally
                    spectrum_1d = extract_optimal(
                        spectrum_2d.data,
                        spectrum_2d.variance,
                        trace
                    )
                    
                    # Apply wavelength solution
                    from .wavelength.apply import apply_wavelength_to_spectrum
                    spectrum_1d = apply_wavelength_to_spectrum(
                        spectrum_1d,
                        default_solution
                    )
                    
                    # Add metadata
                    spectrum_1d.meta['wavelength_rms'] = default_solution.rms_residual
                    spectrum_1d.meta['trace_id'] = i
                    
                    spectra_1d.append(spectrum_1d)
                    processing_log.append(f"Extracted trace {i+1}: SNR={trace.snr_estimate:.1f}")
                
                # Step 7: Compute quality metrics
                logger.info("    Computing quality metrics...")
                quality_metrics = QualityMetrics()
                quality_metrics.compute(spectra_1d[0], spectrum_2d)
                logger.info(f"    Grade: {quality_metrics.overall_grade}, SNR: {quality_metrics.median_snr:.1f}")
                processing_log.append(f"Quality: {quality_metrics.overall_grade}")
                
                # Step 8: Generate diagnostic plots
                logger.info("    Generating diagnostic plots...")
                base_name = science_frame.file_path.stem
                diagnostic_plots = {}
                
                # 2D spectrum
                plot_path = self.output_dir / 'diagnostic_plots' / f"{base_name}_2d.png"
                plot_2d_spectrum(spectrum_2d, plot_path, title=base_name)
                diagnostic_plots['2d_spectrum'] = plot_path
                
                # Wavelength residuals
                plot_path = self.output_dir / 'diagnostic_plots' / f"{base_name}_wavelength.png"
                plot_wavelength_residuals(default_solution, plot_path)
                diagnostic_plots['wavelength'] = plot_path
                
                # Extraction profile
                plot_path = self.output_dir / 'diagnostic_plots' / f"{base_name}_profile.png"
                plot_extraction_profile(traces[0], spectrum_2d, plot_path)
                diagnostic_plots['profile'] = plot_path
                
                # Sky subtraction
                plot_path = self.output_dir / 'diagnostic_plots' / f"{base_name}_sky.png"
                plot_sky_subtraction(spectrum_2d, sky_background, plot_path)
                diagnostic_plots['sky'] = plot_path
                
                processing_log.append("Generated diagnostic plots")
                
                # Step 9: Create ReducedData
                reduced_data = ReducedData(
                    source_frame=science_frame,
                    spectrum_2d=spectrum_2d,
                    spectra_1d=spectra_1d,
                    diagnostic_plots=diagnostic_plots,
                    processing_log=processing_log,
                    quality_metrics=quality_metrics
                )
                
                # Step 10: Save to disk
                logger.info("    Saving reduced data...")
                reduced_data.save_to_disk(self.output_dir)
                
                reduced_data_list.append(reduced_data)
                logger.info(f"  ✓ {science_frame.file_path.name} complete")
            
            logger.info("=" * 60)
            logger.info(f"PIPELINE COMPLETE: {len(reduced_data_list)} spectra reduced")
            logger.info("=" * 60)
            
            return reduced_data_list
        
        except CriticalPipelineError as e:
            logger.error(f"CRITICAL ERROR: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise CriticalPipelineError(f"Pipeline failed: {e}") from e
