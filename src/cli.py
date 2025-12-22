"""
Command-line interface for KOSMOS spectral reduction pipeline.

Per contracts/cli-spec.yaml: Provides kosmos-reduce command
with calibrate subcommand and comprehensive error handling.
"""

import argparse
import sys
import logging
from pathlib import Path

from .pipeline import PipelineRunner, CriticalPipelineError, QualityWarning
from .io.config import load_config
from .calibration.combine import create_master_bias, create_master_flat
from .models import ObservationSet


# Exit codes per contracts/cli-spec.yaml
EXIT_SUCCESS = 0
EXIT_MISSING_CALIBRATIONS = 1
EXIT_INVALID_INPUT = 2
EXIT_WAVELENGTH_FAILED = 3
EXIT_NO_TRACES = 4
EXIT_USER_CANCELED = 5


def setup_logging(verbose: bool = False, log_file: Path = None):
    """
    Configure logging for the pipeline.
    
    Parameters
    ----------
    verbose : bool
        Enable verbose (DEBUG) logging
    log_file : Path, optional
        Log file path
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def cmd_calibrate(args):
    """
    Generate master calibrations only.
    
    Per contracts/cli-spec.yaml: Creates master bias and flat,
    saves to output_dir/calibrations/
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
    
    Returns
    -------
    int
        Exit code
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Creating master calibrations...")
        
        # Discover frames
        obs_set = ObservationSet.from_directory(args.input_dir)
        
        # Check for required frames
        if len(obs_set.bias_frames) == 0:
            logger.error("No bias frames found")
            return EXIT_MISSING_CALIBRATIONS
        
        if len(obs_set.flat_frames) == 0:
            logger.error("No flat frames found")
            return EXIT_MISSING_CALIBRATIONS
        
        # Create master bias
        logger.info(f"Creating master bias from {len(obs_set.bias_frames)} frames...")
        master_bias = create_master_bias(obs_set.bias_frames, method='median')
        
        # Save bias
        output_dir = Path(args.output_dir)
        calib_dir = output_dir / 'calibrations'
        calib_dir.mkdir(exist_ok=True, parents=True)
        
        from .io.fits import write_fits_with_provenance
        from astropy.nddata import CCDData
        import astropy.units as u
        
        bias_ccd = CCDData(master_bias.data, unit=u.adu)
        bias_path = calib_dir / 'master_bias.fits'
        write_fits_with_provenance(bias_ccd, bias_path, {'NCOMBINE': len(obs_set.bias_frames)})
        logger.info(f"  Saved: {bias_path}")
        
        # Create master flat
        logger.info(f"Creating master flat from {len(obs_set.flat_frames)} frames...")
        master_flat = create_master_flat(obs_set.flat_frames, master_bias, method='median')
        
        # Save flat
        flat_ccd = CCDData(master_flat.data, unit=u.adu)
        flat_path = calib_dir / 'master_flat.fits'
        write_fits_with_provenance(flat_ccd, flat_path, {'NCOMBINE': len(obs_set.flat_frames)})
        logger.info(f"  Saved: {flat_path}")
        
        logger.info("✓ Calibrations complete")
        return EXIT_SUCCESS
    
    except Exception as e:
        logger.error(f"Calibration failed: {e}", exc_info=True)
        return EXIT_INVALID_INPUT


def cmd_reduce(args):
    """
    Execute full reduction pipeline.
    
    Per contracts/cli-spec.yaml: Runs end-to-end workflow from
    raw FITS to wavelength-calibrated 1D spectra.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
    
    Returns
    -------
    int
        Exit code
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        if args.config:
            config = load_config(args.config)
        else:
            config_path = Path(__file__).parent.parent / 'config' / 'kosmos_defaults.yaml'
            config = load_config(config_path)
        
        # Override config with command-line options
        if args.max_traces:
            config['max_traces'] = args.max_traces
        
        # Create pipeline runner
        runner = PipelineRunner(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            config=config,
            mode=args.mode
        )
        
        # Validate only mode
        if args.validate_only:
            logger.info("Validation mode: checking inputs only")
            obs_set = ObservationSet.from_directory(args.input_dir)
            
            if not obs_set.validate_completeness():
                logger.error("Incomplete observation set")
                return EXIT_MISSING_CALIBRATIONS
            
            logger.info("✓ Validation passed")
            return EXIT_SUCCESS
        
        # Run pipeline
        reduced_data_list = runner.run()
        
        logger.info(f"✓ Successfully reduced {len(reduced_data_list)} spectra")
        return EXIT_SUCCESS
    
    except CriticalPipelineError as e:
        logger.error(f"Pipeline error: {e}")
        
        # Determine appropriate exit code
        error_msg = str(e).lower()
        if 'calibration' in error_msg or 'missing' in error_msg:
            return EXIT_MISSING_CALIBRATIONS
        elif 'wavelength' in error_msg:
            return EXIT_WAVELENGTH_FAILED
        elif 'trace' in error_msg:
            return EXIT_NO_TRACES
        else:
            return EXIT_INVALID_INPUT
    
    except KeyboardInterrupt:
        logger.info("Pipeline canceled by user")
        return EXIT_USER_CANCELED
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return EXIT_INVALID_INPUT


def main():
    """
    Main entry point for kosmos-reduce CLI.
    
    Per contracts/cli-spec.yaml: Provides main command with
    calibrate subcommand and comprehensive options.
    """
    parser = argparse.ArgumentParser(
        prog='kosmos-reduce',
        description='KOSMOS Spectroscopy Reduction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Full reduction
  kosmos-reduce /path/to/data --output-dir ./reduced
  
  # Generate calibrations only
  kosmos-reduce calibrate /path/to/data --output-dir ./calibrations
  
  # Validate inputs
  kosmos-reduce /path/to/data --validate-only
  
  # Custom configuration
  kosmos-reduce /path/to/data --config my_config.yaml --max-traces 3
        '''
    )
    
    # Global arguments
    parser.add_argument(
        'input_dir',
        type=Path,
        help='Input directory containing raw FITS files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./reduced'),
        help='Output directory (default: ./reduced)'
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='Configuration file (default: config/kosmos_defaults.yaml)'
    )
    parser.add_argument(
        '--mode',
        choices=['batch', 'interactive'],
        default='batch',
        help='Execution mode (default: batch)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--log-file',
        type=Path,
        help='Log file path'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Validate inputs without processing'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output files'
    )
    parser.add_argument(
        '--max-traces',
        type=int,
        help='Maximum number of traces to extract'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # calibrate subcommand
    calibrate_parser = subparsers.add_parser(
        'calibrate',
        help='Generate master calibrations only'
    )
    calibrate_parser.add_argument(
        'input_dir',
        type=Path,
        help='Input directory containing raw FITS files'
    )
    calibrate_parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./calibrations'),
        help='Output directory (default: ./calibrations)'
    )
    calibrate_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    calibrate_parser.add_argument(
        '--log-file',
        type=Path,
        help='Log file path'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose, log_file=args.log_file)
    
    # Execute command
    if args.command == 'calibrate':
        return cmd_calibrate(args)
    else:
        return cmd_reduce(args)


if __name__ == '__main__':
    sys.exit(main())
