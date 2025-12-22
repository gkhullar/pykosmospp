"""
Logging and error handling for KOSMOS pipeline.

Provides structured logging with verbosity control and custom error classes
for tiered error handling per FR-018.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class CriticalPipelineError(Exception):
    """
    Critical failure requiring immediate halt.
    
    Per FR-018: Raised for corrupt FITS files, wrong instrument configuration,
    missing required calibrations. Includes stage, criterion, and suggested remediation.
    """
    
    def __init__(self, stage: str, criterion: str, remediation: str):
        """
        Initialize critical error with diagnostic information.
        
        Parameters
        ----------
        stage : str
            Pipeline stage where failure occurred
        criterion : str
            Validation criterion that was violated
        remediation : str
            Suggested action to fix the issue
        """
        self.stage = stage
        self.criterion = criterion
        self.remediation = remediation
        
        message = (
            f"CRITICAL ERROR in {stage}:\n"
            f"  Criterion violated: {criterion}\n"
            f"  Suggested fix: {remediation}"
        )
        super().__init__(message)


class QualityWarning(UserWarning):
    """
    Quality issue that produces flagged output but doesn't halt pipeline.
    
    Per FR-018: Issued for low SNR, partial arc line coverage, cosmic ray
    contamination. Processing continues with quality flags in logs.
    """
    
    def __init__(self, stage: str, issue: str, quality_flag: str):
        """
        Initialize quality warning.
        
        Parameters
        ----------
        stage : str
            Pipeline stage where issue detected
        issue : str
            Description of quality issue
        quality_flag : str
            Flag to add to output metadata
        """
        self.stage = stage
        self.issue = issue
        self.quality_flag = quality_flag
        
        message = f"QUALITY WARNING in {stage}: {issue} (flagged: {quality_flag})"
        super().__init__(message)


def setup_logger(name: str = 'kosmos-reduce', 
                log_file: Optional[Path] = None,
                verbose: bool = True,
                quiet: bool = False) -> logging.Logger:
    """
    Configure pipeline logger with console and file handlers.
    
    Per contracts/cli-spec.yaml: Support --verbose/--quiet modes and --log-file option.
    
    Parameters
    ----------
    name : str, optional
        Logger name (default: 'kosmos-reduce')
    log_file : Path, optional
        Path to log file. If None, logs only to console.
    verbose : bool, optional
        Enable verbose (DEBUG) logging (default: True)
    quiet : bool, optional
        Suppress console output (default: False)
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    # Console handler (unless quiet)
    if not quiet:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # File handler (if log file specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_processing_step(logger: logging.Logger, stage: str, 
                       input_files: list, parameters: dict) -> None:
    """
    Log processing step with inputs and parameters.
    
    Per FR-011: Log all processing steps with timestamps, input files, and parameters.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    stage : str
        Processing stage name
    input_files : list
        List of input file paths
    parameters : dict
        Parameters used for this stage
    """
    logger.info(f"Starting stage: {stage}")
    logger.debug(f"  Input files: {[str(f) for f in input_files]}")
    logger.debug(f"  Parameters: {parameters}")


def log_validation_result(logger: logging.Logger, stage: str, 
                          passed: bool, metrics: dict) -> None:
    """
    Log validation check result.
    
    Per FR-010: Log validation results at each stage.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    stage : str
        Processing stage name
    passed : bool
        Whether validation passed
    metrics : dict
        Validation metrics
    """
    status = "PASSED" if passed else "FAILED"
    logger.info(f"Validation {status}: {stage}")
    
    for metric, value in metrics.items():
        logger.debug(f"  {metric}: {value}")
