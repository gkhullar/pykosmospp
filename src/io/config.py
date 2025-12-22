"""
Configuration loading and validation for KOSMOS pipeline.

Handles YAML configuration files with PipelineConfig class.
Per tasks.md T019 and data-model.md §15.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class PipelineConfig:
    """
    Pipeline configuration loaded from YAML file.
    
    Per data-model.md §15: Encapsulates all pipeline parameters
    with validation and default value handling.
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize configuration from dictionary.
        
        Parameters
        ----------
        config_dict : dict
            Configuration parameters (typically from YAML)
        """
        self.config = config_dict
        self._validate()
        
    @classmethod
    def from_yaml(cls, filepath: Path) -> 'PipelineConfig':
        """
        Load configuration from YAML file.
        
        Parameters
        ----------
        filepath : Path
            Path to YAML configuration file
            
        Returns
        -------
        PipelineConfig
            Loaded configuration object
            
        Raises
        ------
        FileNotFoundError
            If config file doesn't exist
        ValueError
            If YAML is invalid
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {filepath}: {e}")
        
        return cls(config_dict)
    
    def _validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns
        -------
        bool
            True if configuration valid
            
        Raises
        ------
        ValueError
            If required parameters missing or invalid
        """
        required_sections = ['detector', 'calibration', 'trace_detection', 
                           'extraction', 'wavelength', 'quality']
        
        missing = [sec for sec in required_sections if sec not in self.config]
        if missing:
            raise ValueError(f"Missing required config sections: {missing}")
        
        # Validate detector parameters
        detector = self.config['detector']
        required_detector = ['gain', 'readnoise', 'saturate']
        missing_detector = [p for p in required_detector if p not in detector]
        if missing_detector:
            raise ValueError(f"Missing detector parameters: {missing_detector}")
        
        return True
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key path.
        
        Parameters
        ----------
        key_path : str
            Dot-separated path to config value (e.g., 'detector.gain')
        default : any, optional
            Default value if key not found
            
        Returns
        -------
        any
            Configuration value
            
        Examples
        --------
        >>> config.get('detector.gain')
        1.4
        >>> config.get('trace_detection.min_snr', default=3.0)
        3.0
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to config sections."""
        return self.config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self.config


def load_config(filepath: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file as dictionary.
    
    Simple wrapper for loading YAML configs without full validation.
    
    Parameters
    ----------
    filepath : Path
        Path to YAML configuration file
    
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)
