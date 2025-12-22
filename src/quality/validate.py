"""
Quality validation for calibration frames.

Per FR-010, validates that calibrations meet acceptance criteria.
"""

from typing import Dict, List
import numpy as np
from astropy.stats import sigma_clipped_stats

from ..models import MasterBias, MasterFlat


def validate_calibrations(
    master_bias: MasterBias,
    master_flat: MasterFlat
) -> Dict[str, bool]:
    """
    Validate calibration frames meet quality criteria.
    
    Per FR-010:
    - Bias: level variation < 10 ADU
    - Flat: normalization in range 10k-50k ADU
    - Flat: saturation fraction < 0.01
    
    Parameters
    ----------
    master_bias : MasterBias
        Master bias frame
    master_flat : MasterFlat
        Master flat frame
    
    Returns
    -------
    Dict[str, bool]
        Validation results with 'bias_valid', 'flat_valid', 'overall_valid'
    """
    results = {}
    
    # Validate bias
    mean, median, std = sigma_clipped_stats(master_bias.data.data, sigma=3.0)
    bias_variation = std
    bias_valid = bool(bias_variation < 10.0)  # ADU
    
    results['bias_valid'] = bias_valid
    results['bias_variation'] = float(bias_variation)
    
    # Validate flat normalization
    flat_mean, flat_median, flat_std = sigma_clipped_stats(
        master_flat.data.data, sigma=3.0
    )
    flat_in_range = bool(10000 < flat_median < 50000)
    
    # Validate flat saturation
    saturation_threshold = 65535  # 16-bit
    saturated_pixels = np.sum(master_flat.data.data >= saturation_threshold)
    saturation_fraction = saturated_pixels / master_flat.data.data.size
    flat_not_saturated = bool(saturation_fraction < 0.01)
    
    flat_valid = flat_in_range and flat_not_saturated
    
    results['flat_valid'] = flat_valid
    results['flat_median'] = float(flat_median)
    results['flat_saturation_fraction'] = float(saturation_fraction)
    
    # Overall validation
    results['overall_valid'] = bias_valid and flat_valid
    
    return results


def generate_validation_report(results: Dict) -> str:
    """
    Generate formatted validation report.
    
    Parameters
    ----------
    results : Dict
        Validation results from validate_calibrations()
    
    Returns
    -------
    str
        Formatted report
    """
    report = []
    report.append("Calibration Validation Report")
    report.append("=" * 50)
    report.append("")
    
    # Bias validation
    bias_status = "PASS" if results['bias_valid'] else "FAIL"
    report.append(f"Master Bias: {bias_status}")
    report.append(f"  Variation: {results['bias_variation']:.2f} ADU")
    report.append(f"  Threshold: < 10.0 ADU")
    report.append("")
    
    # Flat validation
    flat_status = "PASS" if results['flat_valid'] else "FAIL"
    report.append(f"Master Flat: {flat_status}")
    report.append(f"  Median: {results['flat_median']:.0f} ADU")
    report.append(f"  Range: 10,000 - 50,000 ADU")
    report.append(f"  Saturation: {results['flat_saturation_fraction']:.4f}")
    report.append(f"  Threshold: < 0.01")
    report.append("")
    
    # Overall
    overall_status = "PASS" if results['overall_valid'] else "FAIL"
    report.append(f"Overall Status: {overall_status}")
    
    return "\n".join(report)
