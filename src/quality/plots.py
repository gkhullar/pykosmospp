"""
Diagnostic plotting functions with LaTeX formatting.

Per research.md §10, uses matplotlib with LaTeX rendering
for publication-quality figures.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def setup_latex_plots():
    """
    Configure matplotlib for LaTeX rendering.
    
    Per research.md §10: Uses text.usetex=True for consistent
    formatting with scientific publications.
    """
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14
    })


def plot_2d_spectrum(
    spectrum_2d,
    output_path: Path,
    title: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> Figure:
    """
    Plot 2D spectrum with log-scale colormap.
    
    Parameters
    ----------
    spectrum_2d : Spectrum2D
        2D spectrum to plot
    output_path : Path
        Output file path
    title : str, optional
        Plot title
    vmin, vmax : float, optional
        Color scale limits
    
    Returns
    -------
    Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use log scale for display
    data = spectrum_2d.data.copy()
    data[data <= 0] = 1e-10  # Avoid log(0)
    
    if vmin is None:
        vmin = np.percentile(data, 5)
    if vmax is None:
        vmax = np.percentile(data, 99)
    
    im = ax.imshow(
        np.log10(data),
        aspect='auto',
        origin='lower',
        cmap='viridis',
        vmin=np.log10(vmin),
        vmax=np.log10(vmax)
    )
    
    ax.set_xlabel(r'Spectral Pixel')
    ax.set_ylabel(r'Spatial Pixel')
    if title:
        ax.set_title(title)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$\\log_{10}$(Flux) [e$^-$]')
    
    # Overlay traces if available
    if spectrum_2d.traces:
        for trace in spectrum_2d.traces:
            ax.plot(
                trace.spectral_pixels,
                trace.spatial_positions,
                'r-',
                linewidth=1,
                alpha=0.7
            )
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_wavelength_residuals(
    wavelength_solution,
    output_path: Path,
    title: Optional[str] = None
) -> Figure:
    """
    Plot wavelength solution residuals.
    
    Parameters
    ----------
    wavelength_solution : WavelengthSolution
        Wavelength solution with residuals
    output_path : Path
        Output file path
    title : str, optional
        Plot title
    
    Returns
    -------
    Figure
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Get line identifications from metadata
    if hasattr(wavelength_solution, 'identified_lines'):
        pixels = wavelength_solution.identified_lines['pixel']
        wavelengths = wavelength_solution.identified_lines['wavelength']
        
        # Compute fitted wavelengths
        fitted = wavelength_solution.wavelength(pixels)
        residuals = wavelengths - fitted
        
        # Top panel: Wavelength vs pixel
        ax1.scatter(pixels, wavelengths, s=30, alpha=0.7, label='Identified Lines')
        pixel_range = np.arange(wavelength_solution.pixel_range[0],
                                 wavelength_solution.pixel_range[1])
        ax1.plot(pixel_range, wavelength_solution.wavelength(pixel_range),
                 'r-', linewidth=1, label='Fit')
        ax1.set_xlabel(r'Pixel')
        ax1.set_ylabel(r'Wavelength [\AA]')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom panel: Residuals
        ax2.scatter(pixels, residuals, s=30, alpha=0.7)
        ax2.axhline(0, color='r', linestyle='--', linewidth=1)
        ax2.axhline(wavelength_solution.rms_residual, color='orange',
                    linestyle=':', linewidth=1, label=f'RMS = {wavelength_solution.rms_residual:.3f} \AA')
        ax2.axhline(-wavelength_solution.rms_residual, color='orange',
                    linestyle=':', linewidth=1)
        ax2.set_xlabel(r'Pixel')
        ax2.set_ylabel(r'Residual [\AA]')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_extraction_profile(
    trace,
    spectrum_2d,
    output_path: Path,
    spectral_slice: int = None,
    title: Optional[str] = None
) -> Figure:
    """
    Plot spatial profile fit vs data.
    
    Parameters
    ----------
    trace : Trace
        Trace with fitted spatial profile
    spectrum_2d : Spectrum2D
        2D spectrum
    output_path : Path
        Output file path
    spectral_slice : int, optional
        Spectral pixel to plot (default: middle)
    title : str, optional
        Plot title
    
    Returns
    -------
    Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Select spectral slice
    if spectral_slice is None:
        spectral_slice = spectrum_2d.data.shape[1] // 2
    
    # Get trace position at this slice
    trace_idx = np.argmin(np.abs(trace.spectral_pixels - spectral_slice))
    trace_center = trace.spatial_positions[trace_idx]
    
    # Extract spatial profile
    spatial_data = spectrum_2d.data[:, spectral_slice]
    spatial_pixels = np.arange(len(spatial_data))
    
    # Plot data
    ax.plot(spatial_pixels, spatial_data, 'ko', markersize=3,
            alpha=0.6, label='Data')
    
    # Plot fitted profile if available
    if trace.spatial_profile is not None:
        profile_model = trace.spatial_profile.evaluate(spatial_pixels)
        ax.plot(spatial_pixels, profile_model, 'r-', linewidth=2,
                label=f'Fit ({trace.spatial_profile.profile_type})')
        
        # Mark center
        ax.axvline(trace_center, color='blue', linestyle='--',
                   linewidth=1, alpha=0.7, label='Trace Center')
    
    ax.set_xlabel(r'Spatial Pixel')
    ax.set_ylabel(r'Flux [e$^-$]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if title:
        ax.set_title(title)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_sky_subtraction(
    spectrum_2d,
    sky_background: np.ndarray,
    output_path: Path,
    title: Optional[str] = None
) -> Figure:
    """
    Plot sky background estimation with highlighted sky regions.
    
    Parameters
    ----------
    spectrum_2d : Spectrum2D
        2D spectrum
    sky_background : np.ndarray
        Estimated sky background
    output_path : Path
        Output file path
    title : str, optional
        Plot title
    
    Returns
    -------
    Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    # Top: Original data
    im1 = axes[0].imshow(
        np.log10(np.maximum(spectrum_2d.data, 1e-10)),
        aspect='auto',
        origin='lower',
        cmap='viridis'
    )
    axes[0].set_ylabel(r'Spatial Pixel')
    axes[0].set_title(r'Original')
    plt.colorbar(im1, ax=axes[0], label=r'$\\log_{10}$(Flux)')
    
    # Overlay traces to show sky regions
    if spectrum_2d.traces:
        for trace in spectrum_2d.traces:
            axes[0].plot(trace.spectral_pixels, trace.spatial_positions,
                         'r-', linewidth=1, alpha=0.7)
    
    # Middle: Sky model
    im2 = axes[1].imshow(
        np.log10(np.maximum(sky_background, 1e-10)),
        aspect='auto',
        origin='lower',
        cmap='viridis'
    )
    axes[1].set_ylabel(r'Spatial Pixel')
    axes[1].set_title(r'Sky Background Model')
    plt.colorbar(im2, ax=axes[1], label=r'$\\log_{10}$(Flux)')
    
    # Bottom: Residual
    residual = spectrum_2d.data - sky_background
    im3 = axes[2].imshow(
        residual,
        aspect='auto',
        origin='lower',
        cmap='RdBu_r',
        vmin=-np.percentile(np.abs(residual), 95),
        vmax=np.percentile(np.abs(residual), 95)
    )
    axes[2].set_xlabel(r'Spectral Pixel')
    axes[2].set_ylabel(r'Spatial Pixel')
    axes[2].set_title(r'Sky-Subtracted')
    plt.colorbar(im3, ax=axes[2], label=r'Flux [e$^-$]')
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig
