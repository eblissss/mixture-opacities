import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
import seaborn as sns
import mpltern

# --- CONFIGURATION ---
USE_TEX = False  # Set True only if you have LaTeX installed on your OS
FIGURE_SIZE = (8, 6)
DPI = 600

# File Paths
FILE_TRAIN_EPOCHS = "training_log_epochs.csv"
FILE_VAL_RESULTS = "validation_results.csv"


def configure_plotting_style():
    # Sets a rigorous, high-quality publication style
    plt.rcParams.update(plt.rcParamsDefault)
    
    # Base seaborn style for nice colors/grids
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks")
    
    params = {
        "font.family": "serif",
        "axes.grid": True,
        "grid.alpha": 0.4,
        "grid.linestyle": ":",
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "axes.linewidth": 1.5,
        "xtick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.major.size": 6,
        "ytick.minor.size": 3,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True, 
        "ytick.right": True,
        "figure.figsize": FIGURE_SIZE,
        "savefig.bbox": "tight",
        "savefig.dpi": DPI,
    }
    
    if USE_TEX:
        params.update({
            "text.usetex": True, 
            "font.serif": ["Computer Modern Roman"]
        })
        
    plt.rcParams.update(params)


def save_figure(filename: str):
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved high-res figure: {filename}")
    plt.close()


def plot_loss_curve(csv_path: str, filename: str):
    # Plots training dynamics with a focus on the final convergence
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    
    plt.figure()
    
    plt.plot(df['epoch'], df['Epoch/TrainLoss'], label='Training', 
             color='#7f8c8d', linestyle='--', alpha=0.8, linewidth=1.5)
    plt.plot(df['epoch'], df['Epoch/ValLoss'], label='Validation', 
             color='#e74c3c', linestyle='-', alpha=1.0, linewidth=2.5)
    
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Smooth L1 Loss')
    plt.title('Training Convergence')
    
    plt.legend(frameon=True, framealpha=0.9, edgecolor='gray')
    plt.grid(True, which="minor", axis='y', alpha=0.2)
    
    save_figure(filename)


def plot_parity(df: pd.DataFrame, target_col: str, pred_col: str, 
                title_prefix: str, filename: str):
    plt.figure()
    
    # Filter strictly positive values for log-log plot
    mask = (df[target_col] > 1e-20) & (df[pred_col] > 1e-20)
    y_true = df.loc[mask, target_col]
    y_pred = df.loc[mask, pred_col]
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    
    # High-Res Hexbin
    hb = plt.hexbin(
        y_true, y_pred, 
        gridsize=150, 
        bins='log', 
        xscale='log', 
        yscale='log', 
        cmap='inferno', 
        mincnt=1,
        linewidths=0
    )
    
    # Reference Line
    plt.plot([min_val, max_val], [min_val, max_val], 'w--', lw=2, alpha=0.7)
    
    # Styling
    cb = plt.colorbar(hb)
    cb.set_label(r'Density ($\log_{10}$ counts)')
    
    plt.xlabel(f'TOPS {title_prefix} Opacity')
    plt.ylabel(f'Predicted {title_prefix} Opacity')
    plt.title(f'{title_prefix} Parity Comparison')
    
    save_figure(filename)


def plot_error_heatmap(df: pd.DataFrame, target_col: str, pred_col: str, 
                       title_prefix: str, filename: str):
    # Uses pcolormesh to generate a heatmap of error across T-rho plane
    df['MAPE'] = np.abs((df[target_col] - df[pred_col]) / df[target_col]) * 100
    
    # Define Grid
    n_bins = 32
    t_edges = np.logspace(np.log10(df['Temperature'].min()), np.log10(df['Temperature'].max()), n_bins)
    d_edges = np.logspace(np.log10(df['Density'].min()), np.log10(df['Density'].max()), n_bins)
    
    # Aggregate Data
    H_sum, _, _ = np.histogram2d(df['Temperature'], df['Density'], bins=[t_edges, d_edges], weights=df['MAPE'])
    H_count, _, _ = np.histogram2d(df['Temperature'], df['Density'], bins=[t_edges, d_edges])
    
    # Calculate Mean
    with np.errstate(invalid='ignore'):
        H_mean = H_sum / H_count
        H_mean = np.ma.masked_invalid(H_mean)
    
    plt.figure()
    
    # Plot using pcolormesh
    pcm = plt.pcolormesh(
        t_edges, d_edges, H_mean.T, 
        cmap='viridis', 
        norm=LogNorm(vmin=0.1, vmax=50),
        shading='flat'
    )
    
    plt.xscale('log')
    plt.yscale('log')
    
    cbar = plt.colorbar(pcm, extend='max')
    cbar.set_label('Mean Absolute Percentage Error (%)')
    
    plt.xlabel('Temperature (keV)')
    plt.ylabel(r'Density (g/cm$^3$)')
    plt.title(f'{title_prefix} Error Landscape')
    
    save_figure(filename)


def plot_ternary_composition(df: pd.DataFrame, filename: str):
    # Generates a Ternary Hexbin plot
    
    # Calculate Max Error per point
    err_r = np.abs((df['Pred_Rosseland'] - df['Rosseland_opacity']) / df['Rosseland_opacity'])
    err_p = np.abs((df['Pred_Planck'] - df['Planck_opacity']) / df['Planck_opacity'])
    max_err = np.maximum(err_r, err_p) * 100
    
    # Setup Ternary Axis
    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(projection='ternary')
    
    # Ternary Hexbin
    hb = ax.hexbin(
        df['Mix_H'], df['Mix_He'], df['Mix_Al'], 
        C=max_err, 
        reduce_C_function=np.mean, 
        gridsize=20, 
        cmap='magma_r', 
        norm=LogNorm(vmin=1, vmax=50),
        edgecolors='face'
    )
    
    # Labels & Grid
    ax.set_tlabel('H Fraction')
    ax.set_llabel('He Fraction')
    ax.set_rlabel('Al Fraction')
    
    ax.grid(True, linestyle='--', linewidth=1.0, alpha=0.5)
    
    # Colorbar
    cbar = plt.colorbar(hb, ax=ax, shrink=0.8)
    cbar.set_label('Mean Max Error (%)')
    
    plt.title("Error Distribution by Composition")
    save_figure(filename)


if __name__ == "__main__":
    configure_plotting_style()
    print("Starting high-res plot generation...")

    if os.path.exists(FILE_TRAIN_EPOCHS):
        plot_loss_curve(FILE_TRAIN_EPOCHS, "fig0_loss.pdf")
        
    if os.path.exists(FILE_VAL_RESULTS):
        print(f"Processing validation data from {FILE_VAL_RESULTS}...")
        df = pd.read_csv(FILE_VAL_RESULTS)
        
        # Parity Plots
        plot_parity(df, 'Rosseland_opacity', 'Pred_Rosseland', 'Rosseland', 'fig1_parity_ross.pdf')
        plot_parity(df, 'Planck_opacity', 'Pred_Planck', 'Planck', 'fig2_parity_planck.pdf')
        
        # Error Heatmaps
        plot_error_heatmap(df, 'Rosseland_opacity', 'Pred_Rosseland', 'Rosseland', 'fig3_heatmap_ross.pdf')
        plot_error_heatmap(df, 'Planck_opacity', 'Pred_Planck', 'Planck', 'fig4_heatmap_planck.pdf')
        
        # Ternary Plot
        plot_ternary_composition(df, 'fig5_ternary_error.pdf')
        
        print("Done. All figures saved.")
    else:
        print("Validation file not found.")
