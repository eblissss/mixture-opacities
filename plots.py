import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

# --- CONFIGURATION ---
USE_TEX = False  # Set to True if LaTeX is available
FIGURE_SIZE = (7, 6)
DPI = 300

# File Paths
FILE_TRAIN_EPOCHS = "training_log_epochs.csv"
FILE_TRAIN_STEPS = "training_log_steps.csv"
FILE_VAL_RESULTS = "validation_results.csv"

def configure_plotting_style():
    """Sets matplotlib parameters for publication-quality figures."""
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("ticks")
    
    params = {
        "font.family": "serif",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "lines.linewidth": 2,
        "figure.figsize": FIGURE_SIZE,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
    }
    
    if USE_TEX:
        params.update({
            "text.usetex": True, 
            "font.serif": ["Computer Modern Roman"]
        })
        
    plt.rcParams.update(params)

def save_figure(filename: str):
    """Helper to finalize and save the current figure."""
    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_learning_rate(csv_path: str, filename: str):
    """Plots the learning rate schedule over global steps."""
    if not os.path.exists(csv_path):
        print(f"Skipping {filename}: {csv_path} not found.")
        return
    
    df = pd.read_csv(csv_path)
    if "Train/LearningRate" not in df.columns:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(df['step'], df['Train/LearningRate'], color='#8e44ad', label='Learning Rate')
    
    plt.xlabel('Global Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule (OneCycleLR)')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    save_figure(filename)

def plot_batch_loss(csv_path: str, filename: str):
    """Plots raw batch loss with a smoothing window."""
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    if "Train/BatchLoss" not in df.columns:
        return

    plt.figure(figsize=(8, 5))
    
    # Plot raw data transparently
    plt.plot(df['step'], df['Train/BatchLoss'], alpha=0.15, color='gray', 
             linewidth=1, label='Raw Batch Loss')
    
    # Calculate rolling average (Window = 1% of total steps, min 10)
    window = max(int(len(df) * 0.01), 10)
    rolling_mean = df['Train/BatchLoss'].rolling(window=window).mean()
    
    plt.plot(df['step'], rolling_mean, color='#2980b9', 
             label=f'Moving Avg (w={window})')
    
    plt.yscale('log')
    plt.xlabel('Global Step')
    plt.ylabel('Loss')
    plt.title('Training Batch Loss')
    plt.legend()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    save_figure(filename)

def plot_loss_curve(csv_path: str, filename: str):
    """Plots Epoch-level Training vs Validation Loss."""
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df['Epoch/TrainLoss'], label='Training Loss', marker='o', markersize=3)
    plt.plot(df['epoch'], df['Epoch/ValLoss'], label='Validation Loss', marker='s', markersize=3)
    
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Smooth L1 Loss')
    plt.title('Training Dynamics')
    plt.legend()
    
    save_figure(filename)
    

def plot_parity(df: pd.DataFrame, target_col: str, pred_col: str, 
                title_prefix: str, filename: str):
    """Generates a log-log hexbin parity plot."""
    plt.figure(figsize=(7, 6))
    
    # Filter strict positives for log scaling
    mask = (df[target_col] > 1e-9) & (df[pred_col] > 1e-9)
    y_true = df.loc[mask, target_col]
    y_pred = df.loc[mask, pred_col]
    
    # Hexbin density plot
    hb = plt.hexbin(
        y_true, y_pred, 
        gridsize=80, 
        bins='log', 
        xscale='log', 
        yscale='log', 
        cmap='inferno', 
        mincnt=1
    )
    
    # Identity line (Ideal prediction)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'w--', lw=2, alpha=0.8)
    
    # Styling
    cb = plt.colorbar(hb)
    cb.set_label(r'Count (log scale)')
    plt.xlabel(f'Ground Truth {title_prefix}')
    plt.ylabel(f'Predicted {title_prefix}')
    plt.title(f'{title_prefix} Parity Plot')
    
    # Add MAPE Annotation
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    plt.text(0.05, 0.95, f"MAPE: {mape:.2f}%", 
             transform=plt.gca().transAxes, 
             color='white', fontweight='bold',
             bbox=dict(facecolor='black', alpha=0.6, boxstyle='round'))
    
    save_figure(filename)

def plot_error_heatmap(df: pd.DataFrame, target_col: str, pred_col: str, 
                       title_prefix: str, filename: str):
    """
    Plots Mean Absolute Percentage Error (MAPE) in Temperature-Density space.
    """
    # 1. Calculate MAPE per point
    df_plot = df.copy()
    df_plot['MAPE'] = np.abs((df[target_col] - df[pred_col]) / df[target_col]) * 100
    
    # 2. Generate Bins AND Labels (Geometric centers for log plot)
    num_bins = 25
    
    # Temperature Bins
    t_edges = np.logspace(np.log10(df['Temperature'].min()), np.log10(df['Temperature'].max()), num_bins)
    t_centers = np.sqrt(t_edges[:-1] * t_edges[1:]) # Geometric mean
    
    # Density Bins
    d_edges = np.logspace(np.log10(df['Density'].min()), np.log10(df['Density'].max()), num_bins)
    d_centers = np.sqrt(d_edges[:-1] * d_edges[1:]) # Geometric mean
    
    # 3. Cut using explicit float labels (Fixes "AttributeError: 'str' object has no attribute 'mid'")
    df_plot['T_bin'] = pd.cut(df_plot['Temperature'], bins=t_edges, labels=t_centers)
    df_plot['D_bin'] = pd.cut(df_plot['Density'], bins=d_edges, labels=d_centers)
    
    # 4. Pivot to Grid
    heatmap_data = df_plot.pivot_table(
        index='D_bin', 
        columns='T_bin', 
        values='MAPE', 
        aggfunc='mean'
    )
    
    # 5. Format Axis Labels (Now using the float centers directly)
    y_vals = heatmap_data.index
    x_vals = heatmap_data.columns
    
    # Convert floats to formatted strings for the axis ticks
    x_labels = [f"{float(x):.1e}" for x in x_vals]
    y_labels = [f"{float(y):.1e}" for y in y_vals]
    
    # 6. Plot
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        heatmap_data, 
        cmap="viridis", 
        norm=LogNorm(vmin=0.1, vmax=50),
        cbar_kws={'label': 'Mean Absolute Percentage Error (%)'},
        xticklabels=4, 
        yticklabels=4
    )
    
    # Beautify ticks (Log formatting: 10^x)
    new_x = [f"$10^{{{np.log10(float(l)):.1f}}}$" for l in x_labels[::4]]
    new_y = [f"$10^{{{np.log10(float(l)):.1f}}}$" for l in y_labels[::4]]
    
    ax.set_xticklabels(new_x, rotation=0)
    ax.set_yticklabels(new_y, rotation=0)
    
    plt.xlabel('Temperature (K)')
    plt.ylabel('Density (g/cm$^3$)')
    plt.title(f'{title_prefix} Error Landscape')
    plt.gca().invert_yaxis()
    
    save_figure(filename)

def plot_error_distribution(df: pd.DataFrame, filename: str):
    """Histogram of relative errors for both opacities."""
    plt.figure(figsize=(8, 5))
    
    ross_err = (df['Pred_Rosseland'] - df['Rosseland_opacity']) / df['Rosseland_opacity']
    planck_err = (df['Pred_Planck'] - df['Planck_opacity']) / df['Planck_opacity']
    
    bins = np.linspace(-0.5, 0.5, 100)
    
    plt.hist(ross_err, bins=bins, alpha=0.6, label='Rosseland', density=True)
    plt.hist(planck_err, bins=bins, alpha=0.6, label='Planck', density=True)
    
    plt.xlabel(r'Relative Error $\frac{y_{pred} - y_{true}}{y_{true}}$')
    plt.ylabel('Probability Density')
    plt.title('Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_figure(filename)

def plot_composition_sensitivity(df: pd.DataFrame, filename: str):
    """Violin plot showing max error distributed by dominant element."""
    df_viol = df.copy()
    
    # Compute max error between the two outputs
    err_r = np.abs((df['Pred_Rosseland'] - df['Rosseland_opacity']) / df['Rosseland_opacity'])
    err_p = np.abs((df['Pred_Planck'] - df['Planck_opacity']) / df['Planck_opacity'])
    df_viol['Max_Error'] = np.maximum(err_r, err_p) * 100
    
    # Classify dominant element
    conditions = [
        (df['Mix_H'] > 0.5),
        (df['Mix_He'] > 0.5),
        (df['Mix_Al'] > 0.5)
    ]
    choices = ['Hydrogen-Rich', 'Helium-Rich', 'Metal-Rich (Al)']
    df_viol['Composition'] = np.select(conditions, choices, default='Mixed')
    
    plt.figure(figsize=(8, 6))
    sns.violinplot(
        x='Composition', 
        y='Max_Error', 
        data=df_viol, 
        hue='Composition', 
        palette="muted", 
        cut=0, 
        legend=False
    )
    
    plt.yscale('log')
    plt.ylim(0.01, 1000)
    plt.ylabel('Max Absolute Percentage Error (%)')
    plt.title('Model Sensitivity to Element Composition')
    
    save_figure(filename)


if __name__ == "__main__":
    configure_plotting_style()
    print("Starting plot generation...")

    # 1. Training Metrics Plots
    if os.path.exists(FILE_TRAIN_EPOCHS):
        plot_loss_curve(FILE_TRAIN_EPOCHS, "fig0_loss_curve.pdf")
    else:
        print(f"Warning: {FILE_TRAIN_EPOCHS} not found.")

    if os.path.exists(FILE_TRAIN_STEPS):
        plot_learning_rate(FILE_TRAIN_STEPS, "fig0b_learning_rate.pdf")
        plot_batch_loss(FILE_TRAIN_STEPS, "fig0c_batch_loss.pdf")
    else:
        print(f"Warning: {FILE_TRAIN_STEPS} not found.")

    # 2. Validation Analysis Plots
    if os.path.exists(FILE_VAL_RESULTS):
        print(f"Loading validation data from {FILE_VAL_RESULTS}...")
        df_val = pd.read_csv(FILE_VAL_RESULTS)
        
        # Parity Plots
        plot_parity(df_val, 'Rosseland_opacity', 'Pred_Rosseland', 'Rosseland', 'fig1_parity_rosseland.pdf')
        plot_parity(df_val, 'Planck_opacity', 'Pred_Planck', 'Planck', 'fig2_parity_planck.pdf')
        
        # Error Heatmaps
        plot_error_heatmap(df_val, 'Rosseland_opacity', 'Pred_Rosseland', 'Rosseland', 'fig3_heatmap_rosseland.pdf')
        plot_error_heatmap(df_val, 'Planck_opacity', 'Pred_Planck', 'Planck', 'fig4_heatmap_planck.pdf')
        
        # Distribution & Sensitivity
        plot_error_distribution(df_val, 'fig5_error_dist.pdf')
        plot_composition_sensitivity(df_val, 'fig6_sensitivity.pdf')
        
        print("\nSuccess! All plots generated.")
    else:
        print(f"\nError: {FILE_VAL_RESULTS} not found.")
        print("Please run 'evaluate_validation.py' to generate the results first.")
