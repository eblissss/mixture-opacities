import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, random_split
from sklearn.preprocessing import StandardScaler

from model import OpacityNet

"""
Evaluate MAPE on validation set only (data the model never saw during training).
This gives a true measure of generalization performance.
"""

CONFIG = {
    "data_path": "opacity_data.csv",
    "model_path": "model.pth",
    "val_split": 0.1,  # Must match training.py
    "random_seed": 42,  # Must match training.py
}


def preprocess_inputs(X: np.ndarray) -> np.ndarray:
    """Apply log-transform to temperature and density"""
    epsilon = 1e-10
    X_processed = np.column_stack([
        X[:, 0],  # mH
        X[:, 1],  # mHe
        X[:, 2],  # mAl
        np.log10(np.maximum(X[:, 3], epsilon)),  # log10(Temperature)
        np.log10(np.maximum(X[:, 4], epsilon))   # log10(Density)
    ])
    return X_processed


def main():
    print("=" * 60)
    print("VALIDATION SET EVALUATION (Unseen Data Only)")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from: {CONFIG['data_path']}")
    df = pd.read_csv(CONFIG["data_path"])
    print(f"Total samples: {len(df)}")
    
    # Extract features and targets
    X = df.iloc[:, :5].values  # [mH, mHe, mAl, T, rho]
    y = df.iloc[:, 5:].values  # [Rosseland, Planck]
    
    # Preprocess (log-transform)
    X_processed = preprocess_inputs(X)
    
    # Apply log to targets (same as training)
    epsilon = 1e-10
    y_log = np.log10(np.maximum(y, epsilon))
    
    # Create dataset and split EXACTLY as in training
    X_tensor = torch.FloatTensor(X_processed)
    y_tensor = torch.FloatTensor(y_log)
    dataset = TensorDataset(X_tensor, y_tensor)
    
    val_size = int(len(dataset) * CONFIG["val_split"])
    train_size = len(dataset) - val_size
    
    # Use same seed as training to get the SAME split
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(CONFIG["random_seed"])
    )
    
    print(f"Train samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Extract validation indices
    val_indices = val_ds.indices
    
    # Get validation data (original scale, before any preprocessing)
    X_val = X[val_indices]
    y_val_true = y[val_indices]  # Original scale opacities
    
    print(f"\nEvaluating on {len(val_indices)} validation samples...")
    
    # Load model
    device = torch.device("cpu")
    checkpoint = torch.load(CONFIG["model_path"], map_location=device, weights_only=False)
    scaler_X = checkpoint["scaler_X"]
    scaler_y = checkpoint["scaler_y"]
    
    model = OpacityNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Preprocess validation inputs
    X_val_processed = preprocess_inputs(X_val)
    X_val_scaled = scaler_X.transform(X_val_processed)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    
    # Predict
    with torch.no_grad():
        predictions_scaled = model(X_val_tensor).numpy()
    
    # Inverse transform predictions
    predictions_log = scaler_y.inverse_transform(predictions_scaled)
    y_val_pred = np.power(10, predictions_log)  # Convert from log-space
    
    # Save results for plotting
    print("\nSaving validation results to validation_results.csv...")
    results_df = pd.DataFrame(X_val, columns=["Mix_H", "Mix_He", "Mix_Al", "Temperature", "Density"])
    results_df["Rosseland_opacity"] = y_val_true[:, 0]
    results_df["Planck_opacity"] = y_val_true[:, 1]
    results_df["Pred_Rosseland"] = y_val_pred[:, 0]
    results_df["Pred_Planck"] = y_val_pred[:, 1]
    results_df.to_csv("validation_results.csv", index=False)

    # Calculate MAPE
    mape = np.abs((y_val_true - y_val_pred) / y_val_true) * 100
    mape_rosseland = np.mean(mape[:, 0])
    mape_planck = np.mean(mape[:, 1])
    
    # Also calculate median (less sensitive to outliers)
    median_rosseland = np.median(mape[:, 0])
    median_planck = np.median(mape[:, 1])
    
    # Calculate percentiles
    p90_rosseland = np.percentile(mape[:, 0], 90)
    p90_planck = np.percentile(mape[:, 1], 90)
    p99_rosseland = np.percentile(mape[:, 0], 99)
    p99_planck = np.percentile(mape[:, 1], 99)
    
    print("\n" + "=" * 60)
    print("RESULTS (Validation Set Only - Unseen Data)")
    print("=" * 60)
    
    print(f"\n{'Metric':<25} {'Rosseland':<15} {'Planck':<15}")
    print("-" * 55)
    print(f"{'Mean MAPE:':<25} {mape_rosseland:<15.2f}% {mape_planck:<15.2f}%")
    print(f"{'Median MAPE:':<25} {median_rosseland:<15.2f}% {median_planck:<15.2f}%")
    print(f"{'90th Percentile:':<25} {p90_rosseland:<15.2f}% {p90_planck:<15.2f}%")
    print(f"{'99th Percentile:':<25} {p99_rosseland:<15.2f}% {p99_planck:<15.2f}%")
    
    # Show error distribution
    print("\n" + "=" * 60)
    print("ERROR DISTRIBUTION")
    print("=" * 60)
    
    bins = [0, 1, 2, 5, 10, 20, 50, 100, float('inf')]
    labels = ['<1%', '1-2%', '2-5%', '5-10%', '10-20%', '20-50%', '50-100%', '>100%']
    
    print(f"\n{'Error Range':<15} {'Rosseland':<20} {'Planck':<20}")
    print("-" * 55)
    
    for i in range(len(bins) - 1):
        ross_count = np.sum((mape[:, 0] >= bins[i]) & (mape[:, 0] < bins[i+1]))
        planck_count = np.sum((mape[:, 1] >= bins[i]) & (mape[:, 1] < bins[i+1]))
        ross_pct = ross_count / len(mape) * 100
        planck_pct = planck_count / len(mape) * 100
        print(f"{labels[i]:<15} {ross_count:>6} ({ross_pct:>5.1f}%)     {planck_count:>6} ({planck_pct:>5.1f}%)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
