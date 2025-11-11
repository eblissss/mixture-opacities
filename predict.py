import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast
import pandas as pd
import numpy as np
from tqdm import tqdm

from model import OpacityNet

CONFIG = {
    "model_path": "model.pth",
    "batch_size": 1024,
    "num_workers": 4,
    "show_progress": True,
}

def preprocess_inputs(X: np.ndarray) -> np.ndarray:
    """
    Apply log-transform to temperature and density
    KEEPS ORIGINAL COLUMN ORDER
    
    Args:
        X: Input array [N, 5] = [mH, mHe, mAl, Temperature, Density]
    
    Returns:
        Preprocessed array [N, 5] = [mH, mHe, mAl, log10(Temperature), log10(Density)]
    """
    # Extract columns (original order)
    mix_H = X[:, 0]
    mix_He = X[:, 1]
    mix_Al = X[:, 2]
    temperature = X[:, 3]
    density = X[:, 4]
    
    # Apply log transform (add epsilon to avoid log(0))
    epsilon = 1e-10
    log_temperature = np.log10(np.maximum(temperature, epsilon))
    log_density = np.log10(np.maximum(density, epsilon))
    
    # Keep original order: [mH, mHe, mAl, log_temp, log_density]
    X_preprocessed = np.column_stack([mix_H, mix_He, mix_Al, log_temperature, log_density])
    
    return X_preprocessed


def predict(X: np.ndarray) -> np.ndarray:
    # Setup
    model_path = CONFIG["model_path"]
    batch_size = CONFIG["batch_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Prediction time! | Model: {model_path} | Device: {device}")

    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    scaler_X = checkpoint["scaler_X"]
    scaler_y = checkpoint["scaler_y"]
    predict_log = checkpoint.get("predict_log", False)

    model = OpacityNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded! Input samples: {len(X)}")
    print(f"Predict log-opacities: {predict_log}")

    # Validate input shape
    if X.shape[1] != 5:
        raise ValueError(f"Input has {X.shape[1]} features, but model expects 5!")

    # Preprocess inputs (log-transform) BEFORE standardization
    print("Preprocessing inputs (log-transform)...")
    X_preprocessed = preprocess_inputs(X)

    # Standardize
    X_scaled = scaler_X.transform(X)
    X_tensor = torch.FloatTensor(X_scaled)

    # Create DataLoader
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True if device.type == "cuda" else False,
    )

    # Predictions time
    print("\nNow predicting...")

    predictions = []
    with torch.no_grad():
        iterator = tqdm(loader, desc="Batches") if CONFIG["show_progress"] else loader

        for (batch,) in iterator:
            batch = batch.to(device, non_blocking=True)

            if device.type == "cuda":
                with autocast():
                    outputs = model(batch)
            else:
                outputs = model(batch)

            predictions.append(outputs.cpu())

    # Process predictions
    predictions = torch.cat(predictions).numpy()
    predictions = scaler_y.inverse_transform(predictions)

    # If model predicted log-opacities, convert back to real scale
    if predict_log:
        print("Converting from log-space to real opacities...")
        predictions = np.power(10, predictions)

    print(f"Predictions complete!\n")

    # Create DataFrame with inputs and outputs
    input_df = pd.DataFrame(
        X, columns=["Mix_H", "Mix_He", "Mix_Al", "Temperature", "Density"]
    )
    output_df = pd.DataFrame(
        predictions,
        columns=[f"pred_{col}" for col in ["Rosseland_opacity", "Planck_opacity"]],
    )
    result_df = pd.concat([input_df, output_df], axis=1)
    result_df.to_csv("predictions.csv", index=False)

    print(f"Saved to: predictions.csv\n")

    # Show some
    print("First 5 predictions:")
    print(predictions[:5])
    print(f"Total: {len(predictions)} predictions")

    return predictions


if __name__ == "__main__":
    # Random numbers for now
    # samples = np.random.randn(1000000, 5)
    # preds = predict(samples)

    n = 10

    # Load data from CSV
    df = pd.read_csv("opacity_data.csv")
    sample_df = df.sample(n=n, random_state=42)
    
    # Extract inputs (first 5 columns) and ground truth (last 2 columns)
    X = sample_df.iloc[:, :5].values
    y_true = sample_df.iloc[:, 5:].values
    
    # Get predictions
    y_pred = predict(X)
    
    # Print results
    print("Comparison: Inputs | Ground Truth | Predictions")
    for i in range(n):
        print(f"\nSample {i+1}:")
        print(f"  Inputs: H={X[i,0]:.6f}, He={X[i,1]:.6f}, Al={X[i,2]:.6f}, T={X[i,3]:.4e}, ρ={X[i,4]:.4e}")
        print(f"  Ground Truth:  Rosseland={y_true[i,0]:.4e}, Planck={y_true[i,1]:.4e}")
        print(f"  Predicted:     Rosseland={y_pred[i,0]:.4e}, Planck={y_pred[i,1]:.4e}")
