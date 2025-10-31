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

    model = OpacityNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded! Input samples: {len(X)}")

    # Validate input shape
    if X.shape[1] != 5:
        raise ValueError(f"Input has {X.shape[1]} features, but model expects 5!")

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
    samples = np.random.randn(1000000, 5)
    preds = predict(samples)
