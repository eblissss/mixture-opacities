import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import time

from model import OpacityNet

CONFIG = {
    "data_path": "opacity_data.csv",
    "val_split": 0.1,
    "batch_size": 1024,
    "epochs": 200,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "early_stopping_patience": 15,
    "num_workers": 8,
    "predict_log": True,
}


def preprocess_data(df):
    """
    Preprocess raw data: apply log transforms to features and targets

    Args:
        df: DataFrame with columns [Mix_H, Mix_He, Mix_Al, Temperature, Density,
                                     Rosseland_opacity, Planck_opacity]

    Returns:
        X: Preprocessed features [N, 5]
        y: Preprocessed targets [N, 2]
    """
    # Extract raw features
    mix_H = df["Mix_H"].values
    mix_He = df["Mix_He"].values
    mix_Al = df["Mix_Al"].values
    temperature = df["Temperature"].values
    density = df["Density"].values

    # Apply log transform to temperature and density (span many orders of magnitude)
    epsilon = 1e-10  # Avoid log(0)
    log_temperature = np.log10(np.maximum(temperature, epsilon))
    log_density = np.log10(np.maximum(density, epsilon))

    # order: [mH, mHe, mAl, log_temp, log_density]
    X = np.column_stack([mix_H, mix_He, mix_Al, log_temperature, log_density])

    # Extract targets
    rosseland = df["Rosseland_opacity"].values
    planck = df["Planck_opacity"].values

    if CONFIG["predict_log"]:
        # Apply log transform to targets
        log_rosseland = np.log10(np.maximum(rosseland, epsilon))
        log_planck = np.log10(np.maximum(planck, epsilon))
        y = np.column_stack([log_rosseland, log_planck])
        print("Targets: log10(opacities)")
    else:
        y = np.column_stack([rosseland, planck])
        print("Targets: raw opacities")

    return X, y


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    train_loss,
    val_loss,
    scaler_X,
    scaler_y,
    filepath,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "config": CONFIG,
            "predict_log": CONFIG["predict_log"],
        },
        filepath,
    )


def train():
    print(f"Training Setup Started!")
    torch.manual_seed(42)
    np.random.seed(42)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    use_amp = device_type == "cuda"
    pin_memory = device_type == "cuda"
    num_workers = 0 if device_type != "cuda" else CONFIG["num_workers"]

    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print(f"Loading data from: {CONFIG['data_path']}")
    df = pd.read_csv(CONFIG["data_path"])
    print(f"Raw data shape: {df.shape}")

    # Preprocess data (apply log transforms)
    print("Preprocessing data (applying log transforms)...")
    X, y = preprocess_data(df)
    print(f"Preprocessed shapes - Features: {X.shape}, Targets: {y.shape}")
    print(f"\nFeature ranges after preprocessing:")
    print(f"  Mix_H: [{X[:, 0].min():.3f}, {X[:, 0].max():.3f}]")
    print(f"  Mix_He: [{X[:, 1].min():.3f}, {X[:, 1].max():.3f}]")
    print(f"  Mix_Al: [{X[:, 2].min():.3f}, {X[:, 2].max():.3f}]")
    print(f"  log10(Temperature): [{X[:, 3].min():.2f}, {X[:, 3].max():.2f}]")
    print(f"  log10(Density): [{X[:, 4].min():.2f}, {X[:, 4].max():.2f}]")

    # Standardize data (now operating on log-transformed data)
    scaler_X = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y)
    X_scaled = torch.FloatTensor(scaler_X.transform(X))
    y_scaled = torch.FloatTensor(scaler_y.transform(y))

    # Train/Val split
    dataset = TensorDataset(X_scaled, y_scaled)
    val_size = int(len(dataset) * CONFIG["val_split"])
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    print(f"\nDataset split - Total: {len(df)}, Train: {train_size}, Val: {val_size}\n")

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    # Model
    model = OpacityNet(predict_log=CONFIG["predict_log"]).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameter count: {num_params:,}")
    print(f"Predicting log-opacities: {CONFIG['predict_log']}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        betas=(0.9, 0.999),
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=CONFIG["learning_rate"] * 7,
        steps_per_epoch=len(train_loader),
        epochs=CONFIG["epochs"],
        pct_start=0.45,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e3,
    )
    criterion = nn.SmoothL1Loss(beta=0.1)
    scaler = GradScaler(device_type) if use_amp else None

    # TensorBoard
    writer = SummaryWriter("runs")
    print(f"TensorBoard: tensorboard --logdir=runs\n")

    # Training state
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    start_time = time.time()

    print("Training Start!")
    for epoch in range(CONFIG["epochs"]):
        # Training section
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

        for _, (features, targets) in enumerate(pbar):
            features, targets = features.to(device), targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            if use_amp:
                with autocast(device_type):
                    outputs = model(features)
                    loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(features)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            train_loss += loss.item()

            # Logging
            writer.add_scalar("Train/BatchLoss", loss.item(), global_step)
            writer.add_scalar(
                "Train/LearningRate", scheduler.get_last_lr()[0], global_step
            )

            global_step += 1
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.6f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

        avg_train_loss = train_loss / len(train_loader)

        # Validation section
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)

                if use_amp:
                    with autocast(device_type):
                        outputs = model(features)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(features)
                    loss = criterion(outputs, targets)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # Epoch logging
        writer.add_scalar("Epoch/TrainLoss", avg_train_loss, epoch)
        writer.add_scalar("Epoch/ValLoss", avg_val_loss, epoch)
        writer.add_scalars(
            "Epoch/Losses", {"train": avg_train_loss, "val": avg_val_loss}, epoch
        )
        print(
            f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            best_path = "best.pth"
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                avg_train_loss,
                avg_val_loss,
                scaler_X,
                scaler_y,
                best_path,
            )
            print(f"New best model saved: {best_path} | Val loss: {avg_val_loss:.6f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= CONFIG["early_stopping_patience"]:
            print(f"Early stopping (Epoch {epoch+1})")
            break

    writer.close()
    elapsed = time.time() - start_time
    print(f"Training complete! Time: {elapsed/60:.1f} minutes")

    # Save final model
    final_checkpoint = torch.load("best.pth", weights_only=False)
    torch.save(
        {
            "model_state_dict": final_checkpoint["model_state_dict"],
            "scaler_X": final_checkpoint["scaler_X"],
            "scaler_y": final_checkpoint["scaler_y"],
            "predict_log": final_checkpoint["predict_log"],
        },
        "model.pth",
    )
    print(f"Final model saved to: model.pth\n")


if __name__ == "__main__":
    train()
