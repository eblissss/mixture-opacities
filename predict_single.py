import torch
import numpy as np
from model import OpacityNet

"""
Predict opacity for specific inputs.
"""

# Load model
device = torch.device("cpu")
checkpoint = torch.load("model.pth", map_location=device, weights_only=False)
scaler_X = checkpoint["scaler_X"]
scaler_y = checkpoint["scaler_y"]

model = OpacityNet().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def predict_opacity(mH, mHe, mAl, T, rho):
    """
    Predict Rosseland and Planck opacities for given inputs.
    
    Args:
        mH: Mass fraction of Hydrogen
        mHe: Mass fraction of Helium
        mAl: Mass fraction of Aluminum
        T: Temperature (keV)
        rho: Density (g/cm³)
    
    Returns:
        kappa_R: Rosseland opacity
        kappa_P: Planck opacity
    """
    # Preprocess
    epsilon = 1e-10
    X = np.array([[
        mH, mHe, mAl,
        np.log10(max(T, epsilon)),
        np.log10(max(rho, epsilon))
    ]])
    
    # Scale
    X_scaled = scaler_X.transform(X)
    X_tensor = torch.FloatTensor(X_scaled)
    
    # Predict
    with torch.no_grad():
        pred_scaled = model(X_tensor).numpy()
    
    # Inverse transform
    pred_log = scaler_y.inverse_transform(pred_scaled)
    pred = np.power(10, pred_log)
    
    return pred[0, 0], pred[0, 1]  # kappa_R, kappa_P


if __name__ == "__main__":
    # input
    mH = 0.4
    mHe = 0.4
    mAl = 0.2
    T = 1.00E+01   # keV
    rho = 1.00E-01  # g/cm³
    
    kappa_R, kappa_P = predict_opacity(mH, mHe, mAl, T, rho)
    
    print("=" * 50)
    print("OPACITY PREDICTION")
    print("=" * 50)
    print(f"\nInputs:")
    print(f"  mH  = {mH}")
    print(f"  mHe = {mHe}")
    print(f"  mAl = {mAl}")
    print(f"  T   = {T:.2e} keV")
    print(f"  rho = {rho:.2e} g/cm³")
    print(f"\nPredicted Opacities:")
    print(f"  Rosseland (κR) = {kappa_R:.4e}")
    print(f"  Planck (κP)    = {kappa_P:.4e}")
    print("=" * 50)
