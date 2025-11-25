import torch
import numpy as np
import time
from model import OpacityNet

# Load model
device = torch.device("cpu")  # Use CPU for fair comparison with table lookups
checkpoint = torch.load("model.pth", map_location=device, weights_only=False)
model = OpacityNet().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

scaler_X = checkpoint["scaler_X"]

def preprocess_single(X):
    """Preprocess a single input"""
    epsilon = 1e-10
    X_processed = np.array([
        X[0], X[1], X[2],  # mH, mHe, mAl
        np.log10(max(X[3], epsilon)),  # log10(Temperature)
        np.log10(max(X[4], epsilon))   # log10(Density)
    ]).reshape(1, -1)
    return scaler_X.transform(X_processed)

# Sample input: [mH, mHe, mAl, Temperature, Density]
sample_input = np.array([0.3, 0.5, 0.2, 1.0, 0.01])

print("=" * 60)
print("INFERENCE TIMING BENCHMARK")
print("=" * 60)

# Warmup
with torch.no_grad():
    for _ in range(100):
        X = preprocess_single(sample_input)
        X_tensor = torch.FloatTensor(X).to(device)
        _ = model(X_tensor)

# ====================
# Single sample timing
# ====================
n_single = 10000
times_single = []

with torch.no_grad():
    for _ in range(n_single):
        start = time.perf_counter()
        
        X = preprocess_single(sample_input)
        X_tensor = torch.FloatTensor(X).to(device)
        output = model(X_tensor)
        
        end = time.perf_counter()
        times_single.append(end - start)

avg_single = np.mean(times_single) * 1e6  # Convert to microseconds
std_single = np.std(times_single) * 1e6

print(f"\n[Single Sample Inference]")
print(f"  Average time: {avg_single:.2f} µs ({avg_single/1000:.3f} ms)")
print(f"  Std dev:      {std_single:.2f} µs")
print(f"  Throughput:   {1e6/avg_single:.0f} predictions/second")

# ====================
# Batch timing
# ====================
print(f"\n[Batch Inference]")

for batch_size in [1, 10, 100, 1000, 10000, 100000]:
    # Create batch of random inputs
    batch_inputs = np.random.rand(batch_size, 5)
    batch_inputs[:, 3] *= 40  # Temperature range
    batch_inputs[:, 4] = 10 ** (np.random.uniform(-5, 2, batch_size))  # Density range
    
    # Preprocess
    epsilon = 1e-10
    batch_processed = np.column_stack([
        batch_inputs[:, 0],
        batch_inputs[:, 1], 
        batch_inputs[:, 2],
        np.log10(np.maximum(batch_inputs[:, 3], epsilon)),
        np.log10(np.maximum(batch_inputs[:, 4], epsilon))
    ])
    batch_scaled = scaler_X.transform(batch_processed)
    batch_tensor = torch.FloatTensor(batch_scaled).to(device)
    
    # Time it (average over multiple runs)
    n_runs = 100 if batch_size <= 1000 else 10
    times = []
    
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            output = model(batch_tensor)
            end = time.perf_counter()
            times.append(end - start)
    
    avg_time = np.mean(times) * 1000  # ms
    per_sample = np.mean(times) / batch_size * 1e6  # µs
    throughput = batch_size / np.mean(times)
    
    print(f"  Batch {batch_size:>6}: {avg_time:>8.3f} ms total | {per_sample:>6.2f} µs/sample | {throughput:>10,.0f} samples/sec")

print("\n" + "=" * 60)
print("COMPARISON CONTEXT")
print("=" * 60)
print("""
Typical lookup table interpolation: 1-10 µs per lookup
Your neural network single sample:  see above

For radiation hydrodynamics:
- If NN is within 2-3x of table lookup, it's competitive
- Batch processing can be much faster than repeated lookups
""")
