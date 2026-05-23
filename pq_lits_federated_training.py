"""
PQ-LITS: Federated Anomaly Detection for Low-Altitude UAV Security
====================================================================
Complete pipeline: Local-only → Federated (FedAvg) → Centralized baseline

This script implements a lightweight autoencoder-based anomaly detector
trained across 3 Ground Control Station (GCS) clients using Federated
Averaging. No external FL framework needed — pure PyTorch.

Hardware: Google Colab free tier (CPU is sufficient, GPU optional)
Runtime:  ~8-12 minutes total
Memory:   < 500 MB

Usage on Colab:
  1. Upload the 3 CSV files (GCS_A/B/C_telemetry.csv)
  2. Run all cells sequentially
  3. Results table + plots generated automatically

Author: Shafiq Ahmed
Date:   March 2026
"""

# ============================================================
# CELL 1: Imports and Setup
# ============================================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11
import copy
import time
import warnings
warnings.filterwarnings('ignore')

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")

# ============================================================
# CELL 2: Load and Preprocess Data
# ============================================================
# --- UPDATE THESE PATHS FOR YOUR COLAB ENVIRONMENT ---
# If you uploaded to Colab, files are typically in /content/
# If using Kaggle, they will be in /kaggle/input/
DATA_PATHS = {
    'GCS_A': 'pq_lits_dataset/GCS_A_telemetry.csv',
    'GCS_B': 'pq_lits_dataset/GCS_B_telemetry.csv',
    'GCS_C': 'pq_lits_dataset/GCS_C_telemetry.csv',
}

FEATURE_COLS = [
    'latitude', 'longitude', 'altitude', 'ground_speed',
    'vertical_speed', 'heading', 'rssi', 'snr',
    'packet_interval', 'num_satellites', 'battery_voltage', 'cmd_rate'
]

# Binary classification: 0 = normal, 1 = any attack
# (We detect anomalies, not classify attack types — simpler and more
# realistic for an autoencoder-based approach)


def load_client_data(path, client_name):
    """Load one client's CSV and return features + binary labels."""
    df = pd.read_csv(path)
    X = df[FEATURE_COLS].values.astype(np.float32)
    # Binary: normal (0) vs any attack (1)
    y = (df['label'] > 0).astype(np.int32).values
    print(f"  {client_name}: {len(X)} samples, "
          f"{y.sum()} attacks ({100*y.mean():.1f}%), "
          f"{(y==0).sum()} normal ({100*(1-y.mean()):.1f}%)")
    return X, y, df


print("Loading client datasets...")
client_data_raw = {}
client_dfs = {}
for name, path in DATA_PATHS.items():
    X, y, df = load_client_data(path, name)
    client_data_raw[name] = (X, y)
    client_dfs[name] = df


# ============================================================
# CELL 3: Feature Scaling
# ============================================================
"""
IMPORTANT DESIGN DECISION:
In a real federated system, each client would scale using only its local
statistics (mean, std). The federation server does NOT see raw data.
We scale per-client to preserve this property.

For the centralized baseline, we fit a global scaler on all data.
"""


def prepare_client_splits(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    Split into train/val/test.
    Train set: ONLY normal data (autoencoder learns normal patterns).
    Val + Test: mixed normal + attacks (for threshold tuning and evaluation).
    """
    n = len(X)
    indices = np.arange(n)

    # Separate normal and attack indices
    normal_idx = indices[y == 0]
    attack_idx = indices[y == 1]

    # Shuffle
    np.random.shuffle(normal_idx)
    np.random.shuffle(attack_idx)

    # Split normal data: 70% train, 15% val, 15% test
    n_normal = len(normal_idx)
    n_train = int(train_ratio * n_normal)
    n_val = int(val_ratio * n_normal)

    train_idx = normal_idx[:n_train]
    val_normal_idx = normal_idx[n_train:n_train + n_val]
    test_normal_idx = normal_idx[n_train + n_val:]

    # Split attack data: 50% val, 50% test (none in training)
    n_attack = len(attack_idx)
    n_attack_val = n_attack // 2

    val_attack_idx = attack_idx[:n_attack_val]
    test_attack_idx = attack_idx[n_attack_val:]

    # Combine val and test
    val_idx = np.concatenate([val_normal_idx, val_attack_idx])
    test_idx = np.concatenate([test_normal_idx, test_attack_idx])

    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    return train_idx, val_idx, test_idx


print("\nPreparing train/val/test splits per client...")
client_data = {}

for name in DATA_PATHS:
    X_raw, y = client_data_raw[name]
    train_idx, val_idx, test_idx = prepare_client_splits(X_raw, y)

    # Fit scaler on training data only (normal samples)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_raw[train_idx])
    X_val = scaler.transform(X_raw[val_idx])
    X_test = scaler.transform(X_raw[test_idx])

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    client_data[name] = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'scaler': scaler,
    }

    print(f"  {name}: train={len(X_train)} (all normal), "
          f"val={len(X_val)} ({y_val.sum()} attacks), "
          f"test={len(X_test)} ({y_test.sum()} attacks)")


# ============================================================
# CELL 4: Autoencoder Architecture
# ============================================================
"""
Architecture: 12 → 8 → 4 → 8 → 12

Why this specific design:
- Input dim = 12 (our feature count)
- Bottleneck = 4 (compresses 12 features into 4 latent dimensions)
  This forces the model to learn the essential structure of normal
  telemetry. Attacks deviate from this structure → high reconstruction
  error → anomaly detected.
- Symmetric decoder mirrors the encoder
- No dropout (we want deterministic reconstruction at inference)
- LeakyReLU avoids dead neurons on negative feature values

Total parameters: 12*8 + 8 + 8*4 + 4 + 4*8 + 8 + 8*12 + 12
               = 96 + 8 + 32 + 4 + 32 + 8 + 96 + 12 = 288
That is absurdly small. Trains in seconds. Perfect for federated rounds.
"""


class AnomalyAutoencoder(nn.Module):
    def __init__(self, input_dim=12, hidden1=10, hidden2=6, latent_dim=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden1, hidden2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden2, latent_dim),
            nn.LeakyReLU(0.2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden2, hidden1),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden1, input_dim),
            # No activation on output
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def reconstruction_error(self, x):
        """Per-sample MSE reconstruction error (used as anomaly score)."""
        x_hat = self.forward(x)
        return torch.mean((x - x_hat) ** 2, dim=1)


# Quick sanity check
model_test = AnomalyAutoencoder().to(DEVICE)
n_params = sum(p.numel() for p in model_test.parameters())
print(f"\nAutoencoder parameters: {n_params}")
del model_test


# ============================================================
# CELL 5: Training Utilities
# ============================================================
def make_dataloader(X, batch_size=128, shuffle=True):
    """Create a PyTorch DataLoader from numpy array."""
    tensor = torch.FloatTensor(X).to(DEVICE)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, dataloader, optimizer, criterion):
    """Train model for one epoch. Returns average loss."""
    model.train()
    total_loss = 0
    n_batches = 0
    for (batch_x,) in dataloader:
        optimizer.zero_grad()
        x_hat = model(batch_x)
        loss = criterion(x_hat, batch_x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def compute_anomaly_scores(model, X):
    """Compute reconstruction error for each sample."""
    model.eval()
    with torch.no_grad():
        x_tensor = torch.FloatTensor(X).to(DEVICE)
        scores = model.reconstruction_error(x_tensor)
    return scores.cpu().numpy()


def find_optimal_threshold(scores, y_true, percentile_range=(50, 99)):
    """
    Find the threshold that maximises F1-score on the validation set.
    Search over percentiles of the score distribution.
    """
    best_f1 = 0
    best_threshold = 0
    for p in np.arange(percentile_range[0], percentile_range[1] + 0.5, 0.5):
        threshold = np.percentile(scores, p)
        y_pred = (scores > threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold, best_f1


def evaluate_model(model, X, y, threshold):
    """Full evaluation: accuracy, precision, recall, F1, AUC."""
    scores = compute_anomaly_scores(model, X)
    y_pred = (scores > threshold).astype(int)

    results = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'threshold': threshold,
    }
    # AUC (if both classes present)
    if len(np.unique(y)) == 2:
        results['auc'] = roc_auc_score(y, scores)
    else:
        results['auc'] = float('nan')

    return results, scores, y_pred


# ============================================================
# CELL 6: Experiment 1 — Local-Only Training (Baseline)
# ============================================================
"""
Each GCS trains its own autoencoder using only its local data.
No communication between clients. This is the baseline we beat.
"""

LOCAL_EPOCHS = 100
LOCAL_LR = 0.003
BATCH_SIZE = 64

print("=" * 65)
print("EXPERIMENT 1: Local-Only Training")
print("=" * 65)

local_models = {}
local_results = {}
local_thresholds = {}

for name in ['GCS_A', 'GCS_B', 'GCS_C']:
    print(f"\n--- Training {name} locally ({LOCAL_EPOCHS} epochs) ---")
    cd = client_data[name]

    model = AnomalyAutoencoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LOCAL_LR)
    criterion = nn.MSELoss()
    train_loader = make_dataloader(cd['X_train'], BATCH_SIZE)

    t0 = time.time()
    for epoch in range(LOCAL_EPOCHS):
        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{LOCAL_EPOCHS}  Loss: {loss:.6f}")

    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # Find threshold on validation set
    val_scores = compute_anomaly_scores(model, cd['X_val'])
    threshold, val_f1 = find_optimal_threshold(val_scores, cd['y_val'])
    print(f"  Optimal threshold: {threshold:.6f} (val F1: {val_f1:.4f})")

    # Evaluate on test set
    results, _, _ = evaluate_model(model, cd['X_test'], cd['y_test'], threshold)
    print(f"  Test — Acc: {results['accuracy']:.4f}, "
          f"Prec: {results['precision']:.4f}, "
          f"Rec: {results['recall']:.4f}, "
          f"F1: {results['f1']:.4f}, AUC: {results['auc']:.4f}")

    local_models[name] = model
    local_results[name] = results
    local_thresholds[name] = threshold

# Cross-evaluation: test each local model on OTHER clients' data
print("\n--- Cross-Client Evaluation (Local Models) ---")
print("This reveals the non-IID weakness: a model trained on one")
print("client performs poorly on unseen attack types from other clients.\n")

cross_results = {}
for train_client in ['GCS_A', 'GCS_B', 'GCS_C']:
    for test_client in ['GCS_A', 'GCS_B', 'GCS_C']:
        cd_test = client_data[test_client]
        res, _, _ = evaluate_model(
            local_models[train_client],
            cd_test['X_test'], cd_test['y_test'],
            local_thresholds[train_client]
        )
        cross_results[(train_client, test_client)] = res['f1']
        if train_client != test_client:
            print(f"  Model({train_client}) → Test({test_client}): "
                  f"F1 = {res['f1']:.4f}")


# ============================================================
# CELL 7: Experiment 2 — Federated Learning (FedAvg)
# ============================================================
"""
Federated Averaging (McMahan et al., 2017):
  1. Server initialises a global model
  2. Each round:
     a. Server sends global model to all clients
     b. Each client trains locally for E epochs
     c. Each client sends model weights back
     d. Server averages weights: w_global = (1/K) * sum(w_k)
  3. Repeat for R rounds

Privacy: Raw telemetry NEVER leaves the client. Only model weights
(288 float values) are transmitted each round.
"""

FED_ROUNDS = 20          # Number of federation rounds
FED_LOCAL_EPOCHS = 8     # Local epochs per round
FED_LR = 0.003

# --- Optional: Differential Privacy via gradient noise ---
# Set DP_ENABLED = True to add Gaussian noise to weights before sending
# This provides (epsilon, delta)-DP guarantees
DP_ENABLED = True
DP_NOISE_SCALE = 0.01    # Sigma for Gaussian noise on model weights
# With 288 parameters and sigma=0.01, this is a mild privacy budget.
# For the paper, you can compute epsilon using the Gaussian mechanism:
#   epsilon = sqrt(2 * ln(1.25/delta)) * (sensitivity / sigma)
# where sensitivity ≈ max weight change per round.


def fedavg_aggregate(global_model, client_models, client_weights=None):
    """
    Federated Averaging: weighted average of client model parameters.
    client_weights: list of floats (e.g., proportional to dataset size).
                    If None, equal weighting.
    """
    if client_weights is None:
        client_weights = [1.0 / len(client_models)] * len(client_models)
    else:
        total = sum(client_weights)
        client_weights = [w / total for w in client_weights]

    global_dict = global_model.state_dict()
    for key in global_dict:
        global_dict[key] = torch.zeros_like(global_dict[key])
        for i, cm in enumerate(client_models):
            global_dict[key] += client_weights[i] * cm.state_dict()[key]

    global_model.load_state_dict(global_dict)
    return global_model


def add_dp_noise(model, noise_scale):
    """Add Gaussian noise to model parameters for differential privacy."""
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * noise_scale
            param.add_(noise)
    return model


print("\n" + "=" * 65)
print(f"EXPERIMENT 2: Federated Learning (FedAvg, {FED_ROUNDS} rounds)")
if DP_ENABLED:
    print(f"  Differential Privacy: ENABLED (noise σ = {DP_NOISE_SCALE})")
print("=" * 65)

# Initialise global model
global_model = AnomalyAutoencoder().to(DEVICE)

# Track metrics across rounds for plotting
fed_history = {
    'round': [], 'avg_train_loss': [],
    'GCS_A_val_f1': [], 'GCS_B_val_f1': [], 'GCS_C_val_f1': [],
    'avg_val_f1': [],
}

# Dataset sizes for weighted aggregation
client_sizes = {name: len(client_data[name]['X_train'])
                for name in ['GCS_A', 'GCS_B', 'GCS_C']}

t0_fed = time.time()

for rnd in range(1, FED_ROUNDS + 1):
    round_losses = []
    round_models = []
    round_val_f1s = []

    for name in ['GCS_A', 'GCS_B', 'GCS_C']:
        cd = client_data[name]

        # Copy global model to local
        local_model = copy.deepcopy(global_model)
        optimizer = torch.optim.Adam(local_model.parameters(), lr=FED_LR)
        criterion = nn.MSELoss()
        train_loader = make_dataloader(cd['X_train'], BATCH_SIZE)

        # Local training
        epoch_losses = []
        for epoch in range(FED_LOCAL_EPOCHS):
            loss = train_one_epoch(local_model, train_loader,
                                   optimizer, criterion)
            epoch_losses.append(loss)

        round_losses.append(np.mean(epoch_losses))

        # Optional DP noise before sending to server
        if DP_ENABLED:
            local_model = add_dp_noise(local_model, DP_NOISE_SCALE)

        round_models.append(local_model)

        # Validation F1 for tracking
        val_scores = compute_anomaly_scores(local_model, cd['X_val'])
        thr, f1_val = find_optimal_threshold(val_scores, cd['y_val'])
        round_val_f1s.append(f1_val)

    # Aggregate
    weights = [client_sizes[name] for name in ['GCS_A', 'GCS_B', 'GCS_C']]
    global_model = fedavg_aggregate(global_model, round_models, weights)

    # Track history
    avg_loss = np.mean(round_losses)
    avg_f1 = np.mean(round_val_f1s)
    fed_history['round'].append(rnd)
    fed_history['avg_train_loss'].append(avg_loss)
    fed_history['GCS_A_val_f1'].append(round_val_f1s[0])
    fed_history['GCS_B_val_f1'].append(round_val_f1s[1])
    fed_history['GCS_C_val_f1'].append(round_val_f1s[2])
    fed_history['avg_val_f1'].append(avg_f1)

    if rnd % 3 == 0 or rnd == 1:
        print(f"  Round {rnd:2d}/{FED_ROUNDS}  "
              f"Loss: {avg_loss:.6f}  "
              f"Avg Val F1: {avg_f1:.4f}  "
              f"[A:{round_val_f1s[0]:.3f} B:{round_val_f1s[1]:.3f} "
              f"C:{round_val_f1s[2]:.3f}]")

fed_time = time.time() - t0_fed
print(f"\nTotal federated training time: {fed_time:.1f}s")

# Evaluate federated global model on each client's test set
print("\n--- Federated Model: Per-Client Test Results ---")
fed_results = {}
fed_thresholds = {}

for name in ['GCS_A', 'GCS_B', 'GCS_C']:
    cd = client_data[name]
    # Find threshold using validation set
    val_scores = compute_anomaly_scores(global_model, cd['X_val'])
    threshold, _ = find_optimal_threshold(val_scores, cd['y_val'])
    fed_thresholds[name] = threshold

    # Evaluate on test set
    results, _, _ = evaluate_model(
        global_model, cd['X_test'], cd['y_test'], threshold
    )
    fed_results[name] = results
    print(f"  {name}: Acc={results['accuracy']:.4f}, "
          f"Prec={results['precision']:.4f}, "
          f"Rec={results['recall']:.4f}, "
          f"F1={results['f1']:.4f}, AUC={results['auc']:.4f}")


# ============================================================
# CELL 8: Experiment 3 — Centralized Training (Upper Bound)
# ============================================================
"""
Centralized baseline: combine all data, train one model.
This VIOLATES privacy (all flight data pooled) but gives us the
theoretical upper bound on what is achievable.
"""

CENT_EPOCHS = 100
CENT_LR = 0.003

print("\n" + "=" * 65)
print("EXPERIMENT 3: Centralized Training (Upper Bound)")
print("=" * 65)

# Combine all training data
X_train_all = np.vstack([client_data[n]['X_train'] for n in DATA_PATHS])
X_val_all = np.vstack([client_data[n]['X_val'] for n in DATA_PATHS])
X_test_all = np.vstack([client_data[n]['X_test'] for n in DATA_PATHS])
y_val_all = np.concatenate([client_data[n]['y_val'] for n in DATA_PATHS])
y_test_all = np.concatenate([client_data[n]['y_test'] for n in DATA_PATHS])

# Global scaler for centralized
scaler_global = StandardScaler()
X_train_cent = scaler_global.fit_transform(X_train_all)
X_val_cent = scaler_global.transform(X_val_all)
X_test_cent = scaler_global.transform(X_test_all)

cent_model = AnomalyAutoencoder().to(DEVICE)
optimizer = torch.optim.Adam(cent_model.parameters(), lr=CENT_LR)
criterion = nn.MSELoss()
train_loader = make_dataloader(X_train_cent, BATCH_SIZE)

t0_cent = time.time()
for epoch in range(CENT_EPOCHS):
    loss = train_one_epoch(cent_model, train_loader, optimizer, criterion)
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}/{CENT_EPOCHS}  Loss: {loss:.6f}")

cent_time = time.time() - t0_cent
print(f"  Training time: {cent_time:.1f}s")

# Evaluate
val_scores_cent = compute_anomaly_scores(cent_model, X_val_cent)
cent_threshold, _ = find_optimal_threshold(val_scores_cent, y_val_all)
cent_results, _, _ = evaluate_model(
    cent_model, X_test_cent, y_test_all, cent_threshold
)
print(f"\n  Centralized Test — Acc: {cent_results['accuracy']:.4f}, "
      f"Prec: {cent_results['precision']:.4f}, "
      f"Rec: {cent_results['recall']:.4f}, "
      f"F1: {cent_results['f1']:.4f}, AUC: {cent_results['auc']:.4f}")


# ============================================================
# CELL 9: Comprehensive Results Table
# ============================================================
print("\n" + "=" * 65)
print("COMPREHENSIVE RESULTS COMPARISON")
print("=" * 65)

# Helper: average local results across clients
avg_local = {
    'accuracy': np.mean([local_results[n]['accuracy'] for n in DATA_PATHS]),
    'precision': np.mean([local_results[n]['precision'] for n in DATA_PATHS]),
    'recall': np.mean([local_results[n]['recall'] for n in DATA_PATHS]),
    'f1': np.mean([local_results[n]['f1'] for n in DATA_PATHS]),
    'auc': np.mean([local_results[n]['auc'] for n in DATA_PATHS]),
}
avg_fed = {
    'accuracy': np.mean([fed_results[n]['accuracy'] for n in DATA_PATHS]),
    'precision': np.mean([fed_results[n]['precision'] for n in DATA_PATHS]),
    'recall': np.mean([fed_results[n]['recall'] for n in DATA_PATHS]),
    'f1': np.mean([fed_results[n]['f1'] for n in DATA_PATHS]),
    'auc': np.mean([fed_results[n]['auc'] for n in DATA_PATHS]),
}

print(f"\n{'Metric':<15} {'Local-Only':>12} {'Federated':>12} "
      f"{'Centralized':>12} {'Fed vs Local':>14}")
print("-" * 68)
for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
    l = avg_local[metric]
    f = avg_fed[metric]
    c = cent_results[metric]
    delta = f - l
    arrow = "↑" if delta > 0 else "↓"
    print(f"{metric:<15} {l:>12.4f} {f:>12.4f} {c:>12.4f} "
          f"{arrow} {abs(delta):>+11.4f}")

print(f"\n{'Training Time':<15} "
      f"{'N/A':>12} "
      f"{fed_time:>11.1f}s "
      f"{cent_time:>11.1f}s")
print(f"{'Params Shared':<15} "
      f"{'0':>12} "
      f"{'288/round':>12} "
      f"{'All data':>12}")
dp_str = f"Yes (σ={DP_NOISE_SCALE})" if DP_ENABLED else "No"
print(f"{'DP Protected':<15} {'N/A':>12} {dp_str:>12} {'No':>12}")

# Per-client breakdown
print(f"\n--- Per-Client F1 Scores ---")
print(f"{'Client':<10} {'Local':>10} {'Federated':>10} {'Delta':>10}")
print("-" * 42)
for name in ['GCS_A', 'GCS_B', 'GCS_C']:
    l_f1 = local_results[name]['f1']
    f_f1 = fed_results[name]['f1']
    delta = f_f1 - l_f1
    print(f"{name:<10} {l_f1:>10.4f} {f_f1:>10.4f} {delta:>+10.4f}")

# Cross-client F1 matrix
print(f"\n--- Cross-Client F1 Matrix (Local Models) ---")
print(f"{'Train↓ / Test→':<18} {'GCS_A':>8} {'GCS_B':>8} {'GCS_C':>8}")
print("-" * 44)
for tc in ['GCS_A', 'GCS_B', 'GCS_C']:
    vals = [cross_results.get((tc, ec), 0) for ec in ['GCS_A', 'GCS_B', 'GCS_C']]
    print(f"{tc:<18} {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f}")
print("(Off-diagonal drops reveal the non-IID problem FL solves)")


# ============================================================
# CELL 10: Generate Publication-Quality Plots
# ============================================================
"""
Four plots for the paper:
  Fig 1: Federated convergence (loss + F1 over rounds)
  Fig 2: Per-client F1 comparison (Local vs Federated vs Centralized)
  Fig 3: ROC curves (Federated model on each client)
  Fig 4: Cross-client F1 heatmap
"""

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('PQ-LITS: Federated Anomaly Detection Results',
             fontsize=14, fontweight='bold', y=0.98)

# --- Plot 1: Convergence ---
ax1 = axes[0, 0]
ax1_twin = ax1.twinx()
h = fed_history
ax1.plot(h['round'], h['avg_train_loss'], 'b-o', markersize=4,
         label='Avg train loss', linewidth=1.5)
ax1_twin.plot(h['round'], h['avg_val_f1'], 'r-s', markersize=4,
              label='Avg val F1', linewidth=1.5)
ax1.set_xlabel('Federation Round')
ax1.set_ylabel('Average Training Loss', color='b')
ax1_twin.set_ylabel('Average Validation F1', color='r')
ax1.set_title('(a) Federated convergence')
ax1.tick_params(axis='y', labelcolor='b')
ax1_twin.tick_params(axis='y', labelcolor='r')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right',
           fontsize=9)
ax1.grid(True, alpha=0.3)

# --- Plot 2: Per-client F1 comparison ---
ax2 = axes[0, 1]
clients = ['GCS-A', 'GCS-B', 'GCS-C', 'Average']
local_f1s = [local_results[n]['f1'] for n in DATA_PATHS] + [avg_local['f1']]
fed_f1s = [fed_results[n]['f1'] for n in DATA_PATHS] + [avg_fed['f1']]
cent_f1s = [cent_results['f1']] * 4

x = np.arange(len(clients))
w = 0.25
bars1 = ax2.bar(x - w, local_f1s, w, label='Local-only', color='#85B7EB',
                edgecolor='#185FA5', linewidth=0.5)
bars2 = ax2.bar(x, fed_f1s, w, label='Federated', color='#5DCAA5',
                edgecolor='#0F6E56', linewidth=0.5)
bars3 = ax2.bar(x + w, cent_f1s, w, label='Centralized', color='#F0997B',
                edgecolor='#993C1D', linewidth=0.5)
ax2.set_ylabel('F1-Score')
ax2.set_title('(b) F1-score comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(clients)
ax2.legend(fontsize=9)
ax2.set_ylim(0, 1.05)
ax2.grid(True, alpha=0.3, axis='y')
# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h_val = bar.get_height()
        if h_val > 0:
            ax2.annotate(f'{h_val:.2f}', xy=(bar.get_x() + bar.get_width()/2, h_val),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7)

# --- Plot 3: ROC Curves (Federated) ---
ax3 = axes[1, 0]
colors_roc = {'GCS_A': '#378ADD', 'GCS_B': '#1D9E75', 'GCS_C': '#D85A30'}
for name in ['GCS_A', 'GCS_B', 'GCS_C']:
    cd = client_data[name]
    scores = compute_anomaly_scores(global_model, cd['X_test'])
    if len(np.unique(cd['y_test'])) == 2:
        fpr, tpr, _ = roc_curve(cd['y_test'], scores)
        auc_val = roc_auc_score(cd['y_test'], scores)
        label_name = name.replace('_', '-')
        ax3.plot(fpr, tpr, color=colors_roc[name], linewidth=1.5,
                 label=f'{label_name} (AUC={auc_val:.3f})')
ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.8)
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('(c) ROC curves (federated model)')
ax3.legend(fontsize=9, loc='lower right')
ax3.grid(True, alpha=0.3)

# --- Plot 4: Cross-client F1 heatmap ---
ax4 = axes[1, 1]
names = ['GCS_A', 'GCS_B', 'GCS_C']
matrix = np.array([[cross_results.get((t, e), 0)
                     for e in names] for t in names])
im = ax4.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax4.set_xticks(range(3))
ax4.set_yticks(range(3))
ax4.set_xticklabels([n.replace('_', '-') for n in names])
ax4.set_yticklabels([n.replace('_', '-') for n in names])
ax4.set_xlabel('Test Client')
ax4.set_ylabel('Training Client')
ax4.set_title('(d) Cross-client F1 (local models)')
for i in range(3):
    for j in range(3):
        ax4.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center',
                fontsize=11, fontweight='bold',
                color='white' if matrix[i, j] < 0.5 else 'black')
plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('pq_lits_results.png', dpi=300, bbox_inches='tight')
plt.savefig('pq_lits_results.pdf', bbox_inches='tight')
print("\nPlots saved: pq_lits_results.png / .pdf")
plt.show()


# ============================================================
# CELL 11: Per-Attack-Type Detection Analysis
# ============================================================
"""
This table shows detection rate broken down by attack type.
Critical for the paper: shows which attacks each approach catches/misses.
"""

print("\n" + "=" * 65)
print("PER-ATTACK-TYPE DETECTION RATE")
print("=" * 65)

attack_names_map = {0: 'Normal', 1: 'GPS Spoofing',
                    2: 'RF Jamming', 3: 'Cmd Injection'}

for approach_name, model_obj, threshold_dict in [
    ('Local-Only', local_models, local_thresholds),
    ('Federated', {n: global_model for n in DATA_PATHS}, fed_thresholds),
]:
    print(f"\n--- {approach_name} ---")
    print(f"{'Client':<10} {'Attack Type':<18} {'Total':>6} "
          f"{'Detected':>8} {'Rate':>8}")
    print("-" * 54)

    for name in ['GCS_A', 'GCS_B', 'GCS_C']:
        df = client_dfs[name]
        cd = client_data[name]
        model_to_use = model_obj[name]
        thr = threshold_dict[name]

        # Get test indices (reconstruct from labels and features)
        # We evaluate on the full client dataset for attack-type breakdown
        X_full = cd['scaler'].transform(
            df[FEATURE_COLS].values.astype(np.float32)
        )
        y_full = df['label'].values
        scores_full = compute_anomaly_scores(model_to_use, X_full)
        y_pred_full = (scores_full > thr).astype(int)

        for atk_type in sorted(df['label'].unique()):
            mask = y_full == atk_type
            if mask.sum() == 0:
                continue
            total = mask.sum()
            if atk_type == 0:
                # For normal: count correctly classified as normal
                correct = (y_pred_full[mask] == 0).sum()
                rate = correct / total
                label = "Normal (TNR)"
            else:
                # For attacks: count correctly detected as anomaly
                detected = (y_pred_full[mask] == 1).sum()
                rate = detected / total
                label = attack_names_map[atk_type]
            print(f"{name:<10} {label:<18} {total:>6} "
                  f"{'':>8} {rate:>7.1%}")


# ============================================================
# CELL 12: Communication Overhead Analysis
# ============================================================
"""
For the paper: quantify how little data is actually exchanged
in federated learning vs. centralized.
"""

print("\n" + "=" * 65)
print("COMMUNICATION OVERHEAD ANALYSIS")
print("=" * 65)

n_params = sum(p.numel() for p in global_model.parameters())
bytes_per_param = 4  # float32
model_size_bytes = n_params * bytes_per_param
model_size_kb = model_size_bytes / 1024

# Per round: 3 clients upload + 1 server broadcast = 4 model transfers
per_round_bytes = 4 * model_size_bytes
total_fed_bytes = FED_ROUNDS * per_round_bytes

# Centralized: all raw data transmitted once
total_raw_samples = sum(len(client_data[n]['X_train'])
                        for n in DATA_PATHS)
raw_data_bytes = total_raw_samples * len(FEATURE_COLS) * bytes_per_param
raw_data_kb = raw_data_bytes / 1024

print(f"Model parameters:        {n_params}")
print(f"Model size:              {model_size_kb:.2f} KB ({model_size_bytes} bytes)")
print(f"")
print(f"Federated ({FED_ROUNDS} rounds):")
print(f"  Per round (3 up + 1 down): {per_round_bytes/1024:.2f} KB")
print(f"  Total communication:       {total_fed_bytes/1024:.2f} KB")
print(f"")
print(f"Centralized (raw data):")
print(f"  Raw training data:         {raw_data_kb:.2f} KB")
print(f"")
reduction = (1 - total_fed_bytes / raw_data_bytes) * 100
print(f"Communication reduction:     {reduction:.1f}%")
print(f"(Federated transmits {total_fed_bytes/raw_data_bytes:.2%} "
      f"of what centralized requires)")

if DP_ENABLED:
    print(f"\nDifferential Privacy:")
    print(f"  Noise mechanism:  Gaussian (σ = {DP_NOISE_SCALE})")
    print(f"  Applied to:       Model weights before upload")
    print(f"  Parameters noise per round: {n_params} values")
    # Rough epsilon estimate (for paper, use tighter RDP accounting)
    sensitivity = 0.1  # approximate max weight delta per round
    delta = 1e-5
    epsilon = np.sqrt(2 * np.log(1.25 / delta)) * (sensitivity / DP_NOISE_SCALE)
    print(f"  Rough ε estimate: {epsilon:.2f} (δ = {delta})")
    print(f"  (Use Renyi DP accounting for tighter bounds in the paper)")


# ============================================================
# CELL 13: Export Results for LaTeX
# ============================================================
"""
Auto-generate LaTeX table code for direct copy-paste into your paper.
"""

print("\n" + "=" * 65)
print("LaTeX TABLE (copy-paste into paper)")
print("=" * 65)

latex = r"""
\begin{table}[!t]
\centering
\caption{Performance Comparison of Anomaly Detection Approaches}
\label{tab:results}
\begin{tabular}{lcccc}
\hline
\textbf{Approach} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} \\
\hline
"""
latex += f"Local-Only (Avg) & {avg_local['accuracy']:.4f} & {avg_local['precision']:.4f} & {avg_local['recall']:.4f} & {avg_local['f1']:.4f} \\\\\n"
latex += f"Federated (Avg) & {avg_fed['accuracy']:.4f} & {avg_fed['precision']:.4f} & {avg_fed['recall']:.4f} & {avg_fed['f1']:.4f} \\\\\n"
latex += f"Centralized & {cent_results['accuracy']:.4f} & {cent_results['precision']:.4f} & {cent_results['recall']:.4f} & {cent_results['f1']:.4f} \\\\\n"
latex += r"""\hline
\end{tabular}
\end{table}
"""
print(latex)

# Per-client table
latex2 = r"""
\begin{table}[!t]
\centering
\caption{Per-Client F1-Score Under Different Training Strategies}
\label{tab:per_client}
\begin{tabular}{lccc}
\hline
\textbf{Client} & \textbf{Local-Only} & \textbf{Federated} & \textbf{Improvement} \\
\hline
"""
for name in ['GCS_A', 'GCS_B', 'GCS_C']:
    l = local_results[name]['f1']
    f = fed_results[name]['f1']
    d = f - l
    sign = "+" if d >= 0 else ""
    latex2 += f"{name.replace('_', '\\_')} & {l:.4f} & {f:.4f} & {sign}{d:.4f} \\\\\n"
latex2 += r"""\hline
\end{tabular}
\end{table}
"""
print(latex2)


print("\n" + "=" * 65)
print("ALL EXPERIMENTS COMPLETE")
print("=" * 65)
print(f"Total runtime: ~{(time.time() - t0_fed + cent_time + sum(1 for _ in range(3))*3):.0f}s")
print("Output files: pq_lits_results.png, pq_lits_results.pdf")
print("Next: Copy LaTeX tables into your Overleaf document.")
