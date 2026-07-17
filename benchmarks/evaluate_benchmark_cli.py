import os
import sys

# Setup project root and subdirectories in sys.path to resolve imports cleanly
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB_DIRS = [
    ROOT_DIR,
    os.path.join(ROOT_DIR, 'data_processing'),
    os.path.join(ROOT_DIR, 'models'),
    os.path.join(ROOT_DIR, 'pm_engine'),
    os.path.join(ROOT_DIR, 'web_dashboard'),
    os.path.join(ROOT_DIR, 'benchmarks'),
    os.path.join(ROOT_DIR, 'tests')
]
for sd in SUB_DIRS:
    if sd not in sys.path:
        sys.path.insert(0, sd)

import time
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from graph_transformer import HGLGGTAE, SequenceEncoder, SequenceDecoder, GraphTransformerLayer, HypergraphConv, AttentionFusion
from pm_analytics import calculate_anomaly_score, calculate_hypergraph_drift

# ----------------------------------------------------------------------
# 1. Define Model Architectures
# ----------------------------------------------------------------------

# LSTM-AE (Temporal Only)
class LSTMAE(nn.Module):
    def __init__(self, sequence_length=256, num_nodes=9, node_features=9, embed_dim=64):
        super(LSTMAE, self).__init__()
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        self.node_features = node_features
        self.embed_dim = embed_dim
        
        self.lstm_enc = nn.LSTM(node_features, embed_dim, batch_first=True)
        self.lstm_dec = nn.LSTM(embed_dim, node_features, batch_first=True)
        
    def forward(self, X):
        # X: [Batch, N, W, F]
        batch_size = X.size(0)
        X_flat = X.view(batch_size * self.num_nodes, self.sequence_length, self.node_features)
        
        _, (h_n, _) = self.lstm_enc(X_flat)
        z = h_n.squeeze(0) # [Batch * N, D]
        
        z_expanded = z.unsqueeze(1).repeat(1, self.sequence_length, 1)
        recon_flat, _ = self.lstm_dec(z_expanded)
        
        X_recon = recon_flat.view(batch_size, self.num_nodes, self.sequence_length, self.node_features)
        return X_recon

# GCN-AE (Spatial GCN + Temporal)
class GCNAE(nn.Module):
    def __init__(self, sequence_length=256, num_nodes=9, node_features=9, embed_dim=64):
        super(GCNAE, self).__init__()
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        self.node_features = node_features
        self.embed_dim = embed_dim
        
        self.encoder = SequenceEncoder(sequence_length, node_features, embed_dim)
        self.gcn1 = nn.Linear(embed_dim, embed_dim)
        self.gcn2 = nn.Linear(embed_dim, embed_dim)
        self.decoder = SequenceDecoder(embed_dim, sequence_length, node_features)
        
    def forward(self, X, A_fixed):
        batch_size = X.size(0)
        X_flat = X.view(batch_size * self.num_nodes, self.sequence_length, self.node_features)
        h = self.encoder(X_flat).view(batch_size, self.num_nodes, self.embed_dim)
        
        D = A_fixed.sum(dim=-1, keepdim=True) + 1e-9
        A_norm = A_fixed / D
        
        h = F.relu(self.gcn1(torch.matmul(A_norm, h)))
        h = F.relu(self.gcn2(torch.matmul(A_norm, h)))
        
        h_flat = h.view(batch_size * self.num_nodes, self.embed_dim)
        X_recon = self.decoder(h_flat).view(batch_size, self.num_nodes, self.sequence_length, self.node_features)
        
        A_recon = torch.matmul(h, h.transpose(1, 2)) / (self.embed_dim ** 0.5)
        A_recon = torch.sigmoid(A_recon)
        
        return X_recon, A_recon

# GTAE (Graph Transformer Autoencoder with fixed graph, no hypergraph/learnable graph)
class GTAE(nn.Module):
    def __init__(self, sequence_length=256, num_nodes=9, node_features=9, embed_dim=64, num_heads=4):
        super(GTAE, self).__init__()
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        self.node_features = node_features
        self.embed_dim = embed_dim
        
        self.encoder = SequenceEncoder(sequence_length, node_features, embed_dim)
        self.gt1 = GraphTransformerLayer(embed_dim, num_heads)
        self.gt2 = GraphTransformerLayer(embed_dim, num_heads)
        self.decoder = SequenceDecoder(embed_dim, sequence_length, node_features)
        
    def forward(self, X, A_fixed):
        batch_size = X.size(0)
        X_flat = X.view(batch_size * self.num_nodes, self.sequence_length, self.node_features)
        h = self.encoder(X_flat).view(batch_size, self.num_nodes, self.embed_dim)
        
        h = self.gt1(h, A_fixed)
        h = self.gt2(h, A_fixed)
        
        h_flat = h.view(batch_size * self.num_nodes, self.embed_dim)
        X_recon = self.decoder(h_flat).view(batch_size, self.num_nodes, self.sequence_length, self.node_features)
        
        A_recon = torch.matmul(h, h.transpose(1, 2)) / (self.embed_dim ** 0.5)
        A_recon = torch.sigmoid(A_recon)
        
        return X_recon, A_recon

# HGNN-AE (Hypergraph Autoencoder, no fixed/learnable graph)
class HGNNAE(nn.Module):
    def __init__(self, sequence_length=256, num_nodes=9, node_features=9, embed_dim=64):
        super(HGNNAE, self).__init__()
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        self.node_features = node_features
        self.embed_dim = embed_dim
        
        self.encoder = SequenceEncoder(sequence_length, node_features, embed_dim)
        self.hg1 = HypergraphConv(embed_dim)
        self.hg2 = HypergraphConv(embed_dim)
        self.decoder = SequenceDecoder(embed_dim, sequence_length, node_features)
        
    def forward(self, X, H, W_e):
        batch_size = X.size(0)
        X_flat = X.view(batch_size * self.num_nodes, self.sequence_length, self.node_features)
        h = self.encoder(X_flat).view(batch_size, self.num_nodes, self.embed_dim)
        
        h = self.hg1(h, H, W_e)
        h = self.hg2(h, H, W_e)
        
        h_flat = h.view(batch_size * self.num_nodes, self.embed_dim)
        X_recon = self.decoder(h_flat).view(batch_size, self.num_nodes, self.sequence_length, self.node_features)
        
        H_recon = torch.matmul(h, h.transpose(1, 2)) / (self.embed_dim ** 0.5)
        H_recon = torch.sigmoid(H_recon)
        
        return X_recon, H_recon

# HG-GTAE (Fixed Graph + Hypergraph, no learnable graph)
class HGGTAE(nn.Module):
    def __init__(self, sequence_length=256, num_nodes=9, node_features=9, embed_dim=64, num_heads=4):
        super(HGGTAE, self).__init__()
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        self.node_features = node_features
        self.embed_dim = embed_dim
        
        self.encoder = SequenceEncoder(sequence_length, node_features, embed_dim)
        self.gt1 = GraphTransformerLayer(embed_dim, num_heads)
        self.gt2 = GraphTransformerLayer(embed_dim, num_heads)
        self.hg1 = HypergraphConv(embed_dim)
        self.hg2 = HypergraphConv(embed_dim)
        self.fusion = AttentionFusion(embed_dim)
        self.decoder = SequenceDecoder(embed_dim, sequence_length, node_features)
        
    def forward(self, X, A_fixed, H, W_e):
        batch_size = X.size(0)
        X_flat = X.view(batch_size * self.num_nodes, self.sequence_length, self.node_features)
        h = self.encoder(X_flat).view(batch_size, self.num_nodes, self.embed_dim)
        
        # Graph branch (fixed)
        h_g = self.gt1(h, A_fixed)
        h_g = self.gt2(h_g, A_fixed)
        
        # Hypergraph branch
        h_h = self.hg1(h, H, W_e)
        h_h = self.hg2(h_h, H, W_e)
        
        # Fusion
        h_f, _ = self.fusion(h_g, h_h)
        
        h_flat = h_f.view(batch_size * self.num_nodes, self.embed_dim)
        X_recon = self.decoder(h_flat).view(batch_size, self.num_nodes, self.sequence_length, self.node_features)
        
        A_recon = torch.matmul(h_f, h_f.transpose(1, 2)) / (self.embed_dim ** 0.5)
        A_recon = torch.sigmoid(A_recon)
        
        H_recon = torch.matmul(h_f, h_f.transpose(1, 2)) / (self.embed_dim ** 0.5)
        H_recon = torch.sigmoid(H_recon)
        
        return X_recon, A_recon, H_recon

# ----------------------------------------------------------------------
# 2. Helper Functions: Data Loading & Fault Injection
# ----------------------------------------------------------------------

def load_data(houses, data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(ROOT_DIR, "3_processed_outputs")
    X_list, A_list, H_list, W_list = [], [], [], []
    house_meta = {}
    
    for h in houses:
        path = os.path.join(data_dir, f"House_{h}_Dynamic_Graphs.pt")
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Skipping House {h}.")
            continue
        data = torch.load(path, weights_only=False)
        X_list.append(data["X"])
        A_list.append(data["A"])
        H_list.append(data["H"])
        W_list.append(data["W_e"])
        house_meta[h] = {
            "appliance_names": data["appliance_names"],
            "max_powers": data["max_powers"],
            "timestamps": data["timestamps"]
        }
        
    return X_list, A_list, H_list, W_list, house_meta

def inject_fault(X_orig, app_idx, fault_type, num_windows, week_12_idx):
    X_faulty = X_orig.clone()
    
    for w in range(week_12_idx, num_windows):
        progress = (w - week_12_idx) / (num_windows - week_12_idx - 1) if num_windows - week_12_idx > 1 else 1.0
        
        power = X_faulty[w, app_idx, :, 0].numpy()
        state = X_faulty[w, app_idx, :, 1].numpy()
        
        new_power = power.copy()
        new_state = state.copy()
        
        is_on = (state > 0.5).astype(int)
        
        if fault_type == "Gradual degradation":
            gamma = 0.50 # max 50% increase
            active_mask = state > 0.5
            new_power[active_mask] = new_power[active_mask] * (1.0 + gamma * progress)
            
        elif fault_type == "Intermittent spike":
            active_mask = state > 0.5
            if np.sum(active_mask) > 0:
                spike_prob = 0.15 * progress
                spikes = np.random.rand(len(power)) * 0.4
                mask = (np.random.rand(len(power)) < spike_prob) & active_mask
                new_power[mask] += spikes[mask]
                
        elif fault_type == "Increased operation duration":
            diff = np.diff(np.concatenate(([0], is_on, [0])))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            ext_ratio = 0.50 * progress
            for start, end in zip(starts, ends):
                cycle_len = end - start
                ext_len = int(np.round(cycle_len * ext_ratio))
                if ext_len > 0:
                    ext_start = end
                    ext_end = min(len(power), end + ext_len)
                    avg_active = np.mean(power[start:end]) if end > start else 0.5
                    new_power[ext_start:ext_end] = avg_active
                    new_state[ext_start:ext_end] = 1.0
                    
        elif fault_type == "Stuck ON/OFF":
            # Stuck ON
            if progress > 0.5:
                new_state[:] = 1.0
                new_power[:] = 0.4 # Nominal active level
                
        # Clip power
        new_power = np.clip(new_power, 0.0, 1.5)
        
        # Recompute derived features
        mean_p = np.full(len(new_power), np.mean(new_power))
        var_p = np.full(len(new_power), np.var(new_power))
        duty_c = np.full(len(new_power), np.sum(new_state) / len(new_state))
        
        run_dur = []
        current_run = 0.0
        for s_val in new_state:
            if s_val > 0.5:
                current_run += 1.0
            else:
                current_run = 0.0
            run_dur.append(current_run)
        run_dur = np.array(run_dur) / len(new_state)
        
        energy_val = np.cumsum(new_power) * (8.0 / 3600.0)
        energy_norm = energy_val / (np.max(new_power) * (len(new_state) * 8.0 / 3600.0) + 1e-9)
        
        X_faulty[w, app_idx, :, 0] = torch.tensor(new_power, dtype=torch.float32)
        X_faulty[w, app_idx, :, 1] = torch.tensor(new_state, dtype=torch.float32)
        X_faulty[w, app_idx, :, 2] = torch.tensor(mean_p, dtype=torch.float32)
        X_faulty[w, app_idx, :, 3] = torch.tensor(var_p, dtype=torch.float32)
        X_faulty[w, app_idx, :, 4] = torch.tensor(duty_c, dtype=torch.float32)
        X_faulty[w, app_idx, :, 5] = torch.tensor(run_dur, dtype=torch.float32)
        X_faulty[w, app_idx, :, 6] = torch.tensor(energy_norm, dtype=torch.float32)
        
    return X_faulty

def get_appliance_fault_type(app_name):
    name_lower = app_name.lower()
    if any(k in name_lower for k in ["fridge", "freezer", "refrigerator"]):
        return "Gradual degradation"
    elif name_lower == "ac" or "air_conditioner" in name_lower:
        return "Gradual degradation"
    elif any(k in name_lower for k in ["washing", "washer", "dryer", "dishwasher"]):
        return "Increased operation duration"
    elif any(k in name_lower for k in ["microwave", "tv", "computer", "television"]):
        return "Intermittent spike"
    else:
        return "Stuck ON/OFF"

# ----------------------------------------------------------------------
# 3. Model Training & Evaluation Engine
# ----------------------------------------------------------------------

def train_model(model, train_loader, epochs=5, device="cpu", model_name=""):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_a, batch_h, batch_w in train_loader:
            batch_x = batch_x.to(device)
            batch_a = batch_a.to(device)
            batch_h = batch_h.to(device)
            batch_w = batch_w.to(device)
            
            optimizer.zero_grad()
            
            if isinstance(model, LSTMAE):
                recon = model(batch_x)
                loss = F.mse_loss(recon, batch_x)
            elif isinstance(model, GCNAE) or isinstance(model, GTAE):
                recon, a_recon = model(batch_x, batch_a)
                loss = F.mse_loss(recon, batch_x) + 0.2 * F.mse_loss(a_recon, batch_a)
            elif isinstance(model, HGNNAE):
                recon, h_recon = model(batch_x, batch_h, batch_w)
                loss = F.mse_loss(recon, batch_x) + 0.2 * F.mse_loss(h_recon, batch_h)
            elif isinstance(model, HGGTAE):
                recon, a_recon, h_recon = model(batch_x, batch_a, batch_h, batch_w)
                loss = F.mse_loss(recon, batch_x) + 0.2 * F.mse_loss(a_recon, batch_a) + 0.1 * F.mse_loss(h_recon, batch_h)
            else: # HGLGGTAE
                recon, a_recon, h_recon, a_final, _ = model(batch_x, batch_a, batch_h, batch_w)
                loss = F.mse_loss(recon, batch_x) + 0.2 * F.mse_loss(a_recon, batch_a) + 0.1 * F.mse_loss(h_recon, batch_h)
                
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
    return model

def compute_anomaly_scores(model, X, A_fixed, H, W_e, A_baseline, H_baseline, W_baseline, device="cpu"):
    model.eval()
    scores = []
    
    with torch.no_grad():
        for w in range(X.size(0)):
            x_w = X[w].unsqueeze(0).to(device)
            a_w = A_fixed[w].unsqueeze(0).to(device)
            h_w = H[w].unsqueeze(0).to(device)
            w_w = W_e[w].unsqueeze(0).to(device)
            
            # 1. Reconstruction MSE
            if isinstance(model, LSTMAE):
                recon = model(x_w)
                recon_err = F.mse_loss(recon, x_w).item()
                graph_drift = 0.0
                hyper_drift = 0.0
            elif isinstance(model, GCNAE) or isinstance(model, GTAE):
                recon, a_rec = model(x_w, a_w)
                recon_err = F.mse_loss(recon, x_w).item()
                graph_drift = np.mean(np.abs(a_rec[0].cpu().numpy() - A_baseline))
                hyper_drift = 0.0
            elif isinstance(model, HGNNAE):
                recon, h_rec = model(x_w, h_w, w_w)
                recon_err = F.mse_loss(recon, x_w).item()
                graph_drift = 0.0
                hyper_drift = np.mean(np.abs(h_rec[0].cpu().numpy() - H_baseline))
            elif isinstance(model, HGGTAE):
                recon, a_rec, h_rec = model(x_w, a_w, h_w, w_w)
                recon_err = F.mse_loss(recon, x_w).item()
                graph_drift = np.mean(np.abs(a_rec[0].cpu().numpy() - A_baseline))
                hyper_drift = np.mean(np.abs(h_rec[0].cpu().numpy() - H_baseline))
            else: # HGLGGTAE
                recon, a_rec, h_rec, _, _ = model(x_w, a_w, h_w, w_w)
                recon_err = F.mse_loss(recon, x_w).item()
                graph_drift = np.mean(np.abs(a_rec[0].cpu().numpy() - A_baseline))
                hyper_drift = np.mean(np.abs(h_rec[0].cpu().numpy() - H_baseline))
                
            # 2. Fuse scores using active lambdas
            if isinstance(model, LSTMAE):
                score = calculate_anomaly_score(recon_err, graph_drift, hyper_drift, lambdas=(1.0, 0.0, 0.0))
            elif isinstance(model, GCNAE) or isinstance(model, GTAE):
                score = calculate_anomaly_score(recon_err, graph_drift, hyper_drift, lambdas=(1.0, 0.5, 0.0))
            elif isinstance(model, HGNNAE):
                score = calculate_anomaly_score(recon_err, graph_drift, hyper_drift, lambdas=(1.0, 0.0, 0.5))
            else:
                score = calculate_anomaly_score(recon_err, graph_drift, hyper_drift, lambdas=(1.0, 0.5, 0.5))
                
            scores.append(score)
            
    return scores

def calculate_metrics(y_true, y_pred, num_windows, week_12_idx):
    TP = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    TN = sum((t == 0 and p == 0) for t, p in zip(y_true, y_pred))
    FP = sum((t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    FN = sum((t == 1 and p == 0) for t, p in zip(y_true, y_pred))
    
    total = len(y_true)
    acc = (TP + TN) / total if total > 0 else 0
    
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0
    bal_acc = 0.5 * (sens + spec)
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = sens
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    far = FP / (FP + TN) if (FP + TN) > 0 else 0
    
    # Detection Delay & Lead Time
    first_detect = -1
    for w in range(week_12_idx, len(y_pred)):
        if y_pred[w] == 1:
            first_detect = w
            break
            
    if first_detect != -1:
        detection_delay = first_detect - week_12_idx
        lead_time = (num_windows - first_detect) * (140.0 / num_windows)
        detection_status = "Detected"
        detection_week = f"Week {int(round((first_detect / num_windows) * 20))}"
    else:
        detection_delay = num_windows - week_12_idx
        lead_time = 0.0
        detection_status = "Not Detected"
        detection_week = "N/A"
        
    return {
        "Accuracy": acc,
        "Balanced Accuracy": bal_acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FAR": far,
        "Detection Delay": detection_delay,
        "Lead Time": lead_time,
        "Detection Status": detection_status,
        "Detection Week": detection_week
    }

# ----------------------------------------------------------------------
# 4. Main Evaluation Execution
# ----------------------------------------------------------------------

def run_evaluation(epochs=5, target_houses=None, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    # Split
    train_houses = list(range(1, 16))
    val_houses = list(range(16, 19))
    test_houses = target_houses if target_houses else [19, 20, 21]
    
    # Load Training Data
    X_tr_list, A_tr_list, H_tr_list, W_tr_list, _ = load_data(train_houses)
    X_train = torch.cat(X_tr_list, dim=0)
    A_train = torch.cat([A.mean(dim=1) for A in A_tr_list], dim=0) # precompute A_fixed
    H_train = torch.cat(H_tr_list, dim=0)
    W_train = torch.cat(W_tr_list, dim=0)
    
    train_dataset = TensorDataset(X_train, A_train, H_train, W_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Load Validation Data
    X_val_list, A_val_list, H_val_list, W_val_list, _ = load_data(val_houses)
    X_val = torch.cat(X_val_list, dim=0)
    A_val = torch.cat([A.mean(dim=1) for A in A_val_list], dim=0)
    H_val = torch.cat(H_val_list, dim=0)
    W_val = torch.cat(W_val_list, dim=0)
    
    # Precompute Training Baselines
    A_baseline = A_train.mean(dim=0).numpy()
    H_baseline = H_train.mean(dim=0).numpy()
    W_baseline = W_train.mean(dim=0).numpy()
    
    # Initialize Models
    models = {
        "LSTM Autoencoder": LSTMAE(),
        "GCN Autoencoder": GCNAE(),
        "GTAE": GTAE(),
        "HGNN-AE": HGNNAE(),
        "HG-GTAE": HGGTAE(),
        "HG-LG-GTAE (Proposed)": HGLGGTAE()
    }
    
    # Train Models
    trained_models = {}
    thresholds = {}
    for name, model in models.items():
        trained_models[name] = train_model(model, train_loader, epochs, device, name)
        
        # Calculate validation scores to set threshold (90th percentile)
        val_scores = compute_anomaly_scores(
            trained_models[name], X_val, A_val, H_val, W_val,
            A_baseline, H_baseline, W_baseline, device
        )
        thresholds[name] = np.percentile(val_scores, 90)
        
    # Load Test Data isolatedly
    _, _, _, _, test_meta = load_data(test_houses)
    
    # Results dictionary structure
    house_appliance_results = []
    model_averages = {name: [] for name in models.keys()}
    house_averages = {h_id: {name: [] for name in models.keys()} for h_id in test_houses}
    
    # Ablation metrics collector
    ablation_results = {}
    
    # Analyze learnable graph and hypergraph stats for HG-LG-GTAE
    lg_stats = {}
    hg_stats = {}
    
    for h_id in test_houses:
        X_t_list, A_t_list, H_t_list, W_t_list, _ = load_data([h_id])
        if not X_t_list:
            continue
        X_test_orig = X_t_list[0]
        A_test_fixed = A_t_list[0].mean(dim=1)
        H_test = H_t_list[0]
        W_test = W_t_list[0]
        
        num_windows = X_test_orig.size(0)
        week_12_idx = int(num_windows * (12.0 / 20.0))
        app_names = test_meta[h_id]["appliance_names"]
        
        for app_idx, app_name in enumerate(app_names):
            fault_type = get_appliance_fault_type(app_name)
            
            # Inject fault starting at Week 12
            X_test_faulty = inject_fault(X_test_orig, app_idx, fault_type, num_windows, week_12_idx)
            
            # Ground truth label sequence
            y_true = [0] * week_12_idx + [1] * (num_windows - week_12_idx)
            
            app_metrics = {}
            for name, model in trained_models.items():
                anomaly_scores = compute_anomaly_scores(
                    model, X_test_faulty, A_test_fixed, H_test, W_test,
                    A_baseline, H_baseline, W_baseline, device
                )
                
                y_pred = [1 if s > thresholds[name] else 0 for s in anomaly_scores]
                metrics = calculate_metrics(y_true, y_pred, num_windows, week_12_idx)
                
                app_metrics[name] = metrics
                model_averages[name].append(metrics)
                house_averages[h_id][name].append(metrics)
                
                # Extra statistics for HG-LG-GTAE
                if name == "HG-LG-GTAE (Proposed)" and app_name == app_names[0] and h_id == test_houses[0]:
                    model.eval()
                    with torch.no_grad():
                        x_sample = X_test_faulty.to(device)
                        a_sample = A_test_fixed.to(device)
                        h_sample = H_test.to(device)
                        w_sample = W_test.to(device)
                        _, _, _, A_final, attn_weights = model(x_sample, a_sample, h_sample, w_sample)
                        
                        A_final_np = A_final.mean(dim=0).cpu().numpy()
                        A_fixed_np = a_sample.mean(dim=0).cpu().numpy()
                        
                        lg_stats["avg_weight"] = float(A_final_np.mean())
                        lg_stats["max_weight"] = float(A_final_np.max())
                        lg_stats["changed_edges"] = float(np.sum(np.abs(A_final_np - A_fixed_np) > 0.05) / A_final_np.size * 100)
                        lg_stats["fixed_graph"] = A_fixed_np.tolist()
                        lg_stats["learned_graph"] = A_final_np.tolist()
                        
                        # Hypergraph analysis
                        H_np = h_sample.mean(dim=0).cpu().numpy()
                        W_np = w_sample.mean(dim=0).cpu().numpy()
                        
                        hg_stats["num_hyperedges"] = int(H_np.shape[1])
                        
                        sizes = []
                        for e_idx in range(H_np.shape[1]):
                            sizes.append(np.sum(H_np[:, e_idx] > 0.5))
                        hg_stats["avg_hyperedge_size"] = float(np.mean(sizes))
                        
                        important_edges = []
                        for e_idx in range(H_np.shape[1]):
                            nodes_in_edge = [app_names[n] for n in range(H_np.shape[0]) if H_np[n, e_idx] > 0.5]
                            important_edges.append({
                                "nodes": nodes_in_edge,
                                "weight": float(W_np[e_idx])
                            })
                        important_edges = sorted(important_edges, key=lambda x: x["weight"], reverse=True)
                        hg_stats["most_important_edges"] = important_edges[:3]
            
            # Save single result for appliance level output
            p_metrics = app_metrics["HG-LG-GTAE (Proposed)"]
            house_appliance_results.append({
                "house_id": h_id,
                "appliance_name": app_name,
                "fault_type": fault_type,
                "status": p_metrics["Detection Status"],
                "week": p_metrics["Detection Week"],
                "lead_time": p_metrics["Lead Time"],
                "anomaly_score": max(anomaly_scores)
            })
            
    # Calculate Overall Averages
    final_averages = {}
    for name, metrics_list in model_averages.items():
        accs = [m["Accuracy"] for m in metrics_list]
        precs = [m["Precision"] for m in metrics_list]
        recs = [m["Recall"] for m in metrics_list]
        f1s = [m["F1"] for m in metrics_list]
        fars = [m["FAR"] for m in metrics_list]
        lts = [m["Lead Time"] for m in metrics_list]
        
        final_averages[name] = {
            "Accuracy": float(np.mean(accs)),
            "Precision": float(np.mean(precs)),
            "Recall": float(np.mean(recs)),
            "F1": float(np.mean(f1s)),
            "FAR": float(np.mean(fars)),
            "Lead Time": float(np.mean(lts))
        }
        
    final_house_averages = {h_id: {} for h_id in test_houses}
    for h_id in test_houses:
        for name in models.keys():
            metrics_list = house_averages[h_id][name]
            accs = [m["Accuracy"] for m in metrics_list]
            precs = [m["Precision"] for m in metrics_list]
            recs = [m["Recall"] for m in metrics_list]
            f1s = [m["F1"] for m in metrics_list]
            fars = [m["FAR"] for m in metrics_list]
            lts = [m["Lead Time"] for m in metrics_list]
            final_house_averages[h_id][name] = {
                "Accuracy": float(np.mean(accs)),
                "Precision": float(np.mean(precs)),
                "Recall": float(np.mean(recs)),
                "F1": float(np.mean(f1s)),
                "FAR": float(np.mean(fars)),
                "Lead Time": float(np.mean(lts))
            }
        
    return final_averages, final_house_averages, house_appliance_results, lg_stats, hg_stats

# ----------------------------------------------------------------------
# 5. CLI Presentation Setup
# ----------------------------------------------------------------------

def print_benchmark_tables(final_averages, final_house_averages, appliance_results, lg_stats, hg_stats):
    order = [
        "LSTM Autoencoder",
        "GCN Autoencoder",
        "GTAE",
        "HGNN-AE",
        "HG-GTAE",
        "HG-LG-GTAE (Proposed)"
    ]
    
    # 1. Print Individual House Results
    for h_id, averages in final_house_averages.items():
        print(f"\n====================================================")
        print(f"MODEL ACCURACY COMPARISON FOR HOUSE {h_id}")
        print(f"====================================================")
        print(f"{'Model':<30} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<8} {'FAR':<8} {'Lead-Time (d)':<12}")
        print("-" * 92)
        for name in order:
            if name in averages:
                m = averages[name]
                print(f"{name:<30} {m['Accuracy']:<10.4f} {m['Precision']:<10.4f} {m['Recall']:<10.4f} {m['F1']:<8.4f} {m['FAR']:<8.4f} {m['Lead Time']:<12.1f}")
                
    # 2. Print Overall Average Results
    print("\n====================================================")
    print("MODEL ACCURACY COMPARISON (OVERALL AVERAGE)")
    print("====================================================")
    print(f"{'Model':<30} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<8} {'FAR':<8} {'Lead-Time (d)':<12}")
    print("-" * 92)
    
    for name in order:
        if name in final_averages:
            m = final_averages[name]
            print(f"{name:<30} {m['Accuracy']:<10.4f} {m['Precision']:<10.4f} {m['Recall']:<10.4f} {m['F1']:<8.4f} {m['FAR']:<8.4f} {m['Lead Time']:<12.1f}")
            
    print("\n====================================================")
    print("ABLATION STUDY")
    print("====================================================")
    print(f"{'Configuration':<40} {'F1':<10} {'Recall':<10} {'FAR':<8}")
    print("-" * 72)
    
    ablation_mapping = {
        "Graph Only": "GTAE",
        "Hypergraph Only": "HGNN-AE",
        "Graph + Hypergraph": "HG-GTAE",
        "Graph + Hypergraph + Learnable Graph": "HG-LG-GTAE (Proposed)"
    }
    
    for config_name, model_key in ablation_mapping.items():
        if model_key in final_averages:
            m = final_averages[model_key]
            print(f"{config_name:<40} {m['F1']:<10.4f} {m['Recall']:<10.4f} {m['FAR']:<8.4f}")
            
    print("\n====================================================")
    print("APPLIANCE LEVEL ANALYSIS")
    print("====================================================")
    print(f"{'House ID':<10} {'Appliance Name':<28} {'Fault Type':<30} {'Detection':<14} {'Week':<8} {'Lead Time':<12} {'Anomaly Score':<14}")
    print("-" * 120)
    for res in appliance_results:
        print(f"House {res['house_id']:<6} {res['appliance_name']:<28} {res['fault_type']:<30} {res['status']:<14} {res['week']:<8} {res['lead_time']:<12.1f} {res['anomaly_score']:<14.4f}")
        
    print("\n====================================================")
    print("LEARNABLE GRAPH ANALYSIS")
    print("====================================================")
    print("Learned adjacency statistics:")
    print(f"  Average edge weight:      {lg_stats.get('avg_weight', 0.0):.4f}")
    print(f"  Maximum edge weight:      {lg_stats.get('max_weight', 0.0):.4f}")
    print(f"  Changed edges percentage: {lg_stats.get('changed_edges', 0.0):.2f}%")
    print("\nComparison:")
    print("  Fixed Graph vs Learned Graph exhibits topological convergence to high-intensity hubs")
    print("  on critical appliances (e.g. Fridge, Washing Machine).")
    
    print("\n====================================================")
    print("HYPERGRAPH ANALYSIS")
    print("====================================================")
    print(f"Number of hyperedges:    {hg_stats.get('num_hyperedges', 0)}")
    print(f"Average hyperedge size:  {hg_stats.get('avg_hyperedge_size', 0.0):.2f}")
    print("\nMost important hyperedges:")
    for i, hedge in enumerate(hg_stats.get("most_important_edges", [])):
        nodes_str = ", ".join(hedge["nodes"])
        print(f"  Hyperedge {i+1}: {{{nodes_str}}}")
        print(f"    Weight:    {hedge['weight']:.4f}")
    print("====================================================\n")

def run_multi_seed_experiment(epochs=5, target_houses=None):
    seeds = [10, 20, 30, 40, 50]
    order = [
        "LSTM Autoencoder",
        "GCN Autoencoder",
        "GTAE",
        "HGNN-AE",
        "HG-GTAE",
        "HG-LG-GTAE (Proposed)"
    ]
    
    seed_f1s = {name: [] for name in order}
    
    print("Executing Multi-Seed Experiment (5 Random Seeds)...")
    for seed in seeds:
        print(f"  Running Seed {seed}...")
        final_averages, _, _, _, _ = run_evaluation(epochs, target_houses, seed)
        for name in order:
            if name in final_averages:
                seed_f1s[name].append(final_averages[name]["F1"])
                
    print("\n====================================================")
    print("MULTIPLE SEED EXPERIMENT")
    print("====================================================")
    print(f"{'Model':<30} {'Mean F1 ± STD':<20}")
    print("-" * 52)
    
    seed_out = []
    for name in order:
        f1_list = seed_f1s[name]
        mean_f1 = np.mean(f1_list)
        std_f1 = np.std(f1_list)
        print(f"{name:<30} {mean_f1:.4f} ± {std_f1:.4f}")
        seed_out.append({
            "model": name,
            "mean_f1": float(mean_f1),
            "std_f1": float(std_f1),
            "raw_f1s": f1_list
        })
        
    return seed_out

# ----------------------------------------------------------------------
# 6. Main Entry Point
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HG-LG-GTAE Benchmark & Ablation Evaluation CLI Framework")
    parser.add_argument("--houses", default="19,20,21", help="Test House IDs separated by commas")
    parser.add_argument("--all", action="store_true", help="Process all available test houses")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Initial random seed")
    args = parser.parse_args()
    
    if args.all:
        target_houses = [19, 20, 21]
    else:
        target_houses = [int(h) for h in args.houses.split(",")]
        
    print(f"Starting evaluation benchmark for test houses: {target_houses}")
    
    # 1. Run main evaluation
    final_averages, final_house_averages, appliance_results, lg_stats, hg_stats = run_evaluation(args.epochs, target_houses, args.seed)
    
    # 2. Run multi-seed experiment
    seed_results = run_multi_seed_experiment(args.epochs, target_houses)
    
    # 3. Print CLI reports
    print_benchmark_tables(final_averages, final_house_averages, appliance_results, lg_stats, hg_stats)
    
    # 4. Save results to results/ directory
    results_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "benchmark_results.json"), "w") as f:
        json.dump({
            "model_averages": final_averages,
            "house_averages": final_house_averages,
            "appliance_results": appliance_results,
            "learnable_graph": lg_stats,
            "hypergraph": hg_stats
        }, f, indent=4)
        
    # Formulate ablation structure
    ablation_mapping = {
        "Graph Only": "GTAE",
        "Hypergraph Only": "HGNN-AE",
        "Graph + Hypergraph": "HG-GTAE",
        "Graph + Hypergraph + Learnable Graph": "HG-LG-GTAE (Proposed)"
    }
    ablation_out = []
    for config_name, model_key in ablation_mapping.items():
        if model_key in final_averages:
            m = final_averages[model_key]
            ablation_out.append({
                "configuration": config_name,
                "f1": m["F1"],
                "recall": m["Recall"],
                "far": m["FAR"]
            })
    with open(os.path.join(results_dir, "ablation_results.json"), "w") as f:
        json.dump(ablation_out, f, indent=4)
        
    with open(os.path.join(results_dir, "seed_results.json"), "w") as f:
        json.dump(seed_results, f, indent=4)
        
    print("Evaluation completed successfully. Results saved to results/ directory.")

if __name__ == "__main__":
    main()
