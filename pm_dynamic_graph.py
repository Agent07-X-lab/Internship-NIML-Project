import os
import re
import json
import argparse
import pandas as pd
import numpy as np
import torch

def compute_mutual_information(s1, s2):
    """Calculates Mutual Information between two binary ON/OFF state sequences."""
    n = len(s1)
    if n == 0:
        return 0.0
    
    # Joint probabilities
    p00 = np.sum((s1 == 0) & (s2 == 0)) / n
    p01 = np.sum((s1 == 0) & (s2 == 1)) / n
    p10 = np.sum((s1 == 1) & (s2 == 0)) / n
    p11 = np.sum((s1 == 1) & (s2 == 1)) / n
    
    # Marginals
    p1_0 = p00 + p01
    p1_1 = p10 + p11
    p2_0 = p00 + p10
    p2_1 = p01 + p11
    
    mi = 0.0
    joint_probs = [p00, p01, p10, p11]
    marginals_1 = [p1_0, p1_0, p1_1, p1_1]
    marginals_2 = [p2_0, p2_1, p2_0, p2_1]
    
    for jp, m1, m2 in zip(joint_probs, marginals_1, marginals_2):
        if jp > 0 and m1 > 0 and m2 > 0:
            mi += jp * np.log(jp / (m1 * m2 + 1e-9) + 1e-9)
            
    return max(0.0, mi)

def calculate_graph_drift(A_t, A_mean):
    """
    Computes graph drift score using Graph Edit Distance (GED)
    and Cosine Similarity of adjacency matrices.
    """
    # GED normalized distance
    ged = np.mean(np.abs(A_t - A_mean))
    
    # Cosine Similarity
    norm_t = np.linalg.norm(A_t)
    norm_m = np.linalg.norm(A_mean)
    if norm_t > 0 and norm_m > 0:
        cosine_sim = np.sum(A_t * A_mean) / (norm_t * norm_m)
    else:
        cosine_sim = 1.0
        
    drift_score = 1.0 - cosine_sim
    return float(drift_score), float(ged)

def build_dynamic_graphs(processed_csv_path, metadata_json_path, output_dir, window_size=256, stride=32):
    print(f"\n--- Constructing Dynamic Graphs from {processed_csv_path} ---")
    
    with open(metadata_json_path, 'r') as f:
        metadata = json.load(f)
        
    house_id = metadata["House"]
    appliance_names = sorted(list(metadata["Thresholds"].keys()))
    num_nodes = len(appliance_names)
    
    df = pd.read_csv(processed_csv_path, parse_dates=True, index_col=0)
    print(f"Loaded timeseries shape: {df.shape}")
    
    # Max powers for normalization
    max_powers = {}
    for app in appliance_names:
        max_power = df[app].max()
        max_powers[app] = max_power if max_power > 0 else 1.0
        
    windows_X = [] # Dynamic node features [Windows, N, W, F]
    windows_A = [] # Dynamic multi-edge adjacency matrices [Windows, 4, N, N]
    window_timestamps = []
    
    n_steps = len(df)
    
    # Hour and minute indices for temporal positional encoding
    hours = df.index.hour.values
    minutes = df.index.minute.values
    time_sin = np.sin(2 * np.pi * (hours + minutes / 60.0) / 24.0)
    time_cos = np.cos(2 * np.pi * (hours + minutes / 60.0) / 24.0)
    
    for start_idx in range(0, n_steps - window_size + 1, stride):
        end_idx = start_idx + window_size
        window_df = df.iloc[start_idx:end_idx]
        
        # 1. Compute Node Features (Shape: [N, W, 9])
        node_features = []
        app_states = []
        app_powers = []
        
        # Previous window error placeholder (to be updated during evaluation)
        prev_error = 0.0
        
        for app in appliance_names:
            power_vals = window_df[app].values
            state_vals = window_df[f"{app}_ON"].values
            
            app_states.append(state_vals)
            app_powers.append(power_vals)
            
            # Normalized Power
            norm_power = power_vals / max_powers[app]
            
            # Mean and Variance (running values in window, simplified as current window stats)
            mean_power = np.full(window_size, np.mean(power_vals) / max_powers[app])
            var_power = np.full(window_size, np.var(power_vals) / (max_powers[app]**2 + 1e-9))
            
            # Duty Cycle
            duty_cycle = np.full(window_size, np.sum(state_vals) / window_size)
            
            # Running Duration
            run_dur = []
            current_run = 0.0
            for val in state_vals:
                if val > 0.5:
                    current_run += 1.0
                else:
                    current_run = 0.0
                run_dur.append(current_run)
            run_dur = np.array(run_dur) / window_size # normalize
            
            # Energy Consumption (integration in Wh: Watts * hours)
            energy_val = np.cumsum(power_vals) * (8.0 / 3600.0)
            energy_norm = energy_val / (max_powers[app] * (window_size * 8.0 / 3600.0) + 1e-9)
            
            # Temporal Positional Encodings
            sin_enc = time_sin[start_idx:end_idx]
            cos_enc = time_cos[start_idx:end_idx]
            
            # Previous window error placeholder
            prev_err_feat = np.full(window_size, prev_error)
            
            # Stack features: shape [W, 9]
            feats = np.stack([
                norm_power,       # 0: Normalized Power
                state_vals,       # 1: Binary ON/OFF
                mean_power,       # 2: Mean Power
                var_power,        # 3: Power Variance
                duty_cycle,       # 4: Duty Cycle
                run_dur,          # 5: Running Duration
                energy_norm,      # 6: Energy Consumption
                sin_enc,          # 7: Temporal Sin
                cos_enc           # 8: Temporal Cos
            ], axis=1)
            
            node_features.append(feats)
            
        X = np.stack(node_features, axis=0) # [N, W, 9]
        
        # 2. Compute Adjacency Matrix A (Shape: [4, N, N])
        # Channel 0: Jaccard Similarity
        # Channel 1: Pearson Correlation
        # Channel 2: Mutual Information
        # Channel 3: Co-occurrence Frequency
        
        A = np.zeros((4, num_nodes, num_nodes))
        
        for i in range(num_nodes):
            state_i = app_states[i]
            power_i = app_powers[i]
            sum_i = np.sum(state_i)
            
            for j in range(i, num_nodes):
                state_j = app_states[j]
                power_j = app_powers[j]
                sum_j = np.sum(state_j)
                
                # --- Channel 0: Jaccard Similarity ---
                intersection = np.sum(state_i * state_j)
                union = sum_i + sum_j - intersection
                jaccard = intersection / union if union > 0 else 0.0
                
                # --- Channel 1: Pearson Correlation ---
                std_i = np.std(power_i)
                std_j = np.std(power_j)
                if std_i > 0 and std_j > 0:
                    pearson = np.corrcoef(power_i, power_j)[0, 1]
                    if np.isnan(pearson):
                        pearson = 0.0
                else:
                    pearson = 0.0
                # Scale Pearson from [-1, 1] to [0, 1] for network gate compatibility
                pearson = (pearson + 1.0) / 2.0
                
                # --- Channel 2: Mutual Information ---
                mi = compute_mutual_information(state_i, state_j)
                # Normalize MI to [0, 1] assuming max entropy log(2)
                mi_norm = min(1.0, mi / np.log(2))
                
                # --- Channel 3: Co-occurrence Frequency ---
                co_occur = np.sum((state_i > 0.5) & (state_j > 0.5)) / window_size
                
                # Assign to matrix channels
                A[0, i, j] = A[0, j, i] = jaccard
                A[1, i, j] = A[1, j, i] = pearson
                A[2, i, j] = A[2, j, i] = mi_norm
                A[3, i, j] = A[3, j, i] = co_occur
                
        # Retain self-loops = 1.0
        for idx in range(num_nodes):
            A[0, idx, idx] = 1.0
            A[1, idx, idx] = 1.0
            A[2, idx, idx] = 1.0
            A[3, idx, idx] = 1.0
            
        windows_X.append(X)
        windows_A.append(A)
        window_timestamps.append(window_df.index[0])
        
    tensor_X = torch.tensor(np.array(windows_X), dtype=torch.float32)
    tensor_A = torch.tensor(np.array(windows_A), dtype=torch.float32)
    
    # Calculate historical baseline mean adjacency matrix
    mean_A = tensor_A.mean(dim=0).numpy()
    
    # Calculate graph drift score for each window
    drift_scores = []
    ged_scores = []
    for w in range(len(windows_A)):
        # Focus Jaccard channel [0] for baseline drift analysis
        drift, ged = calculate_graph_drift(windows_A[w][0], mean_A[0])
        drift_scores.append(drift)
        ged_scores.append(ged)
        
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"House_{house_id}_Dynamic_Graphs.pt")
    
    dataset_dict = {
        "house_id": house_id,
        "appliance_names": appliance_names,
        "max_powers": max_powers,
        "X": tensor_X,
        "A": tensor_A,
        "timestamps": window_timestamps,
        "mean_A": mean_A,
        "drift_scores": drift_scores,
        "ged_scores": ged_scores
    }
    
    torch.save(dataset_dict, out_path)
    print(f"Constructed and saved {len(windows_X)} dynamic graphs to {out_path}")
    print(f"Tensor X Shape: {tensor_X.shape} (Windows x Nodes x TimeSteps x Features [9])")
    print(f"Tensor A Shape: {tensor_A.shape} (Windows x AdjacencyChannels [4] x Nodes x Nodes)")
    
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Construct Dynamic Temporal Graphs (Module 1).")
    parser.add_argument("--input-dir", default="3_processed_outputs", help="Directory containing processed CSV/JSON files")
    parser.add_argument("--output-dir", default="3_processed_outputs", help="Directory to save the Graph objects")
    parser.add_argument("--window-size", type=int, default=256, help="Sliding window size")
    parser.add_argument("--stride", type=int, default=32, help="Sliding window stride")
    parser.add_argument("--house-id", type=int, help="Optional specific House ID to construct graphs for")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return
        
    csv_files = [f for f in os.listdir(args.input_dir) if f.endswith('_Processed.csv')]
    if not csv_files:
        print(f"No processed house CSV files found in '{args.input_dir}'.")
        return
        
    csv_files.sort()
    for file in csv_files:
        match = re.search(r'House_(\d+)', file)
        if not match:
            continue
        house_id = int(match.group(1))
        
        if args.house_id and house_id != args.house_id:
            continue
            
        csv_path = os.path.join(args.input_dir, file)
        meta_path = os.path.join(args.input_dir, f"House_{house_id}_Metadata.json")
        
        if not os.path.exists(meta_path):
            print(f"Warning: Metadata file {meta_path} missing. Skipping.")
            continue
            
        try:
            build_dynamic_graphs(csv_path, meta_path, args.output_dir, args.window_size, args.stride)
        except Exception as e:
            print(f"Error building dynamic graphs for House {house_id}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
