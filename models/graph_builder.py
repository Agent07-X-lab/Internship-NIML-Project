import os
import re
import json
import argparse
import pandas as pd
import numpy as np
import torch

def build_house_graphs(processed_csv_path, metadata_json_path, output_dir, window_size=256, stride=32):
    print(f"\n--- Constructing Graphs from {processed_csv_path} ---")
    
    # Load metadata to get appliance list
    with open(metadata_json_path, 'r') as f:
        metadata = json.load(f)
        
    house_id = metadata["House"]
    # Get sorted list of cleaned appliance names from the threshold logs to ensure consistent node indexing
    appliance_names = sorted(list(metadata["Thresholds"].keys()))
    print(f"House {house_id} monitored appliances ({len(appliance_names)}): {appliance_names}")
    
    # Load standardized time series
    df = pd.read_csv(processed_csv_path, parse_dates=True, index_col=0)
    print(f"Loaded timeseries. Shape: {df.shape}")
    
    # Extract max power for each appliance to perform min-max normalization
    max_powers = {}
    for app in appliance_names:
        max_power = df[app].max()
        max_powers[app] = max_power if max_power > 0 else 1.0
        
    # Prepare list to hold window data
    windows_X = []
    windows_A = []
    window_timestamps = []
    
    n_steps = len(df)
    for start_idx in range(0, n_steps - window_size + 1, stride):
        end_idx = start_idx + window_size
        window_df = df.iloc[start_idx:end_idx]
        
        # 1. Construct Node Features (Shape: [9, W, 2])
        # For each node (appliance), features are: [Normalized Power, Binary State]
        node_features = []
        app_states = [] # To compute adjacency matrix
        
        for app in appliance_names:
            power_vals = window_df[app].values
            state_vals = window_df[f"{app}_ON"].values
            
            # Normalize power by its dataset-wide maximum
            normalized_power = power_vals / max_powers[app]
            
            # Combine power and state: shape [W, 2]
            app_feat = np.stack([normalized_power, state_vals], axis=1)
            node_features.append(app_feat)
            app_states.append(state_vals)
            
        # Stack all nodes: shape [9, W, 2]
        X = np.stack(node_features, axis=0)
        
        # 2. Construct Adjacency Matrix A (Shape: [9, 9])
        A = np.zeros((len(appliance_names), len(appliance_names)))
        
        # Compute pairwise co-occurrences using Jaccard Similarity of binary states
        for i in range(len(appliance_names)):
            state_i = app_states[i]
            sum_i = np.sum(state_i)
            
            for j in range(i, len(appliance_names)):
                state_j = app_states[j]
                sum_j = np.sum(state_j)
                
                # Intersection: both active
                intersection = np.sum(state_i * state_j)
                # Union
                union = sum_i + sum_j - intersection
                
                if union > 0:
                    jaccard = intersection / union
                else:
                    jaccard = 0.0
                    
                A[i, j] = jaccard
                A[j, i] = jaccard # Symmetric matrix
                
        # Ensure self-loops are set to 1.0 (nodes retain self identity)
        for i in range(len(appliance_names)):
            A[i, i] = 1.0
            
        windows_X.append(X)
        windows_A.append(A)
        
        # Record start timestamp of this window
        window_timestamps.append(window_df.index[0])
        
    if not windows_X:
        print(f"Warning: Timeseries too short for window size {window_size}. No graphs built.")
        return None
        
    # Convert lists to PyTorch Tensors
    tensor_X = torch.tensor(np.array(windows_X), dtype=torch.float32)
    tensor_A = torch.tensor(np.array(windows_A), dtype=torch.float32)
    
    # Save as PyTorch dataset dict
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"House_{house_id}_Graphs.pt")
    
    dataset_dict = {
        "house_id": house_id,
        "appliance_names": appliance_names,
        "max_powers": max_powers,
        "X": tensor_X,
        "A": tensor_A,
        "timestamps": window_timestamps
    }
    
    torch.save(dataset_dict, out_path)
    print(f"Successfully constructed and saved {len(windows_X)} graph objects to {out_path}")
    print(f"Tensor X Shape: {tensor_X.shape} (Windows x Nodes x TimeSteps x Features)")
    print(f"Tensor A Shape: {tensor_A.shape} (Windows x Nodes x Nodes)")
    
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Construct Behavioral Networks (Phase 2).")
    parser.add_argument("--input-dir", default="3_processed_outputs", help="Directory containing processed CSV/JSON files")
    parser.add_argument("--output-dir", default="3_processed_outputs", help="Directory to save the Graph objects")
    parser.add_argument("--window-size", type=int, default=256, help="Sliding window size (time steps)")
    parser.add_argument("--stride", type=int, default=32, help="Sliding window stride (time steps)")
    parser.add_argument("--house-id", type=int, help="Optional specific House ID to construct graphs for")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return
        
    # Find processed CSV files
    csv_files = [f for f in os.listdir(args.input_dir) if f.endswith('_Processed.csv')]
    if not csv_files:
        print(f"No processed house CSV files found in '{args.input_dir}'.")
        return
        
    csv_files.sort()
    
    for file in csv_files:
        # Extract house id
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
            build_house_graphs(csv_path, meta_path, args.output_dir, args.window_size, args.stride)
        except Exception as e:
            print(f"Error building graphs for House {house_id}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
