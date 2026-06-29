import os
import re
import argparse
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from refit_processor import process_house, clean_name, get_threshold
from graph_builder import build_house_graphs
from train_eval import print_ascii_bar_chart
from graph_transformer import GraphTransformerAutoencoder
from fault_injector import inject_fridge_cycle_drift, inject_washing_machine_micro_spikes

def run_prediction(input_path, output_dir, epochs=15):
    # Determine house_id from filename or default to 99 (custom)
    match = re.search(r'House_?(\d+)', os.path.basename(input_path), re.IGNORECASE)
    house_id = int(match.group(1)) if match else 99
    
    print(f"\n--- Running Prediction Pipeline for House {house_id} ---")
    print(f"Target CSV file: {input_path}")
    
    # 1. Preprocess the raw CSV
    process_house(input_path, output_dir, house_id)
    
    processed_csv = os.path.join(output_dir, f"House_{house_id}_Processed.csv")
    metadata_json = os.path.join(output_dir, f"House_{house_id}_Metadata.json")
    
    # 2. Build graph sliding windows
    graphs_pt = build_house_graphs(processed_csv, metadata_json, output_dir)
    if not graphs_pt or not os.path.exists(graphs_pt):
        raise ValueError("Graph builder did not produce graphs.pt file.")
    
    # 3. Load graphs
    data = torch.load(graphs_pt, weights_only=False)
    appliance_names = data["appliance_names"]
    X = data["X"]
    A = data["A"]
    
    num_windows = X.size(0)
    num_nodes = len(appliance_names)
    sequence_length = X.size(2)
    node_features = X.size(3)
    
    # Instantiate GTAE Model
    model = GraphTransformerAutoencoder(
        sequence_length=sequence_length,
        num_nodes=num_nodes,
        node_features=node_features,
        embed_dim=64,
        num_heads=4
    )
    
    model_path = os.path.join(output_dir, f"House_{house_id}_GTAE.pth")
    model_exists = os.path.exists(model_path)
    
    # Determine if we should train
    if model_exists:
        print(f"Pre-trained model found at {model_path}. Loading weights...")
        try:
            model.load_state_dict(torch.load(model_path, weights_only=False))
        except Exception as e:
            print(f"Warning: Failed to load pre-trained weights: {e}. Falling back to training.")
            model_exists = False
            
    if not model_exists and epochs > 0:
        print(f"No valid pre-trained model found. Training model for {epochs} epochs...")
        from torch.utils.data import DataLoader, TensorDataset
        
        # 80/20 split
        split_idx = int(num_windows * 0.8)
        if split_idx == 0:
            split_idx = num_windows
            
        X_train = X[:split_idx]
        A_train = A[:split_idx]
        
        train_loader = DataLoader(TensorDataset(X_train, A_train), batch_size=16, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        mse_loss_fn = nn.MSELoss()
        bce_loss_fn = nn.BCELoss()
        
        model.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for batch_X, batch_A in train_loader:
                optimizer.zero_grad()
                X_recon, A_recon = model(batch_X, batch_A)
                loss = mse_loss_fn(X_recon, batch_X) + 0.2 * bce_loss_fn(A_recon, batch_A)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
            epoch_loss /= len(X_train)
            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {epoch_loss:.5f}")
                
        # Save newly trained model
        torch.save(model.state_dict(), model_path)
        print(f"Trained model saved to {model_path}")

    # 4. Inject Faults into Test Set
    split_idx = int(num_windows * 0.8)
    if split_idx == 0:
        split_idx = num_windows
        
    X_test = X[split_idx:]
    A_test = A[split_idx:]
    if len(X_test) == 0:
        X_test = X
        A_test = A
        
    X_test_faulty, fridge_nodes = inject_fridge_cycle_drift(X_test, appliance_names, drift_ratio=0.20)
    X_test_faulty, wm_nodes = inject_washing_machine_micro_spikes(X_test_faulty, appliance_names, spike_magnitude=0.15)
    injected_nodes = set(fridge_nodes + wm_nodes)
    
    # 5. Evaluate Normal vs Faulty
    model.eval()
    
    # Helper to calculate node-level reconstruction error
    def evaluate_reconstruction_error_local(X_in, A_in):
        errors_per_node = []
        with torch.no_grad():
            X_recon, _ = model(X_in, A_in)
            for i in range(num_nodes):
                node_orig = X_in[:, i, :, :]
                node_recon = X_recon[:, i, :, :]
                node_mse = torch.mean((node_orig - node_recon) ** 2, dim=[1, 2]).numpy()
                errors_per_node.append(node_mse)
        errors_per_node = np.stack(errors_per_node, axis=1)
        mean_errors = np.mean(errors_per_node, axis=0)
        return mean_errors
        
    normal_node_errors = evaluate_reconstruction_error_local(X_test, A_test)
    faulty_node_errors = evaluate_reconstruction_error_local(X_test_faulty, A_test)
    
    # Compile metrics
    normal_errors_list = normal_node_errors.tolist()
    faulty_errors_list = faulty_node_errors.tolist()
    
    drift_ratios = []
    statuses = []
    for i, app in enumerate(appliance_names):
        norm_err = normal_errors_list[i]
        f_err = faulty_errors_list[i]
        ratio = f_err / (norm_err + 1e-9)
        drift_ratios.append(ratio)
        
        is_injected = i in injected_nodes
        if is_injected:
            if ratio > 1.15:
                status = "ALERT (FAULT)"
            else:
                status = "WARNING"
        else:
            if ratio > 1.15:
                status = "WARNING"
            else:
                status = "OK"
        statuses.append(status)
        
    # Output to console
    print("\n=== Prediction Diagnostics ===")
    print(f"{'Appliance':<25} | {'Normal Error':<12} | {'Faulty Error':<12} | {'Drift Ratio':<12} | {'Status':<12}")
    print("-" * 75)
    for i, app in enumerate(appliance_names):
        print(f"{app:<25} | {normal_errors_list[i]:<12.5f} | {faulty_errors_list[i]:<12.5f} | {drift_ratios[i]:<12.2f} | {statuses[i]:<12}")
        
    print_ascii_bar_chart(appliance_names, normal_errors_list, "Normal Test Set Node Reconstruction Error")
    print_ascii_bar_chart(appliance_names, faulty_errors_list, "Fault-Injected Test Set Node Reconstruction Error")
    
    # Average Jaccard co-occurrence
    mean_adj = A.mean(dim=0).numpy().tolist()
    
    # Load metadata and summaries to build enriched data response
    # Metadata
    metadata = {}
    if os.path.exists(metadata_json):
        with open(metadata_json, 'r') as f:
            metadata = json.load(f)
            
    # Summary
    summary_csv = os.path.join(output_dir, f"House_{house_id}_Summary.csv")
    summaries = {}
    if os.path.exists(summary_csv):
        with open(summary_csv, 'r') as f:
            reader = csv_reader = pd.read_csv(summary_csv).to_dict(orient='records')
            for row in reader:
                app_name = row["Appliance"]
                summaries[app_name] = {
                    "active_pct": float(row["Active Percentage (%)"]),
                    "mean_power": float(row["Mean Power Active (W)"]),
                    "threshold": float(row["Threshold_W"])
                }
    
    # Return JSON serializable output report dict
    report = {
        "house_id": house_id,
        "appliance_names": appliance_names,
        "mean_adjacency": mean_adj,
        "normal_errors": normal_errors_list,
        "faulty_errors": faulty_errors_list,
        "drift_ratios": drift_ratios,
        "statuses": statuses,
        "injected_nodes": [int(n) for n in injected_nodes],
        "metadata": metadata,
        "summaries": summaries
    }
    return report

def main():
    parser = argparse.ArgumentParser(description="Predict early faults for any house CSV.")
    parser.add_argument("--input-file", required=True, help="Path to raw house CSV file")
    parser.add_argument("--output-dir", default="3_processed_outputs", help="Directory for temporary outputs")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs if model is not pre-trained")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return
        
    report = run_prediction(args.input_file, args.output_dir, args.epochs)
    
    # Save this prediction report to a special JSON file
    out_report_path = os.path.join(args.output_dir, f"Prediction_House_{report['house_id']}.json")
    with open(out_report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"\nSaved prediction report to {out_report_path}")

if __name__ == "__main__":
    main()
