import os
import re
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import components
from graph_transformer import GraphTransformerAutoencoder
from fault_injector import inject_fridge_cycle_drift, inject_washing_machine_micro_spikes

def print_ascii_bar_chart(labels, values, title):
    """Fallback utility to draw a text-based bar chart in the console."""
    print(f"\n=== {title} ===")
    if not values:
        return
    max_val = max(values)
    max_len = 30
    for label, val in zip(labels, values):
        bar_len = int((val / max_val) * max_len) if max_val > 0 else 0
        bar = "#" * bar_len + "-" * (max_len - bar_len)
        print(f"{label:<25} | {bar} | {val:.4f}")

def train_and_evaluate(graphs_pt_path, output_dir, epochs=15, batch_size=16, lr=0.001):
    print(f"\n=== Training Graph Transformer Autoencoder (GTAE) ===")
    print(f"Loading Graph Dataset: {graphs_pt_path}")
    
    # Load dataset
    data = torch.load(graphs_pt_path, weights_only=False)
    house_id = data["house_id"]
    appliance_names = data["appliance_names"]
    X = data["X"] # [Num_Windows, 9, W, 2]
    A = data["A"] # [Num_Windows, 9, 9]
    
    num_windows = X.size(0)
    num_nodes = len(appliance_names)
    sequence_length = X.size(2)
    node_features = X.size(3)
    
    print(f"Dataset summary: {num_windows} windows, {num_nodes} nodes, sequence length {sequence_length}")
    
    # 1. Temporal Train/Test Split (80% train, 20% test to avoid leakage)
    split_idx = int(num_windows * 0.80)
    X_train, X_test = X[:split_idx], X[split_idx:]
    A_train, A_test = A[:split_idx], A[split_idx:]
    
    print(f"Train set: {len(X_train)} windows | Test set: {len(X_test)} windows")
    
    # Setup DataLoaders
    train_dataset = TensorDataset(X_train, A_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Instantiate Model
    model = GraphTransformerAutoencoder(
        sequence_length=sequence_length,
        num_nodes=num_nodes,
        node_features=node_features,
        embed_dim=64,
        num_heads=4
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss_fn = nn.MSELoss()
    bce_loss_fn = nn.BCELoss()
    
    # 3. Training Loop
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_x_loss = 0.0
        epoch_a_loss = 0.0
        
        for batch_X, batch_A in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            X_recon, A_recon = model(batch_X, batch_A)
            
            # Loss computation
            loss_x = mse_loss_fn(X_recon, batch_X)
            loss_a = bce_loss_fn(A_recon, batch_A)
            
            # Total loss (weighted sum)
            loss = loss_x + 0.2 * loss_a
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_X.size(0)
            epoch_x_loss += loss_x.item() * batch_X.size(0)
            epoch_a_loss += loss_a.item() * batch_X.size(0)
            
        epoch_loss /= len(X_train)
        epoch_x_loss /= len(X_train)
        epoch_a_loss /= len(X_train)
        
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {epoch_loss:.5f} [X_Loss: {epoch_x_loss:.5f}, A_Loss: {epoch_a_loss:.5f}]")
            
    # Save model
    model_path = os.path.join(output_dir, f"House_{house_id}_GTAE.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # 4. Inject Faults into Test Set
    # We will inject Fridge Cycle Drift and Washing Machine Micro-spikes
    print("\n--- Simulating Hardware Degradation (Fault Injection) ---")
    X_test_faulty, fridge_nodes = inject_fridge_cycle_drift(X_test, appliance_names, drift_ratio=0.20)
    X_test_faulty, wm_nodes = inject_washing_machine_micro_spikes(X_test_faulty, appliance_names, spike_magnitude=0.15)
    
    injected_nodes = set(fridge_nodes + wm_nodes)
    
    # 5. Evaluate Normal vs Faulty Test Sets
    model.eval()
    
    # Helper to calculate node-level reconstruction error
    def evaluate_reconstruction_error(X_in, A_in):
        errors_per_node = []
        with torch.no_grad():
            X_recon, _ = model(X_in, A_in)
            
            # Compute node-level MSE: mean over time [W] and features [2]
            # Shape of X: [Batch, Nodes, W, 2]
            for i in range(num_nodes):
                node_orig = X_in[:, i, :, :]
                node_recon = X_recon[:, i, :, :]
                node_mse = torch.mean((node_orig - node_recon) ** 2, dim=[1, 2]).numpy()
                errors_per_node.append(node_mse) # List of shape [Nodes] each element is [Batch]
                
        # Transpose to shape [Batch, Nodes] and take average over batch
        errors_per_node = np.stack(errors_per_node, axis=1) # [Batch, Nodes]
        mean_errors = np.mean(errors_per_node, axis=0) # [Nodes]
        return mean_errors, errors_per_node

    normal_node_errors, _ = evaluate_reconstruction_error(X_test, A_test)
    faulty_node_errors, _ = evaluate_reconstruction_error(X_test_faulty, A_test)
    
    # 6. Report Findings
    print("\n=== Anomaly Detection Performance (Structural Drift Analysis) ===")
    print(f"{'Appliance':<25} | {'Normal Error':<12} | {'Faulty Error':<12} | {'Drift Ratio':<12} | {'Status':<12}")
    print("-" * 75)
    
    detection_labels = []
    normal_errors_list = []
    faulty_errors_list = []
    
    for i, app in enumerate(appliance_names):
        norm_err = normal_node_errors[i]
        f_err = faulty_node_errors[i]
        ratio = f_err / (norm_err + 1e-9)
        
        is_injected = i in injected_nodes
        status = "ALERT (FAULT)" if (ratio > 2.0 and is_injected) else ("OK" if not is_injected else "MISS (UNDETECTED)")
        if is_injected and ratio <= 2.0:
            status = "WARNING"
            
        print(f"{app:<25} | {norm_err:<12.5f} | {f_err:<12.5f} | {ratio:<12.2f} | {status:<12}")
        
        detection_labels.append(app)
        normal_errors_list.append(norm_err)
        faulty_errors_list.append(f_err)
        
    # Draw text-based charts
    print_ascii_bar_chart(detection_labels, normal_errors_list, "Normal Test Set Node Reconstruction Error")
    print_ascii_bar_chart(detection_labels, faulty_errors_list, "Fault-Injected Test Set Node Reconstruction Error")
    
    # 7. Attempt Matplotlib Plotting
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x_indices = np.arange(len(appliance_names))
        bar_width = 0.35
        
        ax.bar(x_indices - bar_width/2, normal_errors_list, bar_width, label='Normal Baseline', color='#58a6ff')
        ax.bar(x_indices + bar_width/2, faulty_errors_list, bar_width, label='Fault-Injected (Drift)', color='#d2a8ff')
        
        ax.set_ylabel('Mean Reconstruction Error (MSE)', fontname='sans-serif', fontsize=11)
        ax.set_title(f'Appliance Degradation Anomaly Detection (House {house_id})', fontname='sans-serif', fontsize=13, fontweight='bold')
        ax.set_xticks(x_indices)
        ax.set_xticklabels(appliance_names, rotation=35, ha='right')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Highlight anomalies
        for i, ratio in enumerate(faulty_node_errors / (normal_node_errors + 1e-9)):
            if ratio > 2.0 and i in injected_nodes:
                ax.text(i, faulty_node_errors[i] + (max(faulty_errors_list)*0.02), '⚠️ FAULT', ha='center', color='#ff7b72', fontweight='bold', fontsize=9)
                
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"House_{house_id}_Anomaly_Detection.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\nSuccessfully saved anomaly score comparison chart to {plot_path}")
        
    except ImportError:
        print("\nNote: matplotlib not installed. Skipping graphical chart generation (ASCII chart displayed above).")
        
    mean_adj = A.mean(dim=0).numpy().tolist()
    drift_ratios = []
    statuses = []
    for i, app in enumerate(appliance_names):
        norm_err = float(normal_errors_list[i])
        f_err = float(faulty_errors_list[i])
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

    report_dict = {
        "house_id": int(house_id),
        "appliance_names": appliance_names,
        "mean_adjacency": mean_adj,
        "normal_errors": [float(x) for x in normal_errors_list],
        "faulty_errors": [float(x) for x in faulty_errors_list],
        "drift_ratios": drift_ratios,
        "statuses": statuses,
        "injected_nodes": [int(n) for n in injected_nodes]
    }
    return report_dict

def main():
    parser = argparse.ArgumentParser(description="Train Graph Transformer and Detect Anomalies (Phase 3).")
    parser.add_argument("--input-file", help="Path to a specific constructed Graphs .pt file or raw/processed CSV file")
    parser.add_argument("--input-dir", default="3_processed_outputs", help="Directory containing constructed Graphs")
    parser.add_argument("--output-dir", default="3_processed_outputs", help="Directory to save models and evaluations")
    parser.add_argument("--house-id", type=int, help="Optional specific House ID to process")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    args = parser.parse_args()
    
    import json
    
    reports = []
    
    if args.input_file:
        pt_path = args.input_file
        if args.input_file.endswith('.csv'):
            match = re.search(r'House_?(\d+)', os.path.basename(args.input_file), re.IGNORECASE)
            house_id = int(match.group(1)) if match else 99
            
            from refit_processor import process_house
            from graph_builder import build_house_graphs
            
            print(f"Preprocessing raw CSV for House {house_id}...")
            process_house(args.input_file, args.output_dir, house_id)
            
            processed_csv = os.path.join(args.output_dir, f"House_{house_id}_Processed.csv")
            metadata_json = os.path.join(args.output_dir, f"House_{house_id}_Metadata.json")
            
            print("Building graph datasets from processed CSV...")
            pt_path = build_house_graphs(processed_csv, metadata_json, args.output_dir)
            if not pt_path or not os.path.exists(pt_path):
                raise ValueError("Graph builder did not produce graphs.pt file.")
                
        rep = train_and_evaluate(pt_path, args.output_dir, args.epochs)
        reports.append(rep)
    else:
        if not os.path.exists(args.input_dir):
            print(f"Error: Input directory '{args.input_dir}' does not exist.")
            return
            
        pt_files = [f for f in os.listdir(args.input_dir) if f.endswith('_Graphs.pt')]
        if not pt_files:
            print(f"No constructed Graph .pt files found in '{args.input_dir}'. Run graph_builder.py first.")
            return
            
        pt_files.sort()
        
        for file in pt_files:
            # Extract house id
            match = re.search(r'House_(\d+)', file)
            if not match:
                continue
            house_id = int(match.group(1))
            
            if args.house_id and house_id != args.house_id:
                continue
                
            pt_path = os.path.join(args.input_dir, file)
            try:
                rep = train_and_evaluate(pt_path, args.output_dir, args.epochs)
                reports.append(rep)
            except Exception as e:
                print(f"Error executing ML pipeline for House {house_id}: {e}")
                import traceback
                traceback.print_exc()
                
    # Save the consolidated reports as JSON
    if reports:
        report_path = os.path.join(args.output_dir, "Anomaly_Report.json")
        
        # If running a subset of houses, we might want to merge with an existing report if it exists
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r') as f:
                    existing_reports = json.load(f)
                # Merge: overwrite existing houses with new runs, keep others
                existing_dict = {r["house_id"]: r for r in existing_reports}
                for r in reports:
                    existing_dict[r["house_id"]] = r
                reports = sorted(list(existing_dict.values()), key=lambda x: x["house_id"])
            except Exception as e:
                print(f"Failed to read existing report, overwriting: {e}")
                
        with open(report_path, 'w') as f:
            json.dump(reports, f, indent=4)
        print(f"\nSaved consolidated anomaly report for {len(reports)} houses to {report_path}")

if __name__ == "__main__":
    main()
