import os
import re
import json
import time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Core component imports
from pm_dynamic_graph import build_dynamic_graphs, calculate_graph_drift
from pm_fault_injector import inject_gradual_faults
from pm_analytics import (
    calculate_health_index,
    estimate_fault_severity,
    predict_rul,
    perform_root_cause_analysis,
    get_maintenance_recommendation
)
from pm_xai import calculate_feature_saliency, track_graph_evolution, explain_node_anomaly
from pm_report_exporter import export_json_report, export_html_report

# Model architecture import
from graph_transformer import GraphTransformerAutoencoder

def run_predictive_maintenance_pipeline(input_path, output_dir="3_processed_outputs", epochs=15):
    # 1. Preprocess raw CSV to processed CSV if input is CSV
    match = re.search(r'House_?(\d+)', os.path.basename(input_path), re.IGNORECASE)
    house_id = int(match.group(1)) if match else 99
    
    print(f"\n=======================================================")
    print(f"Executing Predictive Maintenance Pipeline for House {house_id}")
    print(f"=======================================================")
    
    # Run original refit preprocessor
    from refit_processor import process_house
    process_house(input_path, output_dir, house_id)
    
    processed_csv = os.path.join(output_dir, f"House_{house_id}_Processed.csv")
    metadata_json = os.path.join(output_dir, f"House_{house_id}_Metadata.json")
    
    # 2. Build dynamic multi-feature graphs
    graphs_pt = build_dynamic_graphs(processed_csv, metadata_json, output_dir)
    
    # 3. Load graph tensors
    data = torch.load(graphs_pt, weights_only=False)
    appliance_names = data["appliance_names"]
    X = data["X"] # [Windows, N, W, 9]
    A = data["A"] # [Windows, 4, N, N]
    timestamps = data["timestamps"]
    mean_A = data["mean_A"]
    
    num_windows = X.size(0)
    num_nodes = len(appliance_names)
    sequence_length = X.size(2)
    node_features = X.size(3) # Should be 9 features
    
    # Setup GTAE model
    # Note: X has 9 features now. Our GraphTransformerAutoencoder needs to be initialized with 9 features.
    model = GraphTransformerAutoencoder(
        sequence_length=sequence_length,
        num_nodes=num_nodes,
        node_features=node_features,
        embed_dim=64,
        num_heads=4
    )
    
    model_path = os.path.join(output_dir, f"House_{house_id}_GTAE_PM.pth")
    model_exists = os.path.exists(model_path)
    
    # 80/20 train/test split
    split_idx = int(num_windows * 0.8)
    if split_idx == 0:
        split_idx = num_windows
        
    X_train = X[:split_idx]
    A_train = A[:split_idx]
    X_test = X[split_idx:]
    A_test = A[split_idx:]
    
    if len(X_test) == 0:
        X_test = X
        A_test = A
        
    # GTAE expects adjacency matrix with 2 dimensions [B, N, N]
    # We feed Jaccard Similarity (channel 0) as the primary attention gate
    A_train_gate = A_train[:, 0, :, :]
    A_test_gate = A_test[:, 0, :, :]
    
    if model_exists:
        print(f"Pre-trained PM model weights found at {model_path}. Loading...")
        try:
            model.load_state_dict(torch.load(model_path, weights_only=False))
        except Exception as e:
            print(f"Failed to load weights: {e}. Re-training model...")
            model_exists = False
            
    if not model_exists and epochs > 0:
        print(f"Training PM model for {epochs} epochs...")
        train_loader = DataLoader(TensorDataset(X_train, A_train_gate), batch_size=16, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        mse_fn = nn.MSELoss()
        bce_fn = nn.BCELoss()
        
        model.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for batch_X, batch_A in train_loader:
                optimizer.zero_grad()
                X_recon, A_recon = model(batch_X, batch_A)
                loss = mse_fn(X_recon, batch_X) + 0.2 * bce_fn(A_recon, batch_A)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
            epoch_loss /= len(X_train)
            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_loss:.5f}")
                
        torch.save(model.state_dict(), model_path)
        print(f"Saved trained PM model weights to {model_path}")
        
    # 4. Inject Gradual Degradation Faults into the Test Set
    X_test_faulty, injected_nodes_raw = inject_gradual_faults(X_test, appliance_names)
    
    # Filter injected_nodes to only include nodes that are actually graph-connected.
    # Isolated appliances (max off-diagonal Jaccard < 0.05) are excluded so they
    # are never colored red in the graph visualization.
    CONNECTIVITY_THRESHOLD = 0.05  # Jaccard similarity minimum for a meaningful edge
    mean_A_numpy = mean_A.numpy()  # shape [4, N, N], channel 0 = Jaccard
    jaccard_matrix = mean_A_numpy[0]  # [N, N]
    
    connected_nodes = set()
    for i in range(len(appliance_names)):
        for j in range(len(appliance_names)):
            if i != j and jaccard_matrix[i][j] >= CONNECTIVITY_THRESHOLD:
                connected_nodes.add(i)
                connected_nodes.add(j)
    
    # Only flag a node as injected/degraded if it participates in the graph
    injected_nodes = [n for n in injected_nodes_raw if n in connected_nodes]
    
    isolated_nodes = [i for i in range(len(appliance_names)) if i not in connected_nodes]
    if isolated_nodes:
        iso_names = [appliance_names[i] for i in isolated_nodes]
        print(f"  Graph-isolated appliances (excluded from degradation display): {iso_names}")
    print(f"  Graph-connected injected nodes: {injected_nodes} of {injected_nodes_raw} raw")
    
    # 5. Evaluate Normal vs Degraded behavior
    model.eval()
    
    # Standard baseline thresholds for reconstruction errors per node
    # Calculated on normal training set to establish the clean threshold parameters
    baseline_thresholds = {}
    with torch.no_grad():
        X_train_recon, _ = model(X_train, A_train_gate)
        for i, name in enumerate(appliance_names):
            node_mse = torch.mean((X_train[:, i, :, :] - X_train_recon[:, i, :, :]) ** 2).item()
            # Threshold set to 3 standard deviations above mean or a minimum ceiling
            baseline_thresholds[name] = max(0.005, node_mse * 1.5)
            
    # Calculate health index over sliding windows to forecast RUL trends
    # For RUL, we compute health history for each node across windows
    health_histories = {name: [] for name in appliance_names}
    
    # Map node indices to icons
    icon_mappings = {
        "fridge": "fa-snowflake", "freezer": "fa-snowflake", "refrigerator": "fa-snowflake",
        "washing": "fa-soap", "washer": "fa-soap", "dryer": "fa-soap",
        "dishwasher": "fa-glass-water", "kettle": "fa-mug-hot",
        "television": "fa-tv", "tv": "fa-tv", "computer": "fa-tv",
        "microwave": "fa-temperature-arrow-up", "oven": "fa-kitchen-set",
        "lamp": "fa-lightbulb", "light": "fa-lightbulb", "router": "fa-wifi"
    }
    
    def get_icon(name):
        n_lower = name.lower()
        for key, val in icon_mappings.items():
            if key in n_lower:
                return val
        return "fa-plug"
        
    appliance_analytics = []
    
    # Track graph-level drift
    mean_A_test = A_test_gate.mean(dim=0).numpy()
    final_drift_score, _ = calculate_graph_drift(A_test_gate[-1].numpy(), mean_A_test)
    
    # Evaluate final window for report generation
    with torch.no_grad():
        X_test_recon, _ = model(X_test, A_test_gate)
        X_faulty_recon, _ = model(X_test_faulty, A_test_gate)
        
        # Calculate health history over all test windows for RUL estimation
        for w_idx in range(len(X_test)):
            for i, name in enumerate(appliance_names):
                win_mse = torch.mean((X_test_faulty[w_idx, i, :, :] - X_faulty_recon[w_idx, i, :, :]) ** 2).item()
                h_val = calculate_health_index(win_mse, baseline_thresholds[name])
                health_histories[name].append(h_val)
                
        # Final window results
        for i, name in enumerate(appliance_names):
            norm_mse = torch.mean((X_test[:, i, :, :] - X_test_recon[:, i, :, :]) ** 2).item()
            faulty_mse = torch.mean((X_test_faulty[-1, i, :, :] - X_faulty_recon[-1, i, :, :]) ** 2).item()
            
            drift_ratio = faulty_mse / (norm_mse + 1e-9)
            severity = estimate_fault_severity(drift_ratio)
            
            # Predict RUL
            # Scaling window steps to simulated days
            stride_days = max(0.5, 140.0 / num_windows) # 20 weeks = 140 days
            rul_days = predict_rul(health_histories[name], num_windows, stride_days)
            
            # Feature analysis for Root Cause Analysis
            power_val = X_test_faulty[-1, i, :, 0].numpy()
            base_power_val = X_test[-1, i, :, 0].numpy()
            power_ratio = np.mean(power_val) / (np.mean(base_power_val) + 1e-9)
            
            # Calculate active durations ratios
            state_val = X_test_faulty[-1, i, :, 1].numpy()
            base_state_val = X_test[-1, i, :, 1].numpy()
            duration_ratio = np.sum(state_val) / (np.sum(base_state_val) + 1e-9)
            
            # Spike variance ratio
            spike_var_ratio = np.var(power_val) / (np.var(base_power_val) + 1e-9)
            
            # Root Cause & Recommendations
            root_cause, confidence = perform_root_cause_analysis(name, power_ratio, duration_ratio, spike_var_ratio)
            
            # If healthy, override cause
            if severity == "Healthy":
                root_cause = "Normal Operation"
                confidence = 0.98
                recommendations = ["General preventative maintenance check"]
            else:
                recommendations = get_maintenance_recommendation(root_cause)
                
            appliance_analytics.append({
                "name": name,
                "icon": get_icon(name),
                "health": health_histories[name][-1],
                "severity": severity,
                "confidence": float(confidence),
                "rul": int(rul_days),
                "root_cause": root_cause,
                "recommendations": recommendations,
                "drift_ratio": float(drift_ratio),
                "normal_error": float(norm_mse),
                "faulty_error": float(faulty_mse)
            })
            
    # Explain graph evolution
    added, removed, changes = track_graph_evolution(A_test_gate[-1].numpy(), mean_A[0], appliance_names)
            
    # 7. Compile Report File data
    report_out = {
        "house_id": house_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "graph_drift": float(final_drift_score),
        "appliances": appliance_analytics,
        "appliance_names": appliance_names,
        "mean_adjacency": mean_A.tolist(),
        "injected_nodes": injected_nodes,           # Only graph-connected fault nodes
        "isolated_nodes": sorted(list(set(range(len(appliance_names))) - connected_nodes)),  # Truly disconnected
        "connected_nodes": sorted(list(connected_nodes)),                                    # Participates in graph
        "connectivity_threshold": CONNECTIVITY_THRESHOLD,
        "drift_ratios": [app["drift_ratio"] for app in appliance_analytics],
        "statuses": [app["severity"] for app in appliance_analytics],
        "health_history": {name: hist for name, hist in health_histories.items()},
        "added_edges": added,
        "removed_edges": removed,
        "edge_changes": changes
    }
    
    # Save offline audit files
    export_json_report(report_out, os.path.join(output_dir, f"PM_Report_House_{house_id}.json"))
    export_html_report(report_out, os.path.join(output_dir, f"PM_Report_House_{house_id}.html"))
    
    # Save main consolidated audit log
    master_report_path = os.path.join(output_dir, "Anomaly_Report_PM.json")
    master_reports = []
    if os.path.exists(master_report_path):
        try:
            with open(master_report_path, 'r') as f:
                master_reports = json.load(f)
            existing_dict = {r["house_id"]: r for r in master_reports}
            existing_dict[house_id] = report_out
            master_reports = sorted(list(existing_dict.values()), key=lambda x: x["house_id"])
        except Exception:
            master_reports = [report_out]
    else:
        master_reports = [report_out]
        
    with open(master_report_path, 'w') as f:
        json.dump(master_reports, f, indent=4)
        
    # 8. Compile Web Dashboard (Predictive_Maintenance_Dashboard.html)
    template_path = "dashboard_pm_template.html"
    with open(template_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
        
    report_json_str = json.dumps(master_reports, indent=4)
    
    html_content = html_content.replace("__ENRICHED_REPORT_DATA__", report_json_str)
    
    root_out_path = "Predictive_Maintenance_Dashboard.html"
    with open(root_out_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Generated Root Predictive Maintenance Dashboard: {root_out_path}")
    
    # Also save copy in outputs folder
    with open(os.path.join(output_dir, "Predictive_Maintenance_Dashboard.html"), 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"Pipeline finished! Dashboard successfully updated.")
    return report_out

def main():
    parser = argparse.ArgumentParser(description="Run Graph Transformer-based Predictive Maintenance Pipeline.")
    parser.add_argument("--input-file", help="Path to raw Smart Home House CSV file (backwards compatibility)")
    parser.add_argument("--house", default="1", help="House ID (e.g. '1', a list '1,2,3', or 'all' to batch process)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--output-dir", default="3_processed_outputs", help="Output directory")
    args = parser.parse_args()
    
    raw_dir = "1_raw_data"
    
    if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
        print("No raw house data found. Generating synthetic data...")
        import generate_synthetic_refit
        generate_synthetic_refit.main()
        
    if args.house.lower() == "all":
        # Find all house files in raw_dir
        files = [f for f in os.listdir(raw_dir) if f.endswith('.csv') and 'house' in f.lower()]
        def get_house_num(filename):
            match = re.search(r'House_?(\d+)', filename, re.IGNORECASE)
            return int(match.group(1)) if match else 99
        files.sort(key=get_house_num)
        
        print(f"Batch processing {len(files)} houses...")
        for f in files:
            input_path = os.path.join(raw_dir, f)
            try:
                run_predictive_maintenance_pipeline(input_path, args.output_dir, args.epochs)
            except Exception as e:
                print(f"Error processing {f}: {e}")
    elif "," in args.house:
        house_ids = [h.strip() for h in args.house.split(",")]
        for h_id in house_ids:
            input_path = os.path.join(raw_dir, f"CLEAN_House{h_id}.csv")
            if not os.path.exists(input_path):
                matches = [f for f in os.listdir(raw_dir) if f.endswith('.csv') and f"House{h_id}" in f.replace("_", "")]
                if matches:
                    input_path = os.path.join(raw_dir, matches[0])
                else:
                    print(f"Warning: House {h_id} file not found.")
                    continue
            try:
                run_predictive_maintenance_pipeline(input_path, args.output_dir, args.epochs)
            except Exception as e:
                print(f"Error processing House {h_id}: {e}")
    else:
        # Single house processing
        input_file = args.input_file
        if not input_file:
            h_id = args.house
            input_file = os.path.join(raw_dir, f"CLEAN_House{h_id}.csv")
            if not os.path.exists(input_file):
                matches = [f for f in os.listdir(raw_dir) if f.endswith('.csv') and f"House{h_id}" in f.replace("_", "")]
                if matches:
                    input_file = os.path.join(raw_dir, matches[0])
                else:
                    # Fallback to default first CSV file
                    csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv') and 'house' in f.lower()]
                    csv_files.sort()
                    if csv_files:
                           input_file = os.path.join(raw_dir, csv_files[0])
                    else:
                           print("Error: No raw house data CSV files found.")
                           return
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' does not exist.")
            return
        run_predictive_maintenance_pipeline(input_file, args.output_dir, args.epochs)

if __name__ == "__main__":
    main()
