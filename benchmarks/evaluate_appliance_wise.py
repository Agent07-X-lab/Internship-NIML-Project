import os
import sys
import json
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Setup project root and subdirectories in sys.path to resolve imports cleanly
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB_DIRS = [
    ROOT_DIR,
    os.path.join(ROOT_DIR, 'data_processing'),
    os.path.join(ROOT_DIR, 'models'),
    os.path.join(ROOT_DIR, 'pm_engine'),
    os.path.join(ROOT_DIR, 'web_dashboard'),
    os.path.join(ROOT_DIR, 'benchmarks')
]
for sd in SUB_DIRS:
    if sd not in sys.path:
        sys.path.insert(0, sd)

# Import models and helpers from the existing files
from evaluate_benchmark_cli import (
    LSTMAE, GCNAE, GTAE, HGNNAE, HGGTAE,
    load_data, inject_fault, get_appliance_fault_type, compute_anomaly_scores,
    train_model
)
from graph_transformer import HGLGGTAE

def calculate_detailed_metrics(y_true, y_pred):
    """
    Calculates detailed metrics: Accuracy, Precision, Recall, and F1 Score.
    """
    TP = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    TN = sum((t == 0 and p == 0) for t, p in zip(y_true, y_pred))
    FP = sum((t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    FN = sum((t == 1 and p == 0) for t, p in zip(y_true, y_pred))
    
    total = len(y_true)
    accuracy = (TP + TN) / total if total > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN
    }

def main():
    parser = argparse.ArgumentParser(description="Appliance-Wise Performance Evaluation for All Models")
    parser.add_argument("--test_houses", default="19,20,21", help="Comma-separated house IDs to evaluate")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--fast", action="store_true", help="If set, trains on a small subset of houses (Houses 1-2) for speed")
    parser.add_argument("--device", default="auto", help="Device to run on (cuda, cpu, or auto)")
    args = parser.parse_args()
    
    # 1. Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # 2. Select Train, Val, Test splits
    if args.fast:
        print("Fast mode enabled. Using reduced training and validation sets.")
        train_houses = [1, 2]
        val_houses = [3]
    else:
        train_houses = list(range(1, 16))
        val_houses = list(range(16, 19))
        
    test_houses = [int(h) for h in args.test_houses.split(",")]
    
    print(f"Training houses: {train_houses}")
    print(f"Validation houses: {val_houses}")
    print(f"Testing houses: {test_houses}")
    print(f"Training epochs: {args.epochs}")
    
    # 3. Load Datasets
    print("\nLoading Training Data...")
    X_tr_list, A_tr_list, H_tr_list, W_tr_list, _ = load_data(train_houses)
    if not X_tr_list:
        print("Error: No training data found. Please run the data processing pipeline first.")
        sys.exit(1)
        
    X_train = torch.cat(X_tr_list, dim=0)
    A_train = torch.cat([A.mean(dim=1) for A in A_tr_list], dim=0)
    H_train = torch.cat(H_tr_list, dim=0)
    W_train = torch.cat(W_tr_list, dim=0)
    
    train_dataset = TensorDataset(X_train, A_train, H_train, W_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print("Loading Validation Data...")
    X_val_list, A_val_list, H_val_list, W_val_list, _ = load_data(val_houses)
    X_val = torch.cat(X_val_list, dim=0)
    A_val = torch.cat([A.mean(dim=1) for A in A_val_list], dim=0)
    H_val = torch.cat(H_val_list, dim=0)
    W_val = torch.cat(W_val_list, dim=0)
    
    # Precompute Training Baselines for anomaly scoring
    A_baseline = A_train.mean(dim=0).numpy()
    H_baseline = H_train.mean(dim=0).numpy()
    W_baseline = W_train.mean(dim=0).numpy()
    
    # 4. Initialize Models
    models = {
        "LSTM Autoencoder": LSTMAE(),
        "GCN Autoencoder": GCNAE(),
        "GTAE": GTAE(),
        "HGNN-AE": HGNNAE(),
        "HG-GTAE": HGGTAE(),
        "HG-LG-GTAE (Proposed)": HGLGGTAE()
    }
    
    # 5. Train Models and Calculate Thresholds
    trained_models = {}
    thresholds = {}
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        start_time = time.time()
        trained_model = train_model(model, train_loader, args.epochs, device, name)
        duration = time.time() - start_time
        print(f"Finished training {name} in {duration:.1f}s")
        
        # Calculate validation scores to set threshold (90th percentile)
        print(f"Calculating threshold for {name} on validation data...")
        val_scores = compute_anomaly_scores(
            trained_model, X_val, A_val, H_val, W_val,
            A_baseline, H_baseline, W_baseline, device
        )
        thresholds[name] = np.percentile(val_scores, 90)
        print(f"Threshold set to: {thresholds[name]:.6f}")
        trained_models[name] = trained_model
        
    # 6. Load Test Data
    print("\nLoading Test Data...")
    _, _, _, _, test_meta = load_data(test_houses)
    
    # Dictionary to store structured metrics
    # Format: { model_name: { house_id: { appliance_name: { metric: value } } } }
    results = {name: {} for name in models.keys()}
    
    # 7. Evaluate Appliance-Wise
    for h_id in test_houses:
        print(f"\nEvaluating House {h_id}...")
        X_t_list, A_t_list, H_t_list, W_t_list, _ = load_data([h_id])
        if not X_t_list:
            print(f"Skipping House {h_id} (No data found).")
            continue
            
        X_test_orig = X_t_list[0]
        A_test_fixed = A_t_list[0].mean(dim=1)
        H_test = H_t_list[0]
        W_test = W_t_list[0]
        
        num_windows = X_test_orig.size(0)
        week_12_idx = int(num_windows * (12.0 / 20.0))
        app_names = test_meta[h_id]["appliance_names"]
        
        # For each appliance, inject fault and evaluate
        for app_idx, app_name in enumerate(app_names):
            fault_type = get_appliance_fault_type(app_name)
            print(f"  Appliance: {app_name:<25} | Fault Type: {fault_type}")
            
            # Inject fault starting at Week 12
            X_test_faulty = inject_fault(X_test_orig, app_idx, fault_type, num_windows, week_12_idx)
            
            # Ground truth label sequence: 0 before Week 12 (normal), 1 after Week 12 (faulty)
            y_true = [0] * week_12_idx + [1] * (num_windows - week_12_idx)
            
            for name, model in trained_models.items():
                anomaly_scores = compute_anomaly_scores(
                    model, X_test_faulty, A_test_fixed, H_test, W_test,
                    A_baseline, H_baseline, W_baseline, device
                )
                
                y_pred = [1 if s > thresholds[name] else 0 for s in anomaly_scores]
                metrics = calculate_detailed_metrics(y_true, y_pred)
                
                if h_id not in results[name]:
                    results[name][h_id] = {}
                results[name][h_id][app_name] = metrics
                
    # 8. Print Results Tables
    print("\n" + "="*100)
    print("APPLIANCE-WISE METRICS COMPARISON TABLE")
    print("="*100)
    
    # Save results to json
    results_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "appliance_wise_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved detailed results to {json_path}")
    
    # Format and print table for each model
    for model_name in models.keys():
        print(f"\n>>> Model: {model_name} <<<")
        print(f"{'House':<8} | {'Appliance':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
        print("-"*80)
        
        # Collect averages for this model
        model_accs = []
        model_precs = []
        model_recs = []
        model_f1s = []
        
        for h_id in test_houses:
            if h_id not in results[model_name]:
                continue
            for app_name, metrics in results[model_name][h_id].items():
                print(f"House {h_id:<2} | {app_name:<25} | {metrics['Accuracy']:<10.4f} | {metrics['Precision']:<10.4f} | {metrics['Recall']:<10.4f} | {metrics['F1']:<10.4f}")
                model_accs.append(metrics['Accuracy'])
                model_precs.append(metrics['Precision'])
                model_recs.append(metrics['Recall'])
                model_f1s.append(metrics['F1'])
                
        if model_accs:
            print("-"*80)
            print(f"{'OVERALL AVERAGE':<34} | {np.mean(model_accs):<10.4f} | {np.mean(model_precs):<10.4f} | {np.mean(model_recs):<10.4f} | {np.mean(model_f1s):<10.4f}")
        print("="*80)

if __name__ == "__main__":
    main()
