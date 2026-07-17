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

import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Import components and models
from evaluate_benchmark_cli import (
    LSTMAE, GCNAE, GTAE, HGNNAE, HGGTAE,
    load_data, inject_fault, get_appliance_fault_type, compute_anomaly_scores
)
from graph_transformer import HGLGGTAE

def train_model_split(model, train_loader, epochs=5, device="cpu"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    model.train()
    
    for epoch in range(epochs):
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
                recon, a_recon, h_recon, _, _ = model(batch_x, batch_a, batch_h, batch_w)
                loss = F.mse_loss(recon, batch_x) + 0.2 * F.mse_loss(a_recon, batch_a) + 0.1 * F.mse_loss(h_recon, batch_h)
                
            loss.backward()
            optimizer.step()
            
    return model

def calculate_split_metrics(y_true, y_pred):
    TP = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    TN = sum((t == 0 and p == 0) for t, p in zip(y_true, y_pred))
    FP = sum((t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    FN = sum((t == 1 and p == 0) for t, p in zip(y_true, y_pred))
    
    total = len(y_true)
    accuracy = (TP + TN) / total if total > 0 else 0.0
    
    # Check if y_true is all zeros (Before Fault Injection)
    if all(t == 0 for t in y_true):
        if FP == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
    # Check if y_true is all ones (After Fault Injection)
    elif all(t == 1 for t in y_true):
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / total if total > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    else:
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Per-House Split (80% Train, 20% Test) before/after fault injection")
    parser.add_argument("--houses", default="19,20,21", help="House IDs to evaluate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_houses = [int(h) for h in args.houses.split(",")]
    
    print(f"Starting Per-House Split Evaluation on houses: {target_houses}")
    print(f"Split ratio: 80% Train, 20% Test. Epochs: {args.epochs}")
    
    results = {}
    
    for h_id in target_houses:
        X_list, A_list, H_list, W_list, house_meta = load_data([h_id])
        if not X_list:
            print(f"Skipping House {h_id} due to missing data.")
            continue
            
        X = X_list[0]
        A = A_list[0].mean(dim=1)
        H = H_list[0]
        W_e = W_list[0]
        app_names = house_meta[h_id]["appliance_names"]
        
        W = X.size(0)
        W_train = int(0.8 * W)
        W_test = W - W_train
        
        print(f"\nEvaluating House {h_id} (Total windows: {W}, Train: {W_train}, Test: {W_test})")
        
        # Split Data
        X_train, X_test = X[:W_train], X[W_train:]
        A_train, A_test = A[:W_train], A[W_train:]
        H_train, H_test = H[:W_train], H[W_train:]
        W_train_e, W_test_e = W_e[:W_train], W_e[W_train:]
        
        train_dataset = TensorDataset(X_train, A_train, H_train, W_train_e)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        # Baselines
        A_baseline = A_train.mean(dim=0).numpy()
        H_baseline = H_train.mean(dim=0).numpy()
        W_baseline = W_train_e.mean(dim=0).numpy()
        
        # Models
        models = {
            "LSTM Autoencoder": LSTMAE(),
            "GCN Autoencoder": GCNAE(),
            "GTAE": GTAE(),
            "HGNN-AE": HGNNAE(),
            "HG-GTAE": HGGTAE(),
            "HG-LG-GTAE (Proposed)": HGLGGTAE()
        }
        
        results[h_id] = {}
        
        for name, model in models.items():
            # 1. Train Model on 80% train data
            trained_model = train_model_split(model, train_loader, args.epochs, device)
            
            # 2. Compute training anomaly scores to get threshold (mean + 2.5 * std)
            train_scores = compute_anomaly_scores(
                trained_model, X_train, A_train, H_train, W_train_e,
                A_baseline, H_baseline, W_baseline, device
            )
            threshold = np.mean(train_scores) + 2.5 * np.std(train_scores)
            
            # 3. Test CASE A: BEFORE FAULT INJECTION (Clean Test Set)
            test_scores_clean = compute_anomaly_scores(
                trained_model, X_test, A_test, H_test, W_test_e,
                A_baseline, H_baseline, W_baseline, device
            )
            y_true_clean = [0] * W_test
            y_pred_clean = [1 if s > threshold else 0 for s in test_scores_clean]
            metrics_clean = calculate_split_metrics(y_true_clean, y_pred_clean)
            
            # 4. Test CASE B: AFTER FAULT INJECTION (Faulty Test Set)
            # Inject fault in test set for each appliance individually, then average results
            after_metrics_list = []
            for app_idx, app_name in enumerate(app_names):
                fault_type = get_appliance_fault_type(app_name)
                X_test_faulty = inject_fault(X_test, app_idx, fault_type, W_test, 0)
                
                test_scores_faulty = compute_anomaly_scores(
                    trained_model, X_test_faulty, A_test, H_test, W_test_e,
                    A_baseline, H_baseline, W_baseline, device
                )
                
                y_true_faulty = [1] * W_test
                y_pred_faulty = [1 if s > threshold else 0 for s in test_scores_faulty]
                metrics_faulty = calculate_split_metrics(y_true_faulty, y_pred_faulty)
                after_metrics_list.append(metrics_faulty)
                
            # Average metrics across all appliances
            metrics_faulty_avg = {
                "Accuracy": float(np.mean([m["Accuracy"] for m in after_metrics_list])),
                "Precision": float(np.mean([m["Precision"] for m in after_metrics_list])),
                "Recall": float(np.mean([m["Recall"] for m in after_metrics_list])),
                "F1": float(np.mean([m["F1"] for m in after_metrics_list]))
            }
            
            results[h_id][name] = {
                "Before_Fault": metrics_clean,
                "After_Fault": metrics_faulty_avg
            }
            
            print(f"  {name:<25} | Before -> Acc: {metrics_clean['Accuracy']:.4f}, F1: {metrics_clean['F1']:.4f} | After -> Acc: {metrics_faulty_avg['Accuracy']:.4f}, F1: {metrics_faulty_avg['F1']:.4f}")
            
    # Print clean formatted tables for CLI
    print("\n" + "=" * 90)
    print("FINAL PER-HOUSE SPLIT BENCHMARK RESULTS")
    print("=" * 90)
    
    order = [
        "LSTM Autoencoder",
        "GCN Autoencoder",
        "GTAE",
        "HGNN-AE",
        "HG-GTAE",
        "HG-LG-GTAE (Proposed)"
    ]
    
    for h_id in target_houses:
        if h_id not in results:
            continue
        print(f"\n>>> HOUSE {h_id} PERFORMANCE DETAILS <<<")
        print(f"{'Model':<25} | {'Metric':<10} | {'Before Fault (Clean)':<22} | {'After Fault (Faulty)':<22}")
        print("-" * 90)
        for name in order:
            b = results[h_id][name]["Before_Fault"]
            a = results[h_id][name]["After_Fault"]
            
            print(f"{name:<25} | {'Accuracy':<10} | {b['Accuracy']:<22.4f} | {a['Accuracy']:<22.4f}")
            print(f"{'':<25} | {'Precision':<10} | {b['Precision']:<22.4f} | {a['Precision']:<22.4f}")
            print(f"{'':<25} | {'Recall':<10} | {b['Recall']:<22.4f} | {a['Recall']:<22.4f}")
            print(f"{'':<25} | {'F1-Score':<10} | {b['F1']:<22.4f} | {a['F1']:<22.4f}")
            print("-" * 90)
            
    # Save split results to results directory
    results_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "per_house_split_results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nPer-house split evaluation completed. Saved to {os.path.join(results_dir, 'per_house_split_results.json')}")

if __name__ == "__main__":
    main()
