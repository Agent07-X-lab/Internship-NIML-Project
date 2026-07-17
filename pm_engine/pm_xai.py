import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_feature_saliency(model, X_window, A_window):
    """
    Computes input feature saliency by taking the gradient of the feature
    reconstruction error with respect to the input node features.
    
    X_window: [1, N, W, F]
    A_window: [1, N, N] or [1, 4, N, N]
    """
    model.eval()
    
    # We need gradients on the input tensor
    X_var = X_window.clone().detach().requires_grad_(True)
    A_var = A_window.clone().detach()
    
    # If A has 4 channels, select the Jaccard channel [0] if model expects 2D
    # Our GTAE model expects A to have shape [Batch, N, N]
    if A_var.dim() == 4 and A_var.size(1) == 4:
        A_model = A_var[:, 0, :, :]
    else:
        A_model = A_var
        
    X_recon, _ = model(X_var, A_model)
    
    # Compute reconstruction loss
    loss = torch.mean((X_var - X_recon) ** 2)
    
    # Backward pass to accumulate gradients
    loss.backward()
    
    # Saliency is the absolute value of gradients
    saliency = torch.abs(X_var.grad).squeeze(0).numpy() # [N, W, F]
    
    # Aggregate over time steps (W) to get a single importance score per feature per node
    # [N, F] -> node-level feature importance
    feature_importance = np.mean(saliency, axis=1)
    
    return feature_importance

def compute_gated_attention(model, X_window, A_window):
    """
    Reconstructs the multi-head gated attention weights for the two Graph Transformer layers.
    This avoids having to modify the model class code and replicates the forward math exactly.
    """
    model.eval()
    with torch.no_grad():
        # Encode sequences to initial embeddings
        batch_size = X_window.size(0)
        num_nodes = model.num_nodes
        seq_len = model.sequence_length
        node_feats = model.node_features
        embed_dim = model.embed_dim
        
        X_flat = X_window.view(batch_size * num_nodes, seq_len, node_feats)
        h = model.encoder(X_flat)
        h = h.view(batch_size, num_nodes, embed_dim)
        h = model.enc_norm(h)
        
        # Add node embeddings
        node_ids = torch.arange(num_nodes, device=X_window.device).unsqueeze(0).expand(batch_size, -1)
        h = h + model.node_embed(node_ids)
        
        # Prepare A (Jaccard channel)
        if A_window.dim() == 4 and A_window.size(1) == 4:
            A_model = A_window[:, 0, :, :]
        else:
            A_model = A_window
            
        # Replicate GT Layer 1 Attention
        attn_weights = []
        for gt_layer in [model.gt1, model.gt2]:
            # q, k, v projections
            q = gt_layer.q_proj(h).view(batch_size, num_nodes, gt_layer.num_heads, gt_layer.head_dim).transpose(1, 2)
            k = gt_layer.k_proj(h).view(batch_size, num_nodes, gt_layer.num_heads, gt_layer.head_dim).transpose(1, 2)
            
            # Scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / (gt_layer.head_dim ** 0.5)
            attn_probs = F.softmax(scores, dim=-1)
            
            # Gated attention
            A_expanded = A_model.unsqueeze(1)
            gated_attn = attn_probs * A_expanded
            gated_attn = gated_attn / (gated_attn.sum(dim=-1, keepdim=True) + 1e-9)
            
            attn_weights.append(gated_attn.squeeze(0).numpy()) # [Heads, N, N]
            
            # Continue forward path to feed next layer
            v = gt_layer.v_proj(h).view(batch_size, num_nodes, gt_layer.num_heads, gt_layer.head_dim).transpose(1, 2)
            out = torch.matmul(gated_attn, v)
            out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, embed_dim)
            out = gt_layer.out_proj(out)
            h = gt_layer.norm1(h + gt_layer.dropout(out))
            ffn_out = gt_layer.ffn(h)
            h = gt_layer.norm2(h + gt_layer.dropout(ffn_out))
            
        return attn_weights # List of length 2, each element [Heads, N, N]

def track_graph_evolution(A_current, A_mean, appliance_names, threshold=0.08):
    """
    Identifies edges that were added, removed, or changed in strength
    between the current graph window and the normal historical baseline.
    """
    num_nodes = len(appliance_names)
    
    # Calculate difference matrix
    diff = A_current - A_mean
    
    added_edges = []
    removed_edges = []
    strength_changes = []
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            val_diff = diff[i, j]
            current_val = A_current[i, j]
            mean_val = A_mean[i, j]
            
            edge_name = f"{appliance_names[i]} ↔ {appliance_names[j]}"
            
            if val_diff > threshold:
                added_edges.append({
                    "name": edge_name,
                    "from_node": i,
                    "to_node": j,
                    "baseline": float(mean_val),
                    "current": float(current_val),
                    "increase": float(val_diff)
                })
            elif val_diff < -threshold:
                removed_edges.append({
                    "name": edge_name,
                    "from_node": i,
                    "to_node": j,
                    "baseline": float(mean_val),
                    "current": float(current_val),
                    "decrease": float(abs(val_diff))
                })
                
            if abs(val_diff) > 0.01:
                strength_changes.append({
                    "name": edge_name,
                    "from_node": i,
                    "to_node": j,
                    "baseline": float(mean_val),
                    "current": float(current_val),
                    "delta": float(val_diff)
                })
                
    # Sort strength changes by absolute delta
    strength_changes.sort(key=lambda x: abs(x["delta"]), reverse=True)
    
    return added_edges, removed_edges, strength_changes[:10]

def explain_node_anomaly(X_window, X_recon, node_idx, appliance_names, baseline_summary=None):
    """
    Explains why a specific node is anomalous by decomposing its signal changes.
    """
    power = X_window[0, node_idx, :, 0].numpy()
    state = X_window[0, node_idx, :, 1].numpy()
    
    recon_power = X_recon[0, node_idx, :, 0].numpy()
    
    # Compute reconstruction error (MSE) for this node
    node_mse = float(np.mean((power - recon_power) ** 2))
    
    # Calculate operating parameters
    mean_active_power = float(np.mean(power[state > 0.5]) if np.sum(state > 0.5) > 0 else 0.0)
    duty_cycle = float(np.sum(state > 0.5) / len(state))
    
    # Calculate consecutive active cycle durations
    is_on = (state > 0.5).astype(int)
    diff = np.diff(np.concatenate(([0], is_on, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    durations = (ends - starts).tolist()
    mean_duration = float(np.mean(durations) if durations else 0.0)
    
    # Power spikes (variance)
    power_var = float(np.var(power[state > 0.5]) if np.sum(state > 0.5) > 0 else 0.0)
    
    # Ratios compared to baseline summary if provided
    power_ratio = 1.0
    duration_ratio = 1.0
    var_ratio = 1.0
    duty_ratio = 1.0
    
    app_name = appliance_names[node_idx]
    if baseline_summary and app_name in baseline_summary:
        base = baseline_summary[app_name]
        
        # Avoid division by zero
        base_power = base.get("mean_power", 0.0)
        base_pct = base.get("active_pct", 0.0)
        
        if base_power > 0:
            # Re-scale back to Watts for comparison
            power_ratio = mean_active_power / base_power if mean_active_power > 0 else 1.0
            
        if base_pct > 0:
            duty_ratio = (duty_cycle * 100.0) / base_pct
            
        # Placeholders for durations/variance ratio comparisons
        duration_ratio = mean_duration / 35.0 if mean_duration > 0 else 1.0 # default estimated baseline steps
        var_ratio = power_var / 0.02 if power_var > 0 else 1.0
        
    # Compile textual reasons
    reasons = []
    if power_ratio > 1.12:
        reasons.append(f"Mean active power draw increased by {int((power_ratio-1.0)*100)}%")
    elif power_ratio < 0.88:
        reasons.append(f"Mean active power draw decreased by {int((1.0-power_ratio)*100)}%")
        
    if duty_ratio > 1.15:
        reasons.append(f"Appliance duty cycle increased by {int((duty_ratio-1.0)*100)}%")
    elif duty_ratio < 0.85:
        reasons.append(f"Appliance duty cycle decreased by {int((1.0-duty_ratio)*100)}%")
        
    if duration_ratio > 1.20:
        reasons.append(f"Average cycle running duration extended by {int((duration_ratio-1.0)*100)}%")
        
    if var_ratio > 1.25:
        reasons.append("High-frequency power spikes/fluctuations detected during operation")
        
    if not reasons:
        reasons.append("Appliance behavioral relationship with other devices drifted")
        
    return {
        "node_name": app_name,
        "reconstruction_error": node_mse,
        "power_ratio": power_ratio,
        "duration_ratio": duration_ratio,
        "variance_ratio": var_ratio,
        "reasons": reasons
    }
