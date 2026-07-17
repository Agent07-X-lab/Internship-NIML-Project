import copy
import json
import numpy as np
import torch

def load_config(config_path="pm_config.json"):
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception:
        # Fallback defaults if config not found
        return {
            "degradation_rates": {
                "fridge": 0.015, "freezer": 0.015, "refrigerator": 0.015,
                "washing_machine": 0.012, "washer": 0.012, "dryer": 0.012,
                "dishwasher": 0.010, "air_conditioner": 0.018, "fan": 0.008,
                "microwave": 0.005, "tv": 0.005, "kettle": 0.005
            }
        }

def inject_gradual_faults(X_tensor, appliance_names, config_path="pm_config.json"):
    """
    Injects gradual physical faults into the input tensor.
    Fault severity scales linearly with the window index in the test set.
    """
    X_anom = X_tensor.clone()
    num_windows = X_anom.size(0)
    num_nodes = len(appliance_names)
    
    config = load_config(config_path)
    rates = config.get("degradation_rates", {})
    
    # Test set is assumed to be 20 weeks. Week 12 is at 60% of the test set length.
    week_12_idx = int(num_windows * (12.0 / 20.0))
    
    injected_nodes = []
    
    for idx, name in enumerate(appliance_names):
        name_lower = name.lower()
        
        # We inject specific faults based on appliance type, mapped to the 4 requested types
        fault_type = None
        if any(k in name_lower for k in ["fridge", "freezer", "refrigerator", "ac", "air_conditioner"]):
            fault_type = "Gradual power increase"
        elif any(k in name_lower for k in ["washing", "washer", "dryer", "dishwasher"]):
            fault_type = "Increased operation duration"
        elif any(k in name_lower for k in ["microwave", "tv", "computer", "television"]):
            fault_type = "Intermittent spike"
        elif any(k in name_lower for k in ["fan", "kettle"]):
            fault_type = "Stuck ON/OFF"
            
        if fault_type is None:
            continue
            
        injected_nodes.append(idx)
        print(f"Applying fault simulation ({fault_type}) to: {name} (Node index {idx})")
        
        for w in range(num_windows):
            if w < week_12_idx:
                continue # No fault before Week 12
                
            progress = (w - week_12_idx) / (num_windows - week_12_idx - 1) if num_windows - week_12_idx > 1 else 1.0
            
            power = X_anom[w, idx, :, 0].numpy()
            state = X_anom[w, idx, :, 1].numpy()
            
            new_power = copy.deepcopy(power)
            new_state = copy.deepcopy(state)
            
            is_on = (state > 0.5).astype(int)
            diff = np.diff(np.concatenate(([0], is_on, [0])))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            
            # --- 1. Gradual power increase fault (Pnew = P(1+yt)) ---
            if fault_type == "Gradual power increase":
                gamma = 0.30 # max 30% increase
                active_mask = new_state > 0.5
                new_power[active_mask] = new_power[active_mask] * (1.0 + gamma * progress)
                
            # --- 2. Intermittent spike fault (Pnew = P + noise spikes) ---
            elif fault_type == "Intermittent spike":
                active_mask = new_state > 0.5
                # Add spikes to active periods
                if np.sum(active_mask) > 0:
                    spike_prob = 0.1 * progress
                    spikes = np.random.rand(len(power)) * 0.4
                    mask = (np.random.rand(len(power)) < spike_prob) & active_mask
                    new_power[mask] += spikes[mask]
                    
            # --- 3. Increased operation duration fault ---
            elif fault_type == "Increased operation duration":
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
                        
            # --- 4. Stuck ON/OFF fault ---
            elif fault_type == "Stuck ON/OFF":
                # Stuck ON for progress > 0.5
                if progress > 0.5:
                    new_state[:] = 1.0
                    avg_active = np.mean(power[is_on == 1]) if np.sum(is_on) > 0 else 0.8
                    new_power[:] = avg_active
            
            new_power = np.clip(new_power, 0.0, 1.5)
            
            X_anom[w, idx, :, 0] = torch.tensor(new_power, dtype=torch.float32)
            X_anom[w, idx, :, 1] = torch.tensor(new_state, dtype=torch.float32)
            
            if X_anom.size(3) > 2:
                mean_p = np.mean(new_power)
                var_p = np.var(new_power)
                duty_c = np.sum(new_state) / len(new_state)
                
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
                
                X_anom[w, idx, :, 2] = torch.tensor(mean_p, dtype=torch.float32)
                X_anom[w, idx, :, 3] = torch.tensor(var_p, dtype=torch.float32)
                X_anom[w, idx, :, 4] = torch.tensor(duty_c, dtype=torch.float32)
                X_anom[w, idx, :, 5] = torch.tensor(run_dur, dtype=torch.float32)
                X_anom[w, idx, :, 6] = torch.tensor(energy_norm, dtype=torch.float32)
                
    return X_anom, injected_nodes
