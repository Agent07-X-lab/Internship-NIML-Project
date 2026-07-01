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
    
    injected_nodes = []
    
    for idx, name in enumerate(appliance_names):
        name_lower = name.lower()
        
        # Determine if this appliance is subject to degradation
        is_fridge = any(k in name_lower for k in ["fridge", "freezer", "refrigerator"])
        is_washer = any(k in name_lower for k in ["washing", "washer", "dryer"])
        is_dishwasher = "dishwasher" in name_lower
        is_ac = any(k in name_lower for k in ["air_conditioner", "ac"])
        is_fan = "fan" in name_lower
        is_microwave = "microwave" in name_lower
        is_tv = any(k in name_lower for k in ["television", "tv", "computer"])
        is_kettle = "kettle" in name_lower
        
        if not (is_fridge or is_washer or is_dishwasher or is_ac or is_fan or is_microwave or is_tv or is_kettle):
            continue
            
        injected_nodes.append(idx)
        print(f"Applying gradual fault simulation to: {name} (Node index {idx})")
        
        for w in range(num_windows):
            # Progression factor scales from 0.0 (start of test set) to 1.0 (end of test set)
            progress = w / (num_windows - 1) if num_windows > 1 else 1.0
            
            # Extract power and state vectors for this node
            # Features: 0 = Normalized Power, 1 = Binary State, 2 = Mean Power, etc.
            # (Features depend on whether we use dynamic or standard graphs.
            # To be robust, we update Feature 0 [Power] and Feature 1 [State], 
            # and re-calculate others like Mean, Var if needed).
            power = X_anom[w, idx, :, 0].numpy()
            state = X_anom[w, idx, :, 1].numpy()
            
            new_power = copy.deepcopy(power)
            new_state = copy.deepcopy(state)
            
            # Locate active cycles (ON states)
            is_on = (state > 0.5).astype(int)
            diff = np.diff(np.concatenate(([0], is_on, [0])))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            
            # --- 1. Refrigerator Faults: compressor loss, thermostat drift, door leak ---
            if is_fridge:
                # Cycle length extension up to 35% (thermostat/door leak)
                ext_ratio = 0.35 * progress
                for start, end in zip(starts, ends):
                    cycle_len = end - start
                    ext_len = int(np.round(cycle_len * ext_ratio))
                    if ext_len > 0:
                        ext_start = end
                        ext_end = min(len(power), end + ext_len)
                        avg_active = np.mean(power[start:end]) if end > start else 0.5
                        new_power[ext_start:ext_end] = avg_active
                        new_state[ext_start:ext_end] = 1.0
                # Compressor efficiency loss (power level increases by up to 20%)
                active_mask = new_state > 0.5
                new_power[active_mask] = new_power[active_mask] * (1.0 + 0.20 * progress)
                
            # --- 2. Washing Machine Faults: bearing wear, motor overheat, drum imbalance ---
            elif is_washer:
                active_mask = new_state > 0.5
                inactive_mask = ~active_mask
                
                # Motor overheating: standby power leakage up to 0.05 (normalized scale)
                new_power[inactive_mask] = new_power[inactive_mask] + 0.05 * progress
                
                # Bearing wear: high frequency micro-spikes up to 0.22 (magnitude)
                if np.sum(active_mask) > 0:
                    spikes = np.random.rand(len(power)) * 0.22 * progress
                    new_power[active_mask] = new_power[active_mask] + spikes[active_mask]
                    
                # Drum imbalance: transient surges of 0.30 power (5% probability during active phase)
                for t_step in range(len(power)):
                    if active_mask[t_step] and np.random.random() < 0.05:
                        new_power[t_step] += 0.30 * progress
                        
            # --- 3. Dishwasher Faults: pump degradation, heater efficiency drop ---
            elif is_dishwasher:
                # Pump degradation: lower power draw by up to 18% during washing
                active_mask = new_state > 0.5
                new_power[active_mask] = new_power[active_mask] * (1.0 - 0.18 * progress)
                
                # Heater efficiency loss: extends high-power heating duration by up to 25%
                ext_ratio = 0.25 * progress
                for start, end in zip(starts, ends):
                    cycle_len = end - start
                    ext_len = int(np.round(cycle_len * ext_ratio))
                    if ext_len > 0:
                        ext_start = end
                        ext_end = min(len(power), end + ext_len)
                        avg_active = np.mean(power[start:end]) if end > start else 0.4
                        new_power[ext_start:ext_end] = avg_active
                        new_state[ext_start:ext_end] = 1.0
                        
            # --- 4. Air Conditioner Faults: compressor degradation, refrigerant leak ---
            elif is_ac:
                # Higher power draw by up to 25%
                active_mask = new_state > 0.5
                new_power[active_mask] = new_power[active_mask] * (1.0 + 0.25 * progress)
                # Thermostat/Refrigerant leak cycle extension up to 40%
                ext_ratio = 0.40 * progress
                for start, end in zip(starts, ends):
                    cycle_len = end - start
                    ext_len = int(np.round(cycle_len * ext_ratio))
                    if ext_len > 0:
                        ext_start = end
                        ext_end = min(len(power), end + ext_len)
                        new_power[ext_start:ext_end] = np.mean(power[start:end])
                        new_state[ext_start:ext_end] = 1.0
                        
            # --- 5. Fan Faults: bearing friction, motor wear ---
            elif is_fan:
                # Bearing friction: slow startup spikes (high transient peak at beginning of cycles)
                for start in starts:
                    peak_steps = min(5, len(power) - start)
                    new_power[start:start+peak_steps] = new_power[start:start+peak_steps] * (1.5 * progress)
                # Running power variance (higher noise up to 8%)
                active_mask = new_state > 0.5
                noise = np.random.normal(0, 0.08 * progress, len(power))
                new_power[active_mask] = np.clip(new_power[active_mask] + noise[active_mask], 0.0, 1.3)
                
            # --- 6. Microwave / Kettle Faults: heating element degradation ---
            elif is_microwave or is_kettle:
                # Reduced power draw (up to 12% drop)
                active_mask = new_state > 0.5
                new_power[active_mask] = new_power[active_mask] * (1.0 - 0.12 * progress)
                # Extended heating run cycle (up to 20%)
                ext_ratio = 0.20 * progress
                for start, end in zip(starts, ends):
                    cycle_len = end - start
                    ext_len = int(np.round(cycle_len * ext_ratio))
                    if ext_len > 0:
                        ext_start = end
                        ext_end = min(len(power), end + ext_len)
                        new_power[ext_start:ext_end] = np.mean(power[start:end])
                        new_state[ext_start:ext_end] = 1.0
                        
            # --- 7. TV / Electronics: power supply instability ---
            elif is_tv:
                # Flickering power readings (high-frequency noise in both active and standby)
                noise = np.random.normal(0, 0.06 * progress, len(power))
                new_power = np.clip(new_power + noise, 0.0, 1.3)
                
            # Clamping to standard physical values
            new_power = np.clip(new_power, 0.0, 1.3)
            
            # Save back to features
            X_anom[w, idx, :, 0] = torch.tensor(new_power, dtype=torch.float32)
            X_anom[w, idx, :, 1] = torch.tensor(new_state, dtype=torch.float32)
            
            # If dynamic features are present, re-calculate the rolling stats (features 2 to 6)
            if X_anom.size(3) > 2:
                mean_p = np.mean(new_power)
                var_p = np.var(new_power)
                duty_c = np.sum(new_state) / len(new_state)
                
                # Re-calculate running duration
                run_dur = []
                current_run = 0.0
                for s_val in new_state:
                    if s_val > 0.5:
                        current_run += 1.0
                    else:
                        current_run = 0.0
                    run_dur.append(current_run)
                run_dur = np.array(run_dur) / len(new_state)
                
                # Re-calculate energy
                energy_val = np.cumsum(new_power) * (8.0 / 3600.0)
                energy_norm = energy_val / (np.max(new_power) * (len(new_state) * 8.0 / 3600.0) + 1e-9)
                
                X_anom[w, idx, :, 2] = torch.tensor(mean_p, dtype=torch.float32)
                X_anom[w, idx, :, 3] = torch.tensor(var_p, dtype=torch.float32)
                X_anom[w, idx, :, 4] = torch.tensor(duty_c, dtype=torch.float32)
                X_anom[w, idx, :, 5] = torch.tensor(run_dur, dtype=torch.float32)
                X_anom[w, idx, :, 6] = torch.tensor(energy_norm, dtype=torch.float32)
                
    return X_anom, injected_nodes
