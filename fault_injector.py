import copy
import numpy as np
import torch

def inject_fridge_cycle_drift(X_tensor, appliance_names, drift_ratio=0.15):
    """
    Simulates refrigeration seal degradation or thermostat failure by extending
    the duration of ON (active) cycles in the power sequence and binary states.
    """
    X_anom = X_tensor.clone()
    
    # Locate fridge/freezer nodes
    fridge_indices = [i for i, name in enumerate(appliance_names) 
                      if any(k in name.lower() for k in ["fridge", "freezer"])]
                      
    if not fridge_indices:
        print("Warning: No fridge or freezer found for drift injection.")
        return X_anom, []
        
    print(f"Injecting Fridge Cycle Drift into node indices: {fridge_indices} (Appliance: {[appliance_names[i] for i in fridge_indices]})")
    
    # Process each window and each fridge node
    num_windows = X_anom.size(0)
    for idx in fridge_indices:
        for w in range(num_windows):
            power = X_anom[w, idx, :, 0].numpy()
            state = X_anom[w, idx, :, 1].numpy()
            
            # Find contiguous blocks of ON states (1s)
            is_on = (state > 0.5).astype(int)
            diff = np.diff(np.concatenate(([0], is_on, [0])))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0] # exclusive end
            
            new_power = copy.deepcopy(power)
            new_state = copy.deepcopy(state)
            
            for start, end in zip(starts, ends):
                cycle_len = end - start
                extension = int(np.round(cycle_len * drift_ratio))
                if extension <= 0:
                    extension = 1
                    
                # Extend the cycle forward by the extension amount
                ext_start = end
                ext_end = min(len(power), end + extension)
                
                # Set extension power to average active power of this cycle
                avg_active_power = np.mean(power[start:end]) if end > start else 0.5
                
                new_power[ext_start:ext_end] = avg_active_power
                new_state[ext_start:ext_end] = 1.0
                
            X_anom[w, idx, :, 0] = torch.tensor(new_power, dtype=torch.float32)
            X_anom[w, idx, :, 1] = torch.tensor(new_state, dtype=torch.float32)
            
    return X_anom, fridge_indices

def inject_washing_machine_micro_spikes(X_tensor, appliance_names, spike_magnitude=0.12):
    """
    Simulates mechanical bearing wear or motor coil friction by injecting high-frequency
    micro-spikes (current surges) to the active phases of washing machines or dryers.
    """
    X_anom = X_tensor.clone()
    
    # Locate washing machine/dryer/washer nodes
    wm_indices = [i for i, name in enumerate(appliance_names) 
                  if any(k in name.lower() for k in ["washing", "washer", "dryer"])]
                  
    if not wm_indices:
        print("Warning: No washing machine or dryer found for micro-spikes injection.")
        return X_anom, []
        
    print(f"Injecting Washing Machine Micro-spikes into node indices: {wm_indices} (Appliance: {[appliance_names[i] for i in wm_indices]})")
    
    num_windows = X_anom.size(0)
    for idx in wm_indices:
        for w in range(num_windows):
            power = X_anom[w, idx, :, 0].clone()
            state = X_anom[w, idx, :, 1]
            
            # Only add spikes when the appliance is active (state == 1)
            active_mask = state > 0.5
            if active_mask.sum() > 0:
                # Generate random uniform spikes on active time steps
                spikes = torch.rand(power.size()) * spike_magnitude
                # Apply only where active
                power[active_mask] = power[active_mask] + spikes[active_mask]
                
                # Clip values to represent physically standard limits
                X_anom[w, idx, :, 0] = torch.clamp(power, 0.0, 1.3)
                
    return X_anom, wm_indices
