import json
import numpy as np
from pm_fault_injector import load_config

def calculate_health_index(reconstruction_error, baseline_threshold):
    """Calculates the health index percentage based on reconstruction error and threshold."""
    if baseline_threshold <= 0:
        baseline_threshold = 1.0
    
    # Calculate health index score (drops as reconstruction error exceeds threshold)
    ratio = reconstruction_error / baseline_threshold
    health = 100.0 - (ratio * 50.0) # threshold error means 50% health
    
    # Clamp between 0 and 100
    return float(np.clip(health, 0.0, 100.0))

def estimate_fault_severity(drift_ratio, config_path="pm_config.json"):
    """Estimates fault severity based on the Drift Ratio and configurable thresholds."""
    config = load_config(config_path)
    thresholds = config.get("drift_thresholds", {
        "healthy_limit": 1.15,
        "minor_limit": 1.30,
        "moderate_limit": 1.60
    })
    
    h_limit = thresholds.get("healthy_limit", 1.15)
    minor_limit = thresholds.get("minor_limit", 1.30)
    mod_limit = thresholds.get("moderate_limit", 1.60)
    
    if drift_ratio < h_limit:
        return "Healthy"
    elif drift_ratio < minor_limit:
        return "Minor"
    elif drift_ratio < mod_limit:
        return "Moderate"
    else:
        return "Critical"

def predict_rul(health_history, num_windows_total, stride_in_days=1.0):
    """
    Predicts Remaining Useful Life (RUL) in days using Linear and Exponential regression
    on health index history.
    
    health_history: List of float values representing historical health index percentages.
    num_windows_total: Total windows in dataset (used to compute scaling factors).
    stride_in_days: Conversion factor of window steps to days.
    """
    if len(health_history) < 3:
        return 99 # Default placeholder if history is insufficient
        
    t = np.arange(len(health_history))
    y = np.array(health_history)
    
    # 1. Try Linear Fit: y = a * t + b
    slope, intercept = np.polyfit(t, y, 1)
    
    # 2. Try Exponential Fit: y = c * e^(a * t) -> log(y) = log(c) + a * t
    # Clip values to avoid log(0)
    y_clipped = np.clip(y, 1.0, 100.0)
    log_y = np.log(y_clipped)
    exp_slope, exp_intercept = np.polyfit(t, log_y, 1)
    
    # Target failure health score = 50%
    failure_target = 50.0
    
    rul_linear = float('inf')
    if slope < -0.01:
        # y_fail = slope * t_fail + intercept = 50.0
        t_fail = (failure_target - intercept) / slope
        rul_linear = (t_fail - (len(health_history) - 1)) * stride_in_days
        
    rul_exponential = float('inf')
    if exp_slope < -1e-4:
        # log(y_fail) = exp_slope * t_fail + exp_intercept
        t_fail = (np.log(failure_target) - exp_intercept) / exp_slope
        rul_exponential = (t_fail - (len(health_history) - 1)) * stride_in_days
        
    # Choose exponential RUL if valid, otherwise linear, otherwise return high default
    rul_final = min(rul_linear, rul_exponential)
    
    if np.isinf(rul_final) or np.isnan(rul_final) or rul_final <= 0:
        # No decay detected, device is healthy
        return 90
        
    return int(np.clip(np.round(rul_final), 1, 120))

def perform_root_cause_analysis(appliance_name, mean_power_ratio, duration_ratio, spike_variance_ratio):
    """
    Estimates the probable physical cause of a detected anomaly based on changes
    in operating characteristics relative to baseline.
    """
    name_lower = appliance_name.lower()
    
    # Refrigerator Diagnostics
    if any(k in name_lower for k in ["fridge", "freezer", "refrigerator"]):
        if mean_power_ratio > 1.10 and duration_ratio > 1.15:
            return "Compressor efficiency loss & Door Gasket wear", 0.94
        elif duration_ratio > 1.15:
            return "Thermostat calibration drift / Seal leak", 0.88
        elif mean_power_ratio > 1.10:
            return "Compressor electrical motor wear", 0.85
        else:
            return "General refrigeration cooling loss", 0.75
            
    # Motor-Driven Diagnostics (Washer, Dryer)
    elif any(k in name_lower for k in ["washing", "washer", "dryer"]):
        if spike_variance_ratio > 1.15:
            return "Bearing wear & Mechanical friction", 0.92
        elif mean_power_ratio > 1.10 and spike_variance_ratio > 1.05:
            return "Motor windings overheat / Drum Imbalance", 0.89
        elif mean_power_ratio > 1.05:
            return "Standby board current leakage", 0.80
        else:
            return "Mechanical transmission slip", 0.70
            
    # Water Pump & Heating Diagnostics (Dishwasher)
    elif "dishwasher" in name_lower:
        if mean_power_ratio < 0.90:
            return "Wash pump impeller degradation", 0.91
        elif duration_ratio > 1.15:
            return "Heating element efficiency loss", 0.87
        else:
            return "Solenoid valve or drain block", 0.78
            
    # General heating loads (Kettle, Microwave)
    elif any(k in name_lower for k in ["kettle", "microwave"]):
        if mean_power_ratio < 0.92:
            return "Heating coil/magnetron aging", 0.89
        elif duration_ratio > 1.12:
            return "Control relay contact degradation", 0.82
        else:
            return "Thermostatic sensor drift", 0.75
            
    # Electronics (TV, Computer)
    elif any(k in name_lower for k in ["television", "tv", "computer"]):
        if spike_variance_ratio > 1.10:
            return "Power supply unit capacitor aging", 0.88
        else:
            return "Internal component thermal drift", 0.72
            
    return "Generic hardware degradation", 0.65

def get_maintenance_recommendation(root_cause_text):
    """Maps root cause text to highly specific maintenance recommendations."""
    cause_lower = root_cause_text.lower()
    
    if "gasket" in cause_lower or "seal" in cause_lower:
        return [
            "Inspect and clean door gasket seals.",
            "Replace magnetic door gasket strip if cracked.",
            "Ensure unit is leveled so door closes firmly."
        ]
    elif "compressor" in cause_lower:
        return [
            "Clean dust from condenser coils.",
            "Verify condenser fan is rotating freely.",
            "Check compressor current draw against rated plate value."
        ]
    elif "bearing" in cause_lower or "friction" in cause_lower:
        return [
            "Inspect motor and drum bearing assemblies for noise.",
            "Lubricate rotating bearings if serviceable.",
            "Check belt tension and alignment."
        ]
    elif "imbalance" in cause_lower:
        return [
            "Re-level appliance feet to damp vibrations.",
            "Check drum suspension springs and shock dampers.",
            "Advise user to balance heavy laundry loads."
        ]
    elif "pump" in cause_lower:
        return [
            "Clean filters and sump assembly.",
            "Check pump impeller for debris or foreign objects.",
            "Verify drain hose is not kinked or blocked."
        ]
    elif "heating coil" in cause_lower or "magnetron" in cause_lower or "heater" in cause_lower:
        return [
            "Inspect heater terminals and wire harness contacts.",
            "Measure element resistance (Ohms) to check for partial shorts.",
            "Clean scale build-up from heating element."
        ]
    elif "relay" in cause_lower or "switch" in cause_lower:
        return [
            "Check control board relay contacts for pitting.",
            "Test control board output voltage to relay.",
            "Replace control relay if clicking is intermittent."
        ]
    elif "capacitor" in cause_lower or "power supply" in cause_lower:
        return [
            "Inspect power supply board for bulged or leaking capacitors.",
            "Verify input voltage stability and plug tightness.",
            "Replace power board if ripple voltage is high."
        ]
        
    return [
        "Perform general physical inspection.",
        "Check for standard utility connection tightness.",
        "Update control firmware if applicable."
    ]
