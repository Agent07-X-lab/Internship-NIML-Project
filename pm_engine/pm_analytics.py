import json
import numpy as np
from pm_fault_injector import load_config

def calculate_anomaly_score(recon_error, graph_drift, hypergraph_drift, lambdas=(1.0, 0.5, 0.5)):
    """
    Calculates the final anomaly score based on reconstruction error, graph drift, and hypergraph drift.
    """
    l1, l2, l3 = lambdas
    score = l1 * recon_error + l2 * graph_drift + l3 * hypergraph_drift
    return float(score)

def calculate_hypergraph_drift(H_t, W_t, H_baseline, W_baseline):
    """
    Calculates Hypergraph Drift: DH = 0.5 * incidence drift + 0.5 * weight drift
    """
    incidence_drift = np.mean(np.abs(H_t - H_baseline))
    weight_drift = np.mean(np.abs(W_t - W_baseline))
    
    return float(0.5 * incidence_drift + 0.5 * weight_drift)


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
