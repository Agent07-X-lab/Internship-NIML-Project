import os
import re
import json
import argparse
import pandas as pd
import numpy as np

# Official REFIT Appliance Mappings for Houses 1 to 21 (excluding 14)
REFIT_HOUSE_MAPPINGS = {
    1: {
        "Appliance1": "Fridge",
        "Appliance2": "Chest Freezer",
        "Appliance3": "Upright Freezer",
        "Appliance4": "Tumble Dryer",
        "Appliance5": "Washing Machine",
        "Appliance6": "Dishwasher",
        "Appliance7": "Computer Site",
        "Appliance8": "Television Site",
        "Appliance9": "Electric Heater"
    },
    2: {
        "Appliance1": "Fridge-Freezer",
        "Appliance2": "Washing Machine",
        "Appliance3": "Dishwasher",
        "Appliance4": "Television",
        "Appliance5": "Microwave",
        "Appliance6": "Toaster",
        "Appliance7": "Hi-Fi",
        "Appliance8": "Kettle",
        "Appliance9": "Oven Extractor Fan"
    },
    3: {
        "Appliance1": "Toaster",
        "Appliance2": "Fridge-Freezer",
        "Appliance3": "Freezer",
        "Appliance4": "Tumble Dryer",
        "Appliance5": "Dishwasher",
        "Appliance6": "Washing Machine",
        "Appliance7": "Television",
        "Appliance8": "Microwave",
        "Appliance9": "Kettle"
    },
    4: {
        "Appliance1": "Fridge",
        "Appliance2": "Freezer",
        "Appliance3": "Fridge-Freezer",
        "Appliance4": "Washing Machine (1)",
        "Appliance5": "Washing Machine (2)",
        "Appliance6": "Computer Site",
        "Appliance7": "Television Site",
        "Appliance8": "Microwave",
        "Appliance9": "Kettle"
    },
    5: {
        "Appliance1": "Fridge-Freezer",
        "Appliance2": "Tumble Dryer",
        "Appliance3": "Washing Machine",
        "Appliance4": "Dishwasher",
        "Appliance5": "Computer Site",
        "Appliance6": "Television Site",
        "Appliance7": "Combination Microwave",
        "Appliance8": "Kettle",
        "Appliance9": "Toaster"
    },
    6: {
        "Appliance1": "Freezer (Utility Room)",
        "Appliance2": "Washing Machine",
        "Appliance3": "Dishwasher",
        "Appliance4": "MJY Computer",
        "Appliance5": "Television Site",
        "Appliance6": "Microwave",
        "Appliance7": "Kettle",
        "Appliance8": "Toaster",
        "Appliance9": "PGM Computer"
    },
    7: {
        "Appliance1": "Fridge",
        "Appliance2": "Freezer (Garage)",
        "Appliance3": "Freezer",
        "Appliance4": "Tumble Dryer",
        "Appliance5": "Washing Machine",
        "Appliance6": "Dishwasher",
        "Appliance7": "Television Site",
        "Appliance8": "Toaster",
        "Appliance9": "Kettle"
    },
    8: {
        "Appliance1": "Fridge",
        "Appliance2": "Freezer",
        "Appliance3": "Dryer",
        "Appliance4": "Washing Machine",
        "Appliance5": "Toaster",
        "Appliance6": "Computer",
        "Appliance7": "Television Site",
        "Appliance8": "Microwave",
        "Appliance9": "Kettle"
    },
    9: {
        "Appliance1": "Fridge-Freezer",
        "Appliance2": "Washer Dryer",
        "Appliance3": "Washing Machine",
        "Appliance4": "Dishwasher",
        "Appliance5": "Television Site",
        "Appliance6": "Microwave",
        "Appliance7": "Kettle",
        "Appliance8": "Hi-Fi",
        "Appliance9": "Electric Heater"
    },
    10: {
        "Appliance1": "Magimix (Blender)",
        "Appliance2": "Freezer",
        "Appliance3": "Chest Freezer (In Garage)",
        "Appliance4": "Fridge-Freezer",
        "Appliance5": "Washing Machine",
        "Appliance6": "Dishwasher",
        "Appliance7": "Television Site",
        "Appliance8": "Microwave",
        "Appliance9": "Kenwood KMix"
    },
    11: {
        "Appliance1": "Fridge",
        "Appliance2": "Fridge-Freezer",
        "Appliance3": "Washing Machine",
        "Appliance4": "Dishwasher",
        "Appliance5": "Computer Site",
        "Appliance6": "Microwave",
        "Appliance7": "Kettle",
        "Appliance8": "Router",
        "Appliance9": "Hi-Fi"
    },
    12: {
        "Appliance1": "Fridge-Freezer",
        "Appliance2": "Television Site(Lounge)",
        "Appliance3": "Microwave",
        "Appliance4": "Kettle",
        "Appliance5": "Toaster",
        "Appliance6": "Television Site(Bedroom)",
        "Appliance7": "Not Used",
        "Appliance8": "Not Used",
        "Appliance9": "Not Used"
    },
    13: {
        "Appliance1": "Television Site",
        "Appliance2": "Unknown",
        "Appliance3": "Washing Machine",
        "Appliance4": "Dishwasher",
        "Appliance5": "Tumble Dryer",
        "Appliance6": "Television Site",
        "Appliance7": "Computer Site",
        "Appliance8": "Microwave",
        "Appliance9": "Kettle"
    },
    15: {
        "Appliance1": "Fridge-Freezer",
        "Appliance2": "Tumble Dryer",
        "Appliance3": "Washing Machine",
        "Appliance4": "Dishwasher",
        "Appliance5": "Computer Site",
        "Appliance6": "Television Site",
        "Appliance7": "Microwave",
        "Appliance8": "Kettle",
        "Appliance9": "Toaster"
    },
    16: {
        "Appliance1": "Fridge-Freezer (1)",
        "Appliance2": "Fridge-Freezer (2)",
        "Appliance3": "Electric Heater (1)?",
        "Appliance4": "Electric Heater (2)",
        "Appliance5": "Washing Machine",
        "Appliance6": "Dishwasher",
        "Appliance7": "Computer Site",
        "Appliance8": "Television Site",
        "Appliance9": "Dehumidifier/Heater"
    },
    17: {
        "Appliance1": "Freezer (Garage)",
        "Appliance2": "Fridge-Freezer",
        "Appliance3": "Tumble Dryer (Garage)",
        "Appliance4": "Washing Machine",
        "Appliance5": "Computer Site",
        "Appliance6": "Television Site",
        "Appliance7": "Microwave",
        "Appliance8": "Kettle",
        "Appliance9": "Plug Site (Bedroom)"
    },
    18: {
        "Appliance1": "Fridge(garage)",
        "Appliance2": "Freezer(garage)",
        "Appliance3": "Fridge-Freezer",
        "Appliance4": "Washer Dryer(garage)",
        "Appliance5": "Washing Machine",
        "Appliance6": "Dishwasher",
        "Appliance7": "Desktop Computer",
        "Appliance8": "Television Site",
        "Appliance9": "Microwave"
    },
    19: {
        "Appliance1": "Fridge & Freezer",
        "Appliance2": "Washing Machine",
        "Appliance3": "Television Site",
        "Appliance4": "Microwave",
        "Appliance5": "Kettle",
        "Appliance6": "Toaster",
        "Appliance7": "Bread-maker",
        "Appliance8": "Lamp (80Watts)",
        "Appliance9": "Hi-Fi"
    },
    20: {
        "Appliance1": "Fridge",
        "Appliance2": "Freezer",
        "Appliance3": "Tumble Dryer",
        "Appliance4": "Washing Machine",
        "Appliance5": "Dishwasher",
        "Appliance6": "Computer Site",
        "Appliance7": "Television Site",
        "Appliance8": "Microwave",
        "Appliance9": "Kettle"
    },
    21: {
        "Appliance1": "Fridge-Freezer",
        "Appliance2": "Tumble Dryer",
        "Appliance3": "Washing Machine",
        "Appliance4": "Dishwasher",
        "Appliance5": "Food Mixer",
        "Appliance6": "Television",
        "Appliance7": "Kettle/Toaster",
        "Appliance8": "Vivarium",
        "Appliance9": "Pond Pump"
    }
}

def clean_name(name):
    """Clean appliance names to be valid, standardized snake_case identifiers."""
    cleaned = re.sub(r'[^a-zA-Z0-9]', '_', name)
    cleaned = re.sub(r'_+', '_', cleaned)
    return cleaned.strip('_')

def get_threshold(appliance_name):
    """Determine a wattage threshold for the binary active state (ON/OFF)."""
    name_lower = appliance_name.lower()
    if "kettle" in name_lower:
        return 1000.0
    elif any(k in name_lower for k in ["washing", "washer", "dryer"]):
        return 20.0
    elif "dishwasher" in name_lower:
        return 50.0
    elif any(k in name_lower for k in ["fridge", "freezer"]):
        return 15.0
    elif "microwave" in name_lower:
        return 100.0
    elif "toaster" in name_lower:
        return 100.0
    elif any(k in name_lower for k in ["heater", "dehumidifier"]):
        return 100.0
    elif any(k in name_lower for k in ["television", "tv", "computer", "router", "hi_fi", "lamp"]):
        return 10.0
    else:
        return 10.0 # Default fallback threshold

def extract_house_id(filename):
    """Extract house ID integer from filename (e.g. CLEAN_House1.csv -> 1)."""
    match = re.search(r'House_?(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def process_house(input_path, output_dir, house_id=None):
    if house_id is None:
        house_id = extract_house_id(os.path.basename(input_path))
        if house_id is None:
            raise ValueError(f"Could not extract House ID from filename: {input_path}")
            
    print(f"\n--- Processing House {house_id} from {input_path} ---")
    
    # Check if we have mappings for this house
    if house_id not in REFIT_HOUSE_MAPPINGS:
        print(f"Warning: No explicit mappings found for House {house_id}. Using default Appliance1..9 names.")
        mappings = {f"Appliance{i}": f"Appliance{i}" for i in range(1, 10)}
    else:
        mappings = REFIT_HOUSE_MAPPINGS[house_id]
        
    # Load raw data
    df = pd.read_csv(input_path)
    total_raw_rows = len(df)
    print(f"Loaded raw data. Rows: {total_raw_rows}. Columns: {list(df.columns)}")
    
    # 1. Temporal Standardization
    # Find DateTime column (case insensitive)
    time_col = None
    for col in df.columns:
        if col.lower() in ['time', 'datetime']:
            time_col = col
            break
    if not time_col:
        raise ValueError(f"No Time or DateTime column found in {input_path}")
        
    # Drop Unix timestamp column
    unix_cols = [c for c in df.columns if c.lower() in ['unix', 'unix timestamp', 'unix_timestamp']]
    if unix_cols:
        df = df.drop(columns=unix_cols)
        print(f"Dropped Unix columns: {unix_cols}")
        
    # Convert DateTime strings to objects
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)
    
    # Resample all data to a strict 8-second frequency (using .mean())
    df_resampled = df.resample('8s').mean()
    print(f"Resampled data to strict 8-second intervals. Resampled rows: {len(df_resampled)}")
    
    # 2. Gap Mitigation
    # Apply linear interpolation (limit=5) to bridge minor sensor drops
    df_interpolated = df_resampled.interpolate(method='linear', limit=5)
    # Fill remaining NaNs with 0
    df_standardized = df_interpolated.fillna(0.0)
    print("Applied gap mitigation (linear interpolation with limit=5, remaining NaNs filled with 0)")
    
    # 3. Semantic Mapping & State Extraction
    # We will build the new timeseries dataframe
    final_cols = []
    timeseries_data = pd.DataFrame(index=df_standardized.index)
    
    # Keep Aggregate if present
    agg_col = None
    for col in df_standardized.columns:
        if col.lower() == 'aggregate':
            agg_col = col
            break
            
    if agg_col:
        timeseries_data['Aggregate'] = df_standardized[agg_col]
        final_cols.append('Aggregate')
        
    summary_stats = []
    metadata_mappings = {}
    metadata_thresholds = {}
    
    # Process each appliance 1 to 9
    for i in range(1, 10):
        raw_col = f"Appliance{i}"
        
        # Determine the mapped name
        original_mapped_name = mappings.get(raw_col, raw_col)
        clean_mapped_name = clean_name(original_mapped_name)
        
        # Handle duplicates in mapping (e.g. "Television Site" appearing twice)
        # We append a suffix if it's already used
        counter = 1
        base_name = clean_mapped_name
        while clean_mapped_name in timeseries_data.columns:
            clean_mapped_name = f"{base_name}_{counter}"
            counter += 1
            
        # Get threshold
        threshold_w = get_threshold(original_mapped_name)
        
        metadata_mappings[raw_col] = original_mapped_name
        metadata_thresholds[clean_mapped_name] = threshold_w
        
        # Extract power values (if column missing in raw data, default to 0.0)
        if raw_col in df_standardized.columns:
            power_series = df_standardized[raw_col]
        else:
            power_series = pd.Series(0.0, index=df_standardized.index)
            
        # Calculate active state binary (1 if >= Threshold, else 0)
        state_series = (power_series >= threshold_w).astype(int)
        
        # Save to timeseries DataFrame
        timeseries_data[clean_mapped_name] = power_series
        timeseries_data[f"{clean_mapped_name}_ON"] = state_series
        
        final_cols.extend([clean_mapped_name, f"{clean_mapped_name}_ON"])
        
        # Calculate summary statistics
        total_steps = len(df_standardized)
        active_steps = state_series.sum()
        active_pct = (active_steps / total_steps * 100) if total_steps > 0 else 0.0
        
        active_power = power_series[state_series == 1]
        mean_power_active = active_power.mean() if len(active_power) > 0 else 0.0
        
        summary_stats.append({
            "Appliance": clean_mapped_name,
            "Active Percentage (%)": round(active_pct, 4),
            "Mean Power Active (W)": round(mean_power_active, 2),
            "Threshold_W": threshold_w
        })
        
    # Reorder timeseries columns to group by appliance name and state
    timeseries_data = timeseries_data[final_cols]
    
    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    
    # Output 1: Time Series Data (.csv)
    ts_out_path = os.path.join(output_dir, f"House_{house_id}_Processed.csv")
    timeseries_data.to_csv(ts_out_path)
    print(f"Saved Processed Timeseries to {ts_out_path}. Shape: {timeseries_data.shape}")
    
    # Output 2: Summary Statistics (.csv)
    stats_df = pd.DataFrame(summary_stats)
    stats_out_path = os.path.join(output_dir, f"House_{house_id}_Summary.csv")
    stats_df.to_csv(stats_out_path, index=False)
    print(f"Saved Summary Statistics to {stats_out_path}")
    
    # Output 3: Metadata (.json)
    metadata = {
        "House": house_id,
        "SourceFile": os.path.basename(input_path),
        "TotalRawRows": total_raw_rows,
        "ProcessedRows": len(timeseries_data),
        "StartTime": timeseries_data.index.min().strftime("%Y-%m-%d %H:%M:%S"),
        "EndTime": timeseries_data.index.max().strftime("%Y-%m-%d %H:%M:%S"),
        "SamplingFrequency": "8s",
        "Mappings": metadata_mappings,
        "Thresholds": metadata_thresholds
    }
    meta_out_path = os.path.join(output_dir, f"House_{house_id}_Metadata.json")
    with open(meta_out_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved Metadata JSON to {meta_out_path}")
    
    return summary_stats

def main():
    parser = argparse.ArgumentParser(description="Process REFIT House CSV files.")
    parser.add_argument("--input-file", help="Path to a single raw House CSV file")
    parser.add_argument("--input-dir", default="1_raw_data", help="Directory containing raw House CSV files")
    parser.add_argument("--output-dir", default="3_processed_outputs", help="Directory to save processed outputs")
    parser.add_argument("--house-id", type=int, help="Optional specific House ID to process")
    args = parser.parse_args()
    
    if args.input_file:
        process_house(args.input_file, args.output_dir, args.house_id)
    else:
        # Process files in input directory matching CLEAN_House*.csv or House*.csv
        if not os.path.exists(args.input_dir):
            print(f"Error: Input directory '{args.input_dir}' does not exist.")
            return
            
        csv_files = [f for f in os.listdir(args.input_dir) if f.endswith('.csv') and 'house' in f.lower()]
        if not csv_files:
            print(f"No matching house CSV files found in '{args.input_dir}'.")
            return
            
        csv_files.sort(key=lambda x: extract_house_id(x) or 0)
        
        for file in csv_files:
            file_house_id = extract_house_id(file)
            if args.house_id and file_house_id != args.house_id:
                continue
            input_path = os.path.join(args.input_dir, file)
            try:
                process_house(input_path, args.output_dir, file_house_id)
            except Exception as e:
                print(f"Error processing {input_path}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()
