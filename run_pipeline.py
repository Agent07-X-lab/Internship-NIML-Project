import os
import csv
import json
import time
import pandas as pd
import numpy as np

def generate_verification_report(summary_rows, output_dir):
    report_path = os.path.join(output_dir, "Verification_Report.html")
    print(f"Generating Verification Report HTML at {report_path}...")
    
    # Process summary rows to JSON for embedding in the dashboard
    json_data = json.dumps(summary_rows)
    
    # Calculate global metrics
    total_houses = len(set(r['House'] for r in summary_rows))
    total_appliances = len(summary_rows)
    avg_active_pct = sum(r['Active Percentage (%)'] for r in summary_rows) / total_appliances if total_appliances > 0 else 0.0
    
    # Load report template from file
    template_path = os.path.join(os.path.dirname(__file__), "report_template.html")
    if not os.path.exists(template_path):
        template_path = "report_template.html"
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
        
    html_content = (
        template
        .replace("__TOTAL_HOUSES__", str(total_houses))
        .replace("__TOTAL_APPLIANCES__", str(total_appliances))
        .replace("__AVG_ACTIVE_PCT__", f"{avg_active_pct:.2f}")
        .replace("__JSON_DATA__", json_data)
        .replace("__TIMESTAMP__", time.strftime("%Y-%m-%d %H:%M:%S"))
    )
    with open(report_path, 'w') as f:
        f.write(html_content)
    print("Verification Report HTML generated successfully!")

def main():
    print("=== Starting REFIT Data Processing Pipeline Execution ===")
    start_time = time.time()
    
    # 1. Check if raw data folder exists and contains csv files
    raw_dir = "1_raw_data"
    output_dir = "3_processed_outputs"
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv') and 'house' in f.lower()]
    
    # If no CSV files are found, generate synthetic data automatically
    if not csv_files:
        print("No raw house CSV files found in 1_raw_data. Running synthetic data generator...")
        import generate_synthetic_refit
        generate_synthetic_refit.main()
        csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv') and 'house' in f.lower()]
        
    print(f"Found {len(csv_files)} house file(s) to process: {csv_files}")
    
    # Import pipeline processing function
    from refit_processor import process_house, extract_house_id
    
    master_summary_rows = []
    
    # Process each house file
    for file in csv_files:
        house_id = extract_house_id(file)
        if house_id is None:
            print(f"Skipping file with unrecognized format: {file}")
            continue
        input_path = os.path.join(raw_dir, file)
        
        try:
            summary = process_house(input_path, output_dir, house_id)
            for row in summary:
                # Add house ID to summary rows for master consolidation
                row_copy = row.copy()
                row_copy['House'] = house_id
                master_summary_rows.append(row_copy)
        except Exception as e:
            print(f"Failed to process house file {file}: {e}")
            
    # Compile Master_Summary.csv
    master_summary_path = os.path.join(output_dir, "Master_Summary.csv")
    print(f"\nCompiling Master Summary CSV at {master_summary_path}...")
    
    # Define columns order
    fieldnames = ['House', 'Appliance', 'Active Percentage (%)', 'Mean Power Active (W)', 'Threshold_W']
    
    with open(master_summary_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in master_summary_rows:
            writer.writerow(row)
            
    print("Master Summary CSV compiled successfully!")
    
    # Generate Verification Report HTML
    generate_verification_report(master_summary_rows, output_dir)
    
    duration = time.time() - start_time
    print(f"\n=== Pipeline completed successfully in {duration:.2f} seconds! ===")

if __name__ == "__main__":
    main()
