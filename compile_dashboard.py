import os
import json
import csv

def main():
    outputs_dir = "3_processed_outputs"
    report_path = os.path.join(outputs_dir, "Anomaly_Report.json")
    
    if not os.path.exists(report_path):
        print(f"Error: Anomaly_Report.json not found at {report_path}. Run train_eval.py first.")
        return

    # Load report
    with open(report_path, 'r') as f:
        report_data = json.load(f)

    # Enrich report data with metadata and summary CSVs
    enriched_report = []
    for house_rep in report_data:
        house_id = house_rep["house_id"]
        
        # Load metadata
        meta_path = os.path.join(outputs_dir, f"House_{house_id}_Metadata.json")
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
                
        # Load summary CSV
        summary_path = os.path.join(outputs_dir, f"House_{house_id}_Summary.csv")
        summaries = {}
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    app_name = row["Appliance"]
                    summaries[app_name] = {
                        "active_pct": float(row["Active Percentage (%)"]),
                        "mean_power": float(row["Mean Power Active (W)"]),
                        "threshold": float(row["Threshold_W"])
                    }
                    
        house_rep["metadata"] = metadata
        house_rep["summaries"] = summaries
        enriched_report.append(house_rep)
        
    print(f"Enriched report data loaded for {len(enriched_report)} houses.")

    # Load HTML template from file
    template_path = os.path.join(os.path.dirname(__file__), "dashboard_template.html")
    if not os.path.exists(template_path):
        template_path = "dashboard_template.html"
    with open(template_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Inject actual data stringified into Javascript representation
    report_json_str = json.dumps(enriched_report, indent=4)
    html_content = html_content.replace("__ENRICHED_REPORT_DATA__", report_json_str)

    # Write dashboard to root directory
    root_out_path = "Predictive_Maintenance_Dashboard.html"
    with open(root_out_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Generated root dashboard: {root_out_path}")

    # Write a copy to outputs folder as well
    copy_out_path = os.path.join(outputs_dir, "Predictive_Maintenance_Dashboard.html")
    with open(copy_out_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Saved copy of dashboard to: {copy_out_path}")

if __name__ == "__main__":
    main()
