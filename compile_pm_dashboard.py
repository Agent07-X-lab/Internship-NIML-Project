"""
compile_pm_dashboard.py
Quickly rebuilds Predictive_Maintenance_Dashboard.html from:
  - dashboard_pm_template.html (the JS/HTML template)
  - 3_processed_outputs/PM_Report_House_*.json (individual house reports with isolated_nodes etc.)

Run this any time you update the template or patch report JSON files without
re-running the full training pipeline.
"""

import os
import json
import glob

OUTPUT_DIR = "3_processed_outputs"
TEMPLATE = "dashboard_pm_template.html"
ROOT_OUT  = "Predictive_Maintenance_Dashboard.html"
COPY_OUT  = os.path.join(OUTPUT_DIR, "Predictive_Maintenance_Dashboard.html")

def main():
    # Load all individual PM report JSONs in house-id order
    pattern = os.path.join(OUTPUT_DIR, "PM_Report_House_*.json")
    report_files = sorted(glob.glob(pattern), key=lambda p: int(
        os.path.basename(p).replace("PM_Report_House_", "").replace(".json", "")
    ))

    if not report_files:
        print(f"ERROR: No PM_Report_House_*.json files found in {OUTPUT_DIR}")
        return

    master_reports = []
    for rfile in report_files:
        with open(rfile, 'r', encoding='utf-8') as f:
            d = json.load(f)
        master_reports.append(d)
        house_id = d.get('house_id', '?')
        isolated = d.get('isolated_nodes', [])
        injected = d.get('injected_nodes', [])
        print(f"  House {house_id}: {len(d.get('appliance_names', []))} appliances | "
              f"isolated={isolated} | injected={injected}")

    print(f"\nLoaded {len(master_reports)} house reports.")

    # Load template
    if not os.path.exists(TEMPLATE):
        print(f"ERROR: Template not found: {TEMPLATE}")
        return

    with open(TEMPLATE, 'r', encoding='utf-8') as f:
        html_content = f.read()

    report_json_str = json.dumps(master_reports, indent=2)
    html_content = html_content.replace("__ENRICHED_REPORT_DATA__", report_json_str)

    with open(ROOT_OUT, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Generated: {ROOT_OUT}")

    with open(COPY_OUT, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Saved copy: {COPY_OUT}")

if __name__ == "__main__":
    main()
