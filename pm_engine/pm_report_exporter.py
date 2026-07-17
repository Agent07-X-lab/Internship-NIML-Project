import os
import json
import time

def export_json_report(report_data, out_path):
    """Exports structured JSON report to output directory."""
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=4)
    print(f"Exported JSON report to {out_path}")

def export_html_report(report_data, out_path):
    """Generates a self-contained, beautiful HTML audit report representing research findings."""
    house_id = report_data.get("house_id", 99)
    appliances = report_data.get("appliances", [])
    timestamp = report_data.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
    graph_drift = report_data.get("graph_drift", 0.0)
    
    # Custom colored HTML status cells based on severity
    table_rows_html = ""
    for app in appliances:
        severity = app.get("severity", "Healthy")
        sev_class = severity.lower()
        if sev_class == "alert (fault)":
            sev_class = "critical"
            
        recs_list = app.get("recommendations", [])
        recs_html = "".join(f"<li>{r}</li>" for r in recs_list)
        
        table_rows_html += f"""
        <tr class="severity-{sev_class}">
            <td class="font-bold">{app.get("name")}</td>
            <td>
                <div class="progress-bar-container">
                    <div class="progress-bar-fill" style="width: {app.get("health")}%"></div>
                    <span class="progress-bar-text">{app.get("health")}%</span>
                </div>
            </td>
            <td><span class="badge badge-{sev_class}">{severity}</span></td>
            <td>{int(app.get("confidence", 0.95) * 100)}%</td>
            <td class="font-mono font-bold">{app.get("rul")} Days</td>
            <td class="cause-text">{app.get("root_cause", "N/A")}</td>
            <td>
                <ul class="recs-list">
                    {recs_html}
                </ul>
            </td>
        </tr>
        """
        
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predictive Maintenance Audit Report - House {house_id}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-base: #f9fbfd;
            --bg-card: #ffffff;
            --border-color: #e3e8ee;
            --text-main: #333c48;
            --text-muted: #6b7c93;
            --primary: #388bfd;
            --success: #24b47e;
            --warning: #ff9f1c;
            --danger: #ff5c5c;
            --critical: #d12424;
        }}
        body {{
            background-color: var(--bg-base);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 40px 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.04);
            padding: 40px;
        }}
        header {{
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 24px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        header h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 26px;
            margin: 0;
            color: #1a202c;
        }}
        header p {{
            margin: 4px 0 0 0;
            font-size: 13px;
            color: var(--text-muted);
        }}
        .meta-stamp {{
            text-align: right;
            font-size: 12px;
            color: var(--text-muted);
            font-family: monospace;
        }}
        .kpi-row {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }}
        .kpi-card {{
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            background: #fafcfd;
        }}
        .kpi-card h3 {{
            margin: 0 0 8px 0;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
        }}
        .kpi-card .value {{
            font-size: 22px;
            font-weight: 700;
            color: #1a202c;
            font-family: 'Outfit', sans-serif;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 14px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
            font-size: 13px;
        }}
        th {{
            background-color: #fafcfd;
            font-weight: 600;
            color: #1a202c;
        }}
        .font-bold {{ font-weight: 600; }}
        .font-mono {{ font-family: monospace; }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .badge-healthy {{ background-color: #e2f5ec; color: var(--success); }}
        .badge-minor {{ background-color: #fff4e5; color: var(--warning); }}
        .badge-moderate {{ background-color: #fff4e5; color: var(--warning); }}
        .badge-critical {{ background-color: #ffebeb; color: var(--danger); }}
        
        .progress-bar-container {{
            position: relative;
            background: #e9ecef;
            border-radius: 4px;
            height: 18px;
            overflow: hidden;
            width: 120px;
        }}
        .progress-bar-fill {{
            background: var(--success);
            height: 100%;
        }}
        .progress-bar-text {{
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            font-size: 9px;
            font-weight: 700;
            color: #333;
        }}
        .recs-list {{
            margin: 0;
            padding-left: 16px;
            font-size: 11px;
            color: #4a5568;
        }}
        .cause-text {{
            font-size: 12px;
            color: #2d3748;
            font-weight: 500;
        }}
        footer {{
            margin-top: 40px;
            text-align: center;
            font-size: 11px;
            color: var(--text-muted);
            border-top: 1px solid var(--border-color);
            padding-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>Predictive Maintenance Audit Report</h1>
                <p>Spatiotemporal Graph Learning &amp; AI-Driven Failure Forecasting</p>
            </div>
            <div class="meta-stamp">
                Generated: {timestamp}<br>
                Source: House {house_id} Database
            </div>
        </header>
        
        <div class="kpi-row">
            <div class="kpi-card">
                <h3>Target Household</h3>
                <div class="value">House {house_id}</div>
            </div>
            <div class="kpi-card">
                <h3>Monitored Appliances</h3>
                <div class="value">{len(appliances)}</div>
            </div>
            <div class="kpi-card">
                <h3>System Status</h3>
                <div class="value" style="color: {'var(--danger)' if any(a.get('health') < 70 for a in appliances) else 'var(--success)'}">
                    {'Alert (Failing)' if any(a.get('health') < 70 for a in appliances) else 'Healthy'}
                </div>
            </div>
            <div class="kpi-card">
                <h3>Graph Drift Score</h3>
                <div class="value">{graph_drift:.4f}</div>
            </div>
        </div>
        
        <h2>Appliance Health &amp; Failure Analysis</h2>
        <table>
            <thead>
                <tr>
                    <th>Appliance</th>
                    <th>Health Index</th>
                    <th>Severity Status</th>
                    <th>Model Confidence</th>
                    <th>Est. Useful Life</th>
                    <th>Probable Cause</th>
                    <th>Actionable Recommendations</th>
                </tr>
            </thead>
            <tbody>
                {table_rows_html}
            </tbody>
        </table>
        
        <footer>
            <p>NILM Predictor Early Fault Detection System • Spatiotemporal Graph Transformer &amp; Research-Grade Diagnostics • 2026</p>
        </footer>
    </div>
</body>
</html>
"""
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Exported HTML audit report to {out_path}")
