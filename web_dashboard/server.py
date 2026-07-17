import os
import sys

# Add project root and all subdirectories to sys.path for seamless imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB_DIRS = [
    ROOT_DIR,
    os.path.join(ROOT_DIR, 'data_processing'),
    os.path.join(ROOT_DIR, 'models'),
    os.path.join(ROOT_DIR, 'pm_engine'),
    os.path.join(ROOT_DIR, 'web_dashboard')
]
for sd in SUB_DIRS:
    if sd not in sys.path:
        sys.path.insert(0, sd)

import re
import json
import urllib.parse
import socketserver
from http.server import SimpleHTTPRequestHandler
from pm_pipeline import run_predictive_maintenance_pipeline

PORT = 8000
OUTPUT_DIR = "3_processed_outputs"

class DashboardAPIHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        # Default route serves the dashboard HTML
        if self.path == '/' or self.path == '/index.html':
            self.path = '/Predictive_Maintenance_Dashboard.html'
            return super().do_GET()
            
        # Handler for report exports (Module 14)
        elif self.path.startswith('/export-report'):
            parsed_url = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed_url.query)
            
            fmt = params.get('format', ['json'])[0]
            if fmt == 'csv':
                self.send_error_response(400, "CSV exports are deprecated and no longer supported. Please use JSON or HTML.")
                return
                
            house_id_str = params.get('house_id', ['2'])[0]
            
            try:
                house_id = int(house_id_str)
            except ValueError:
                house_id = 99
                
            filename = f"PM_Report_House_{house_id}.{fmt}"
            filepath = os.path.join(OUTPUT_DIR, filename)
            abs_path = os.path.abspath(filepath)
            
            if os.path.exists(filepath):
                response_data = json.dumps({
                    "path": abs_path,
                    "filename": filename
                }).encode('utf-8')
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(response_data)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(response_data)
                print(f"Report location served: {abs_path}")
            else:
                self.send_error_response(404, f"Report file {filename} not found. Run the predictive maintenance pipeline first.")
            return
            
        return super().do_GET()

    def parse_multipart(self, content_type, content_length):
        raw_body = self.rfile.read(content_length)
        boundary_match = re.search(r'boundary=(.+)', content_type)
        if not boundary_match:
            return {}
        boundary = boundary_match.group(1).encode('utf-8')
        parts = raw_body.split(b'--' + boundary)
        
        parsed = {}
        for part in parts:
            if b'Content-Disposition:' in part:
                parts_split = part.split(b'\r\n\r\n', 1)
                if len(parts_split) < 2:
                    continue
                headers, content = parts_split
                
                # Clean up trailing newlines / boundary markers
                if content.endswith(b'\r\n'):
                    content = content[:-2]
                elif content.endswith(b'\r\n--'):
                    content = content[:-4]
                elif content.endswith(b'--\r\n'):
                    content = content[:-4]
                elif content.endswith(b'--'):
                    content = content[:-2]
                    
                name_match = re.search(rb'name="([^"]+)"', headers)
                if name_match:
                    name = name_match.group(1).decode('utf-8')
                    filename_match = re.search(rb'filename="([^"]+)"', headers)
                    if filename_match:
                        filename = filename_match.group(1).decode('utf-8', errors='ignore')
                        parsed[name] = {"filename": filename, "content": content}
                    else:
                        parsed[name] = content.decode('utf-8', errors='ignore').strip()
        return parsed

    def do_POST(self):
        if self.path == '/predict':
            content_type = self.headers.get('Content-Type', '')
            content_length = int(self.headers.get('Content-Length', 0))
            
            if not content_type.startswith('multipart/form-data') or content_length == 0:
                self.send_error_response(400, "Invalid content type or empty body.")
                return
                
            try:
                parsed = self.parse_multipart(content_type, content_length)
                file_data = parsed.get("file")
                if not file_data:
                    self.send_error_response(400, "No file uploaded in the request.")
                    return
                    
                filename = file_data["filename"]
                file_bytes = file_data["content"]
                
                print(f"Received file upload for PM prediction: {filename} ({len(file_bytes)} bytes)")
                
                temp_path = os.path.join(OUTPUT_DIR, f"temp_{filename}")
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                with open(temp_path, 'wb') as f:
                    f.write(file_bytes)
                    
                report = run_predictive_maintenance_pipeline(temp_path, OUTPUT_DIR, epochs=15)
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
                response_data = json.dumps(report).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(response_data)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(response_data)
                print(f"Successfully processed predictive maintenance pipeline for {filename}.")
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_error_response(500, f"Error processing predictive maintenance pipeline: {e}")
                
        elif self.path == '/disaggregate':
            content_type = self.headers.get('Content-Type', '')
            content_length = int(self.headers.get('Content-Length', 0))
            
            if not content_type.startswith('multipart/form-data') or content_length == 0:
                self.send_error_response(400, "Invalid content type or empty body.")
                return
                
            try:
                parsed = self.parse_multipart(content_type, content_length)
                file_data = parsed.get("file")
                configs_str = parsed.get("configs", "{}")
                
                if not file_data:
                    self.send_error_response(400, "No CSV file uploaded.")
                    return
                    
                appliance_configs = json.loads(configs_str)
                filename = file_data["filename"]
                
                print(f"Received file upload for NILM disaggregation: {filename} ({len(file_data['content'])} bytes)")
                
                temp_path = os.path.join(OUTPUT_DIR, f"temp_nilm_{filename}")
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                with open(temp_path, 'wb') as f:
                    f.write(file_data["content"])
                    
                from nilm_disaggregator import run_nilm_disaggregation
                df_out, stats = run_nilm_disaggregation(temp_path, appliance_configs)
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
                output_filename = f"House_99_Processed.csv"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                
                # Write matching metadata JSON for House 99
                mappings = {}
                thresholds = {}
                app_keys = sorted(appliance_configs.keys())
                for i, app in enumerate(app_keys, 1):
                    sanitized_name = app.replace(" ", "_").replace("-", "_")
                    mappings[f"Appliance{i}"] = sanitized_name
                    thresholds[sanitized_name] = float(appliance_configs[app]) * 0.45
                    
                # Fill remaining up to 9
                for i in range(len(appliance_configs) + 1, 10):
                    mappings[f"Appliance{i}"] = f"Unused_{i}"
                    thresholds[f"Unused_{i}"] = 10.0
                    df_out[f"Unused_{i}"] = 0.0
                    df_out[f"Unused_{i}_ON"] = 0
                    
                # Add state _ON columns for the disaggregated ones
                for app in appliance_configs.keys():
                    col_name = app.replace(" ", "_").replace("-", "_")
                    thresh = float(appliance_configs[app]) * 0.45
                    df_out[f"{col_name}_ON"] = (df_out[col_name] >= thresh).astype(int)
                    
                # Re-save processed CSV with _ON columns and proper indexing
                df_out.to_csv(output_path, index=False)
                
                metadata = {
                    "House": 99,
                    "SourceFile": filename,
                    "TotalRawRows": len(df_out),
                    "ProcessedRows": len(df_out),
                    "StartTime": str(df_out.iloc[0]['Time']) if 'Time' in df_out.columns else "2026-07-13 00:00:00",
                    "EndTime": str(df_out.iloc[-1]['Time']) if 'Time' in df_out.columns else "2026-07-13 06:40:00",
                    "SamplingFrequency": "8s",
                    "Mappings": mappings,
                    "Thresholds": thresholds
                }
                
                with open(os.path.join(OUTPUT_DIR, "House_99_Metadata.json"), 'w') as f:
                    json.dump(metadata, f, indent=4)
                    
                # Convert the first 1000 rows to list of dicts for frontend rendering
                timeseries_sample = df_out.head(1000).to_dict(orient='records')
                
                response_data = json.dumps({
                    "success": True,
                    "stats": stats,
                    "timeseries_sample": timeseries_sample,
                    "disaggregated_csv_path": os.path.abspath(output_path),
                    "disaggregated_csv_filename": output_filename
                }).encode('utf-8')
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(response_data)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(response_data)
                print(f"Successfully processed NILM disaggregation for {filename}.")
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_error_response(500, f"NILM disaggregation failed: {e}")
                
        elif self.path == '/pipeline-from-disaggregated':
            try:
                processed_csv = os.path.join(OUTPUT_DIR, "House_99_Processed.csv")
                if not os.path.exists(processed_csv):
                    self.send_error_response(400, "No disaggregated data available. Run disaggregation first.")
                    return
                    
                print("Running Predictive Maintenance Pipeline from disaggregated House 99 dataset...")
                # Run with 1 epoch for super fast feedback
                report = run_predictive_maintenance_pipeline(processed_csv, OUTPUT_DIR, epochs=1)
                
                response_data = json.dumps(report).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(response_data)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(response_data)
                print("Successfully processed PM pipeline from disaggregated dataset.")
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_error_response(500, f"Error processing pipeline from disaggregated data: {e}")
        else:
            self.send_error_response(404, "Endpoint not found.")

    def send_error_response(self, code, message):
        response_dict = {"error": message}
        response_data = json.dumps(response_dict).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response_data)))
        self.end_headers()
        self.wfile.write(response_data)

def main():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), DashboardAPIHandler) as httpd:
        print(f"Server started at http://localhost:{PORT}")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")

if __name__ == '__main__':
    main()
