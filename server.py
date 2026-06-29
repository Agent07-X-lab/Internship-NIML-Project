import os
import re
import json
import socketserver
from http.server import SimpleHTTPRequestHandler
from predict import run_prediction

PORT = 8000
OUTPUT_DIR = "3_processed_outputs"

class DashboardAPIHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        # Default route serves the dashboard HTML
        if self.path == '/' or self.path == '/index.html':
            self.path = '/Predictive_Maintenance_Dashboard.html'
        return super().do_GET()

    def do_POST(self):
        if self.path == '/predict':
            content_type = self.headers.get('Content-Type', '')
            content_length = int(self.headers.get('Content-Length', 0))
            
            if not content_type.startswith('multipart/form-data') or content_length == 0:
                self.send_error_response(400, "Invalid content type or empty body.")
                return
                
            # Read request body
            raw_body = self.rfile.read(content_length)
            
            # Find boundary
            boundary_match = re.search(r'boundary=(.+)', content_type)
            if not boundary_match:
                self.send_error_response(400, "Multipart boundary not found.")
                return
            boundary = boundary_match.group(1).encode('utf-8')
            
            # Parse body parts
            parts = raw_body.split(b'--' + boundary)
            file_bytes = None
            filename = "temp_upload.csv"
            
            for part in parts:
                if b'filename=' in part:
                    # Parse headers and payload from part
                    parts_split = part.split(b'\r\n\r\n', 1)
                    if len(parts_split) < 2:
                        continue
                    headers, content = parts_split
                    # Extract original filename if available
                    fn_match = re.search(rb'filename="([^"]+)"', headers)
                    if fn_match:
                        filename = fn_match.group(1).decode('utf-8', errors='ignore')
                    # Strip trailing \r\n
                    if content.endswith(b'\r\n'):
                        content = content[:-2]
                    elif content.endswith(b'\r\n--'):
                        content = content[:-4]
                    file_bytes = content
                    break
                    
            if not file_bytes:
                self.send_error_response(400, "No file uploaded in the request.")
                return
                
            print(f"Received file upload: {filename} ({len(file_bytes)} bytes)")
            
            # Save uploaded bytes to a temporary location
            temp_path = os.path.join(OUTPUT_DIR, f"temp_{filename}")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(temp_path, 'wb') as f:
                f.write(file_bytes)
                
            try:
                # Run prediction
                report = run_prediction(temp_path, OUTPUT_DIR, epochs=15)
                
                # Cleanup temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
                # Send JSON response
                response_data = json.dumps(report).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(response_data)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(response_data)
                print(f"Successfully processed prediction for {filename}.")
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                self.send_error_response(500, f"Error processing prediction model: {e}")
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
    # Allow port reuse so restarting server doesn't fail
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), DashboardAPIHandler) as httpd:
        print(f"Server started at http://localhost:{PORT}")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")

if __name__ == '__main__':
    main()
