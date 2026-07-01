import urllib.request
import os
import json

def main():
    url = "http://localhost:8000/predict"
    file_path = "1_raw_data/CLEAN_House6.csv"
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # Construct multipart boundary and body
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    boundary_bytes = boundary.encode('utf-8')

    with open(file_path, 'rb') as f:
        file_content = f.read()

    body = (
        b"--" + boundary_bytes + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="CLEAN_House6.csv"\r\n'
        b"Content-Type: text/csv\r\n\r\n"
        + file_content + b"\r\n"
        b"--" + boundary_bytes + b"--\r\n"
    )

    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body))
    }

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")

    try:
        print(f"Uploading {file_path} to {url}...")
        with urllib.request.urlopen(req) as response:
            response_bytes = response.read()
            print("\nResponse Status Code:", response.status)
            resp_json = json.loads(response_bytes.decode('utf-8'))
            print("\nResponse Keys:", list(resp_json.keys()))
            print("House ID:", resp_json.get("house_id"))
            print("Appliances:", resp_json.get("appliance_names"))
            print("Statuses:", resp_json.get("statuses"))
            print("\nAPI Upload Verification SUCCESSFUL!")
    except Exception as e:
        print("API Upload Verification FAILED:", e)

if __name__ == "__main__":
    main()
