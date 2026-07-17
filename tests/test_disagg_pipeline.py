import urllib.request
import os
import json

def main():
    # 1. Test /disaggregate
    url_disagg = "http://localhost:8000/disaggregate"
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(ROOT_DIR, "1_raw_data", "mock_aggregate.csv")
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # Construct multipart boundary and body
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    boundary_bytes = boundary.encode('utf-8')

    with open(file_path, 'rb') as f:
        file_content = f.read()

    configs = {
        "Kettle": 2200,
        "Washing Machine": 2500,
        "Fridge-Freezer": 150
    }
    configs_json = json.dumps(configs)

    body = (
        b"--" + boundary_bytes + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="mock_aggregate.csv"\r\n'
        b"Content-Type: text/csv\r\n\r\n"
        + file_content + b"\r\n"
        b"--" + boundary_bytes + b"\r\n"
        b'Content-Disposition: form-data; name="configs"\r\n\r\n'
        + configs_json.encode('utf-8') + b"\r\n"
        b"--" + boundary_bytes + b"--\r\n"
    )

    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body))
    }

    req = urllib.request.Request(url_disagg, data=body, headers=headers, method="POST")

    try:
        print(f"Uploading {file_path} to {url_disagg}...")
        with urllib.request.urlopen(req) as response:
            response_bytes = response.read()
            print("Response Status Code:", response.status)
            resp_json = json.loads(response_bytes.decode('utf-8'))
            print("Disaggregate Response Keys:", list(resp_json.keys()))
            assert resp_json.get("success") is True, "Disaggregation was not successful"
            print("Disaggregation completed successfully.")
    except Exception as e:
        print("Disaggregation Verification FAILED:", e)
        return

    # 2. Test /pipeline-from-disaggregated
    url_pipe = "http://localhost:8000/pipeline-from-disaggregated"
    req_pipe = urllib.request.Request(url_pipe, method="POST")
    try:
        print(f"Triggering PM pipeline from disaggregated data...")
        with urllib.request.urlopen(req_pipe) as response:
            response_bytes = response.read()
            print("Response Status Code:", response.status)
            resp_json = json.loads(response_bytes.decode('utf-8'))
            print("Pipeline Response Keys:", list(resp_json.keys()))
            print("House ID:", resp_json.get("house_id"))
            print("Appliances:", resp_json.get("appliance_names"))
            print("Statuses:", resp_json.get("statuses"))
            assert resp_json.get("house_id") == 99, "House ID should be 99"
            print("End-to-End Disaggregation & Diagnostic Pipeline Verification SUCCESSFUL!")
    except Exception as e:
        print("Pipeline Verification FAILED:", e)

if __name__ == "__main__":
    main()
