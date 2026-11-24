
import tomllib
import requests
import time

# Load API key
with open(".streamlit/secrets.toml", "rb") as f:
    api_key = tomllib.load(f)["GEMINI_API_KEY"]

url = "https://generativelanguage.googleapis.com/v1beta/interactions"

headers = {
    "Content-Type": "application/json",
    "x-goog-api-key": api_key
}

payload = {
    "model": "gemini-3.5-flash-lite",
    "input": "Reply with exactly: DataGPT is working"
}

print("Testing Gemini REST API...")
start = time.time()

try:
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=(10, 45)
    )

    print("\nHTTP STATUS:", response.status_code)
    print("RESPONSE:")
    print(response.text[:5000])

except requests.exceptions.Timeout:
    print("\nERROR: REST API request timed out.")

except requests.exceptions.ConnectionError as e:
    print("\nCONNECTION ERROR:", str(e))

except Exception as e:
    print("\nERROR:", type(e).__name__, str(e))

finally:
    print("\nTime:", round(time.time() - start, 2), "seconds")
