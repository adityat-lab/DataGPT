
import tomllib
import time
from google import genai
from google.genai import types

# Load API key
with open(".streamlit/secrets.toml", "rb") as f:
    api_key = tomllib.load(f)["GEMINI_API_KEY"]

print("Connecting to Gemini 3.8...")

# Limit timeout and disable repeated retries
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(
        timeout=30000,
        retry_options=types.HttpRetryOptions(
            attempts=1
        )
    )
)

print("Sending request...")

start = time.time()

try:
    response = client.interactions.create(
        model="gemini-3.8-flash",
        input="Reply with exactly: DataGPT is working",
        automatic_continuation=False
    )

    print("\nSUCCESS!")
    print("Gemini response:", response.output_text)

except Exception as e:
    print("\nREQUEST FAILED")
    print("Error type:", type(e).__name__)
    print("Error details:", str(e))

finally:
    print("Time taken:", round(time.time() - start, 2), "seconds")
