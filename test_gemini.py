
import tomllib
import time
from google import genai

with open(".streamlit/secrets.toml", "rb") as f:
    secrets = tomllib.load(f)

api_key = secrets["GEMINI_API_KEY"]

print("Connecting to Gemini...")

client = genai.Client(
    api_key=api_key,
    http_options={"timeout": 30000}
)

print("Sending simple request...")

start = time.time()

try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say exactly: Hello DataGPT"
    )

    print("SUCCESS!")
    print(response.text)

except Exception as e:
    print("ERROR:", type(e).__name__)
    print(str(e))

finally:
    print("Time:", round(time.time() - start, 2), "seconds")
