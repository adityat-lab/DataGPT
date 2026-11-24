
import streamlit as st
from google import genai

api_key = st.secrets["GEMINI_API_KEY"]

client = genai.Client(api_key=api_key)

print("\nAVAILABLE GEMINI MODELS\n")

try:
    for model in client.models.list():
        print(model.name)

except Exception as e:
    print("ERROR:", e)

finally:
    client.close()
