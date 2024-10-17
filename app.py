import streamlit as st
import requests
import os
import time

# Function to call the GPT-Neo API with retry logic
def call_gptneo_api(text, retries=3):
    api_url = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-125M"
    headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

    for attempt in range(retries):
        response = requests.post(api_url, headers=headers, json={"inputs": text})
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:  # Service Unavailable
            st.warning("Model is currently loading, retrying...")
            time.sleep(5)  # Wait before retrying
        else:
            return response.json()  # Return the error response

    return {"error": "Failed to get a response from the model after multiple attempts."}

# Streamlit app
st.title("Hugging Face GPT-Neo Text Generation")

# Input box for text
input_text = st.text_area("Enter text to generate next tokens:")

if st.button("Generate"):
    with st.spinner("Generating..."):
        result = call_gptneo_api(input_text)

        # Display results
        if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
            generated_text = result[0]['generated_text']  # Extract the generated text
            st.write("Generated Text:")
            st.write(generated_text)  # Display the extracted text
        else:
            st.error("Error: " + str(result))
