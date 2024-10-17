import streamlit as st
import requests
import os

# Function to call the Hugging Face API
def call_hugging_face_api(text):
    api_url = "https://api-inference.huggingface.co/models/BAAI/Emu3-Gen"
    headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

    # Send a POST request to the Hugging Face API with the appropriate input
    response = requests.post(api_url, headers=headers, json={"inputs": text})
    return response.json()  # Return the API response as JSON

# Streamlit app
st.title("Hugging Face Model Integration")

# Input box for text
input_text = st.text_area("Enter text to generate next tokens:")

if st.button("Generate"):
    with st.spinner("Generating..."):
        result = call_hugging_face_api(input_text)

        # Display results
        if 'generated_text' in result:
            st.write("Generated Text:")
            st.write(result['generated_text'])
        else:
            st.error("Error: " + str(result))
