import streamlit as st
import requests
import os

# Function to call GPT-2 API
def call_gpt2_api(text):
    api_url = "https://api-inference.huggingface.co/models/gpt2"
    headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}
    response = requests.post(api_url, headers=headers, json={"inputs": text})
    return response.json()

# Streamlit app
st.set_page_config(page_title="Hugging Face Model Integration", page_icon="🤖")
st.markdown('<link rel="stylesheet" href="style.css">', unsafe_allow_html=True)

# Header
st.title("Hugging Face Model Integration")

# Input box for text
input_text = st.text_area("Enter text to generate next tokens:", height=150, key="input_text", help="Type your text here", css_class="text-area")

if st.button("Generate", key="generate_button", help="Click to generate text", css_class="button"):
    with st.spinner("Generating..."):
        result = call_gpt2_api(input_text)

        # Display results
        if 'generated_text' in result:
            st.write("Generated Text:")
            st.write(result['generated_text'])
        else:
            st.error("Error: " + str(result))
