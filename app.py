import streamlit as st
import requests
import os

# Function to call the GPT-2 API
def call_gpt2_api(text):
    api_url = "https://api-inference.huggingface.co/models/gpt2"
    headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

    # Send a POST request to the Hugging Face API
    response = requests.post(api_url, headers=headers, json={"inputs": text})

    # Check if the response is successful
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}

# Streamlit app
st.set_page_config(page_title="Hugging Face Model Integration", page_icon="🤖")
st.markdown(
    """
    <style>
        body {
            background-color: #f0f2f5;
            font-family: 'Arial', sans-serif;
        }
        h1 {
            color: #4a90e2;
            text-align: center;
        }
        .text-area {
            border: 2px solid #4a90e2;
            border-radius: 5px;
            padding: 10px;
            width: 100%;
        }
        .button {
            background-color: #4a90e2;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 15px;
            cursor: pointer;
        }
        .button:hover {
            background-color: #357ab8;
        }
    </style>
    """, unsafe_allow_html=True
)

# Header
st.title("Hugging Face Model Integration")

# Input box for text
input_text = st.text_area("Enter text to generate next tokens:", height=150)

if st.button("Generate"):
    with st.spinner("Generating..."):
        result = call_gpt2_api(input_text)

        # Display results
        if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
            generated_text = result[0]['generated_text']  # Extract the generated text
            st.write("### Generated Text:")
            st.write(generated_text)  # Display the extracted text
        else:
            st.error("Error: " + str(result.get('error', 'An unexpected error occurred.')))
