import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# Function to call the Hugging Face API
def call_hugging_face_api(image):
    api_url = "https://api-inference.huggingface.co/models/BAAI/Emu3-Gen"  # Update with your model's API URL
    headers = {"Authorization": f"Bearer YOUR_HUGGING_FACE_API_TOKEN"}
    # Prepare the image for sending
    image_data = image.convert("RGB")
    buffered = BytesIO()
    image_data.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()

    # Send a POST request to the Hugging Face API
    response = requests.post(api_url, headers=headers, files={"file": image_bytes})
    return response.json()  # Return the API response as JSON

# Streamlit app
st.title("Hugging Face Model Integration")

uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate"):
        with st.spinner("Generating..."):
            result = call_hugging_face_api(image)
            # Display results (assuming the response contains image URLs)
            if 'generated_images' in result:
                for idx, img_url in enumerate(result['generated_images']):
                    st.image(img_url, caption=f"Generated Image {idx + 1}", use_column_width=True)
            else:
                st.error("Error: " + str(result))
