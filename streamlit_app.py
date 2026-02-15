import streamlit as st
import requests

st.title("Medicinal Plant Identifier 🌿")

uploaded_file = st.file_uploader("Upload Plant Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width="stretch")


    if st.button("Predict"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://127.0.0.1:8000/predict", files=files)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Plant: {result['plant_name']}")
            st.write(f"Confidence: {result['confidence']}")
            st.write("Medicinal Uses:")
            st.write(result['medicinal_uses'])
        else:
            st.error("Error connecting to API")
