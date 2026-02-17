import streamlit as st
from model_helper import predict


st.set_page_config(page_title= "Automated Car Damage Detection", page_icon="🚘", layout="centered")

st.title("🚘 Automated Car Damage Detection")


uploaded_file = st.file_uploader("Upload the Photo", type=['jpg', 'png'])

if uploaded_file:
    image_path = "temp_file.jpg"
    with open(image_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
        st.image(uploaded_file, caption= "Uploaded File", use_container_width= True)
        
        prediction = predict(image_path)
        st.info(f"Predicted Class: {prediction}")

        
