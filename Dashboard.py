import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Load your trained YOLOv8 classification model
model = YOLO('runs/classify/train/weights/best.pt')

# Define bin color mapping
bin_color = {
    'paper': 'Blue Bin',
    'plastic': 'Blue Bin',
    'metal': 'Yellow Bin',
    'glass': 'Yellow Bin',
    'cardboard': 'Blue Bin',
    'biological': 'Green Bin',
    'battery': 'Red Bin',
    'clothes': 'Green Bin',
    'shoe': 'Green Bin',
    'trash': 'Grey Bin'
}

# Streamlit UI setup
st.set_page_config(page_title="Smart Waste Sorter", page_icon="♻️", layout="centered")

st.title("♻️ Smart Waste Classification System")
st.write("Upload an image of waste and let the model suggest the correct bin color.")

uploaded_file = st.file_uploader("Upload a Waste Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Predict with YOLOv8
    st.write("🔍 Analyzing image...")
    results = model.predict(image, verbose=False)

    # Extract prediction
    probs = results[0].probs
    top_class = results[0].names[probs.top1]
    confidence = float(probs.top1conf) * 100
    bin_suggestion = bin_color.get(top_class.lower(), "Unknown Bin")

    # Display results
    st.success(f"**Predicted Waste Type:** {top_class.capitalize()}")
    st.info(f"**Suggested Bin Color:** {bin_suggestion}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Optional: show all class probabilities
    with st.expander("See all predictions"):
        for cls, conf in zip(results[0].names.values(), probs.data.tolist()):
            st.write(f"{cls.capitalize()}: {conf * 100:.2f}%")

st.markdown("---")
st.caption("Developed using Streamlit and YOLOv8")
