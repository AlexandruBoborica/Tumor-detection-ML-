import streamlit as st
from PIL import Image
from ultralytics import YOLO

# --- CONFIG ---
st.set_page_config(page_title="Multi-Model Tumor Detector", layout="wide")
st.title("🧠 Brain Tumor Detection: Multi-Model Comparison")

# --- Define Models ---
model_paths = {
    "YOLOv8 Tumor Model": "runs1/train/yolo_tumor_finetune2/weights/best.pt",
    "YOLOv8n Baseline": "runs2/train/yolo_tumor_finetune/weights/best.pt",
    "YOLOv8m BRATS": "runs/train/yolo_medical_seg_brats/weights/best.pt",
    "YOLOv8m ISLES": "runs/train/yolo_medical_seg_isles/weights/best.pt"
}

# --- Load Models ---
st.sidebar.header("📦 Models Loaded")
models = {}
for name, path in model_paths.items():
    try:
        models[name] = YOLO(path)
        st.sidebar.success(name)
    except Exception as e:
        st.sidebar.error(f"{name} - Error: {e}")

# --- Sidebar Controls ---
st.sidebar.markdown("### 🖼️ Upload MRI Image")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
run_button = st.sidebar.button("🔍 Run Detection")

# --- Main Logic ---
if uploaded_file:
    image = Image.open(uploaded_file)
    st.subheader("🖼️ Original Uploaded Image")
    st.image(image, width=300)  # Smaller display of uploaded image

    if run_button:
        st.subheader("📊 Detection Results")

        cols = st.columns(len(models))
        detection_results = {}

        for i, (name, model) in enumerate(models.items()):
            with st.spinner(f"Running {name}..."):
                result = model.predict(image, conf=conf_threshold)[0]
                detection_results[name] = result

            with cols[i]:
                st.markdown(f"**{name}**")
                result_img = result.plot()
                st.image(result_img, use_container_width=True, caption="Detection")

        st.divider()
        st.subheader("📋 Detection Details")

        for name, result in detection_results.items():
            boxes = result.boxes
            st.markdown(f"#### 🔍 {name}")
            if len(boxes) == 0:
                st.warning("No tumors detected.")
            else:
                st.success(f"Detected {len(boxes)} region(s).")
                for i, box in enumerate(boxes):
                    coords = box.xyxy.cpu().numpy()[0]
                    conf = box.conf.cpu().numpy()[0]
                    st.write(f"""
                        **Box {i+1}**  
                        - x1: {coords[0]:.1f}, y1: {coords[1]:.1f}  
                        - x2: {coords[2]:.1f}, y2: {coords[3]:.1f}  
                        - Confidence: {conf:.2f}
                    """)
else:
    st.info("👈 Please upload an MRI image from the sidebar to begin.")
