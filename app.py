import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from streamlit_cropper import st_cropper

# --------- CONFIG ----------
YOLO_WEIGHTS = "yolov8n.pt"
CONFIDENCE = 0.25
# ---------------------------

# Load YOLO once
@st.cache_resource
def load_model():
    return YOLO(YOLO_WEIGHTS)

model = load_model()

# Streamlit page settings
st.set_page_config(page_title="YOLO People Counter", page_icon="👥", layout="wide")

# ---- Custom CSS for attractive UI ----
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        background: transparent;
    }
    .title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #ff6a00, #ee0979);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
        animation: fadeIn 2s ease-in-out;
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(-20px);}
        to {opacity: 1; transform: translateY(0);}
    }
    .metric-card {
        padding: 20px;
        border-radius: 16px;
        background: linear-gradient(135deg, #ff6a00, #ee0979);
        color: white;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        box-shadow: 0px 6px 20px rgba(0,0,0,0.4);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 25px;
    }
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0px 10px 25px rgba(0,0,0,0.6);
    }
    .roi-card {
        padding: 20px;
        border-radius: 16px;
        background: linear-gradient(135deg, #56ab2f, #a8e063);
        color: white;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        box-shadow: 0px 6px 20px rgba(0,0,0,0.4);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 25px;
    }
    .roi-card:hover {
        transform: scale(1.05);
        box-shadow: 0px 10px 25px rgba(0,0,0,0.6);
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease-in-out;
    }
    .stButton button:hover {
        transform: scale(1.08);
        background: linear-gradient(135deg, #764ba2, #667eea);
        box-shadow: 0px 6px 15px rgba(0,0,0,0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Title ----
st.markdown("<div class='title'>👥 YOLO People Counter with ROI + Heatmap</div>", unsafe_allow_html=True)

# ---- Sidebar ----
st.sidebar.header("⚙️ Controls")
heatmap = st.sidebar.checkbox("🔥 Show heatmap", value=False)
roi_enable = st.sidebar.checkbox("📐 Enable ROI selection", value=False)

uploaded_file = st.sidebar.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png", "bmp"])

# ---- Detection function ----
def detect_people(img_bgr, heatmap=False, annotate=True):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = model(img_rgb, imgsz=640, conf=CONFIDENCE, verbose=False)
    if len(results) == 0:
        return 0, img_bgr

    r = results[0]
    boxes = []
    try:
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls_arr = r.boxes.cls.cpu().numpy().astype(int)
            for i, c in enumerate(cls_arr):
                if c == 0:
                    boxes.append(xyxy[i].astype(int).tolist())
    except:
        pass

    person_count = len(boxes)
    annotated = img_bgr.copy()

    if heatmap and person_count > 0:
        h, w = img_bgr.shape[:2]
        heatmap_mask = np.zeros((h, w), dtype=np.float32)
        for (x1, y1, x2, y2) in boxes:
            cx, cy = (x1+x2)//2, (y1+y2)//2
            cv2.circle(heatmap_mask, (cx, cy), 60, 1, -1)
        heatmap_mask = cv2.GaussianBlur(heatmap_mask, (99, 99), 0)
        if np.max(heatmap_mask) > 0:
            heatmap_mask = np.uint8(255 * heatmap_mask / np.max(heatmap_mask))
            heatmap_color = cv2.applyColorMap(heatmap_mask, cv2.COLORMAP_JET)
            annotated = cv2.addWeighted(annotated, 0.6, heatmap_color, 0.6, 0)

    if annotate:
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,255), 2)
        cv2.putText(annotated, f"Total: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2, cv2.LINE_AA)

    return person_count, annotated

# ---- Main content ----
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # --- Full Image Detection Layer ---
    st.subheader("🔍 Full Image Detection")
    full_count, full_annot = detect_people(img_array.copy(), heatmap=heatmap)
    st.image(cv2.cvtColor(full_annot, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.markdown(f"<div class='metric-card'>👥 People Count (Full): {full_count}</div>", unsafe_allow_html=True)

    # --- ROI Detection Layer ---
    st.subheader("📐 ROI Detection")
    if roi_enable:
        cropped = st_cropper(image, realtime_update=True, box_color="lime", aspect_ratio=None)
        if cropped is not None:
            crop_arr = np.array(cropped)
            crop_bgr = cv2.cvtColor(crop_arr, cv2.COLOR_RGB2BGR)
            crop_count, crop_annot = detect_people(crop_bgr, heatmap=heatmap)
            st.image(cv2.cvtColor(crop_annot, cv2.COLOR_BGR2RGB), caption=f"ROI People Count: {crop_count}", use_container_width=True)
            st.markdown(f"<div class='roi-card'>🟢 ROI Count: {crop_count}</div>", unsafe_allow_html=True)
    else:
        st.info("✅ Enable ROI selection from the sidebar to crop and detect.")

else:
    st.info("📂 Please upload an image to begin.")
