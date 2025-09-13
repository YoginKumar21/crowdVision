from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import base64
import io

# --------- CONFIG ----------
YOLO_WEIGHTS = "yolov8n.pt"
CONFIDENCE = 0.25
# ---------------------------

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests from the frontend

# Load YOLO once
def load_model():
    return YOLO(YOLO_WEIGHTS)

model = load_model()

def detect_people(img_bgr, heatmap=False, annotate=True):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = model(img_rgb, imgsz=640, conf=CONFIDENCE, verbose=False)
    
    boxes = []
    if results and len(results) > 0:
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls_arr = r.boxes.cls.cpu().numpy().astype(int)
            for i, c in enumerate(cls_arr):
                if c == 0:  # Class '0' is for 'person' in COCO dataset
                    boxes.append(xyxy[i].astype(int).tolist())

    person_count = len(boxes)
    annotated = img_bgr.copy()

    # Heatmap and annotation logic
    if heatmap and person_count > 0:
        h, w = img_bgr.shape[:2]
        heatmap_mask = np.zeros((h, w), dtype=np.float32)
        for (x1, y1, x2, y2) in boxes:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(heatmap_mask, (cx, cy), 60, 1, -1)
        heatmap_mask = cv2.GaussianBlur(heatmap_mask, (99, 99), 0)
        if np.max(heatmap_mask) > 0:
            heatmap_mask = np.uint8(255 * heatmap_mask / np.max(heatmap_mask))
            heatmap_color = cv2.applyColorMap(heatmap_mask, cv2.COLORMAP_JET)
            annotated = cv2.addWeighted(annotated, 0.6, heatmap_color, 0.6, 0)

    if annotate:
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(annotated, f"Count: {person_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    # Encode the annotated image to a base64 string
    _, buffer = cv2.imencode('.png', annotated)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return person_count, img_base64

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Read image from file stream
    img_stream = io.BytesIO(file.read())
    img_pil = Image.open(img_stream).convert("RGB")
    img_np = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Get optional parameters
    heatmap = request.form.get('heatmap') == 'true'

    # Perform detection
    count, annotated_img_base64 = detect_people(img_bgr, heatmap=heatmap)
    
    return jsonify({
        "count": count,
        "image": annotated_img_base64
    })

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=5000)
