import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import cv2
import base64

app = FastAPI(title="AI Powered CAD API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
# Use absolute path relative to script location to avoid CWD issues
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "lung_model.h5")
IMG_SIZE = 128 # Matches Supersonic MobileNetV3 training
CLASS_NAMES = ["normal", "pneumonia", "tuberculosis"]

def build_model(num_classes):
    base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model.output
    
    # Matching the 'Deep Diagnostic Neck' from train.py (with corrected BN count)
    x = Concatenate()([GlobalAveragePooling2D()(x), GlobalMaxPooling2D()(x)])
    x = BatchNormalization()(x) 
    
    x = Dense(1024, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(512, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    predictions = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=predictions)

# Load model globally
model = None

@app.on_event("startup")
async def startup_event():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            print(f"LOADING: Starting model load sequence from {MODEL_PATH}...", flush=True)
            model = build_model(len(CLASS_NAMES))
            model.load_weights(MODEL_PATH)
            print(f"SUCCESS: Model weights loaded from {MODEL_PATH}", flush=True)
        except Exception as e:
            print(f"CRITICAL ERROR loading model from {MODEL_PATH}: {e}", flush=True)
    else:
        print(f"CRITICAL ERROR: Model file NOT FOUND at {MODEL_PATH}", flush=True)

def apply_clahe(img):
    """Apply Contrast Limited Adaptive Histogram Equalization for X-ray enhancement."""
    img = np.uint8(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    if len(img.shape) == 2:
        return clahe.apply(img)
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l2 = clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def apply_heatmap(img_rgb, heatmap, intensity=0.5):
    # Rescale heatmap to a range 0-255
    heatmap = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    
    # Use jet colormap to colorize heatmap
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose the heatmap on original image
    superimposed_img = heatmap_colored * intensity + img_rgb
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    # Convert back to BGR for encoding (cv2 expects BGR)
    img_bgr = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')

# 11. ENSURE PREDICTION PIPELINE MATCHES TRAINING
CLASS_NAMES = ["Normal", "Pneumonia", "Tuberculosis"]

def generate_diagnostic_report(base_img, heatmap_img, prediction, confidence, findings):
    """Generates a professional Dual-Scan diagnostic report."""
    # Landscape Report (800x1200)
    report_h = 800
    report_w = 1200
    report = np.ones((report_h, report_w, 3), dtype=np.uint8) * 255 # White background
    
    # 1. Sidebar Header (Medical Dark)
    cv2.rectangle(report, (0, 0), (report_w, 90), (25, 25, 35), -1)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(report, "AI LUNG DIAGNOSTIC REPORT", (40, 60), font, 1.1, (255, 255, 255), 2)
    cv2.putText(report, "OFFICIAL RADIOLOGY ANALYSIS v2.6", (report_w - 380, 60), font, 0.45, (180, 180, 180), 1)
    
    # 2. Dual Scan Area: Original (Left), Heatmap (Right)
    disp_w = 520
    disp_h = 520
    
    # Ensure images are RGB and correctly sized
    orig_disp = cv2.resize(base_img, (disp_w, disp_h))
    heat_disp = cv2.resize(heatmap_img, (disp_w, disp_h))
    if heat_disp.shape[2] == 3:
        # Check if it's BGR from imdecode fallback
        heat_disp = cv2.cvtColor(heat_disp, cv2.COLOR_BGR2RGB)
    
    y_off = 130
    # Original Scan
    cv2.putText(report, "ORIGINAL CLINICAL SCAN", (40, y_off - 10), font, 0.45, (100, 100, 100), 1)
    cv2.rectangle(report, (40, y_off), (40 + disp_w, y_off + disp_h), (240, 240, 240), 1)
    report[y_off:y_off+disp_h, 40:40+disp_w] = orig_disp
    
    # Neural Heatmap
    cv2.putText(report, "NEURAL BIO-MARKER ANALYSIS", (report_w - disp_w - 40, y_off - 10), font, 0.45, (100, 100, 100), 1)
    cv2.rectangle(report, (report_w - disp_w - 40, y_off), (report_w - 40, y_off + disp_h), (240, 240, 240), 1)
    report[y_off:y_off+disp_h, report_w-disp_w-40:report_w-40] = heat_disp
    
    # 3. Clinical Results (Footer Area)
    res_y = y_off + disp_h + 50
    cv2.line(report, (40, res_y - 20), (report_w - 40, res_y - 20), (230, 230, 230), 1)
    
    # Diagnosis Badge
    color = (0, 140, 0) if prediction == "Normal" else (0, 0, 180) 
    cv2.putText(report, "PRIMARY DIAGNOSIS:", (40, res_y), font, 0.5, (100, 100, 100), 1)
    cv2.putText(report, prediction.upper(), (40, res_y + 45), font, 1.4, color, 3)
    
    # Accuracy/Confidence
    cv2.putText(report, "NEURAL CONFIDENCE:", (350, res_y), font, 0.5, (100, 100, 100), 1)
    cv2.putText(report, f"{confidence}%", (350, res_y + 45), font, 1.2, (30, 30, 30), 2)
    
    # Findings Summary
    cv2.putText(report, "CLINICAL FINDINGS:", (650, res_y), font, 0.5, (100, 100, 100), 1)
    for i, f in enumerate(findings[:3]):
        cv2.putText(report, f"> {f}", (650, res_y + 30 + (i * 25)), font, 0.4, (80, 80, 80), 1)
    
    # 4. Final Disclaimer
    cv2.rectangle(report, (0, report_h - 40), (report_w, report_h), (25, 25, 35), -1)
    cv2.putText(report, "CONFIDENTIAL RADIOLOGY OUTPUT | AI CAD SYSTEM v2.6 | VERIFY WITH CLINICAL RADIOLOGIST", (40, report_h - 15), font, 0.35, (180, 180, 180), 1)
    
    _, buffer = cv2.imencode('.png', report)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")
    
    try:
        contents = await file.read()
        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Invalid image format"}
            
        # Convert BGR to RGB (Crucial for model accuracy as training used RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. Apply CLAHE enhancement (Optional: alignment with training)
        # Note: Training script defines this but doesn't use it in the map function.
        # If accuracy is low, it's usually because CLAHE was NOT used during training.
        # img_final = apply_clahe(img_rgb) 
        img_final = img_rgb # Use raw RGB to match training for now
        
        # Preprocessing for Accuracy
        # A. Sharpening filter to enhance lung structural details
        img_blur = cv2.GaussianBlur(img_rgb, (0, 0), 2)
        img_final = cv2.addWeighted(img_rgb, 1.3, img_blur, -0.3, 0)
        
        # B. Resize with INTER_AREA (best for downsampling medical images)
        img_resized = cv2.resize(img_final, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        
        # C. Normalization
        img_array_exp = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0
        
        # 2. Prediction with Advanced TTA (Test-Time Augmentation)
        # We average 4 views: Original, H-Flip, and 2 Translations
        views = []
        
        # Pass 1: Original
        views.append(img_array_exp)
        
        # Pass 2: Horizontal Flip
        img_flipped = cv2.flip(img_resized, 1)
        views.append(np.expand_dims(img_flipped, axis=0).astype(np.float32) / 255.0)
        
        # Pass 3: Slight translation (helps with boundary features)
        M = np.float32([[1, 0, 5], [0, 1, 5]])
        img_shifted = cv2.warpAffine(img_resized, M, (IMG_SIZE, IMG_SIZE))
        views.append(np.expand_dims(img_shifted, axis=0).astype(np.float32) / 255.0)
        
        # Combine all views for robust prediction
        all_preds = []
        for view in views:
            all_preds.append(model.predict(view))
        
        preds = np.mean(all_preds, axis=0)
        
        # 6. IMPROVE PREDICTION DEBUGGING (Print raw probabilities)
        print("\n--- Raw Prediction Probabilities ---")
        for i, name in enumerate(CLASS_NAMES):
            print(f"{name}: {preds[0][i]:.4f}")
            
        idx = np.argmax(preds[0])
        label = CLASS_NAMES[idx]
        confidence = float(preds[0][idx])
        
        try:
            # For MobileNetV3Small, the last conv layer is 'conv_1' (lowercase)
            last_conv_layer = "conv_1"
            heatmap = get_gradcam_heatmap(model, img_array_exp, last_conv_layer, idx)
            heatmap_base64 = apply_heatmap(img_resized, heatmap)
            
            # Analyze heatmap to find "Points of Interest"
            # Divide into quadrants or regions
            h, w = heatmap.shape
            top_half = heatmap[:h//2, :]
            bottom_half = heatmap[h//2:, :]
            left_half = heatmap[:, :w//2]
            right_half = heatmap[:, w//2:]
            
            regions = {
                "Upper Lungs": (np.mean(top_half), (w//2, h//4)),
                "Lower Lungs": (np.mean(bottom_half), (w//2, 3*h//4)),
                "Left Lung": (np.mean(left_half), (w//4, h//2)),
                "Right Lung": (np.mean(right_half), (w//4*3, h//2))
            }
            
            # Sort regions by intensity
            significant_regions = sorted(regions.items(), key=lambda x: x[1][0], reverse=True)
            findings = [f"Abnormal patterns detected in {r[0]}" for r in significant_regions if r[1][0] > 0.3]
            
            if not findings:
                findings = ["No localized abnormalities detected."]
                
            # Generate marker coordinates (normalized 0-100)
            markers = []
            for name, (intensity, (x, y)) in regions.items():
                if intensity > 0.35: # Threshold for showing a marker
                    markers.append({
                        "id": name.lower().replace(" ", "_"),
                        "label": name,
                        "x": round((x / w) * 100, 2),
                        "y": round((y / h) * 100, 2),
                        "intensity": round(float(intensity), 2)
                    })

        except Exception as cam_e:
            print(f"Grad-CAM/Analysis Error: {cam_e}")
            heatmap_base64 = None
            markers = []
            if not findings:
                findings = ["Localized analysis failed."]

        # Final Formatting
        confidence_pct = round(confidence * 100, 2)
        
        # Generate Complete Report (X-ray + Bio-marker + Text)
        report_base64 = None
        if heatmap_base64:
            try:
                # Decode the heatmap to pass it to report generator
                decoded_heatmap = cv2.imdecode(np.frombuffer(base64.b64decode(heatmap_base64), np.uint8), cv2.IMREAD_COLOR)
                report_base64 = generate_diagnostic_report(img_resized, decoded_heatmap, label, confidence_pct, findings)
            except Exception as report_e:
                print(f"Report Generation Error: {report_e}")

        return {
            "prediction": label,
            "confidence": confidence_pct,
            "findings": findings,
            "markers": markers,
            "bio_marker": f"data:image/png;base64,{heatmap_base64}" if heatmap_base64 else None,
            "report": f"data:image/png;base64,{report_base64}" if report_base64 else None,
            "all_scores": {CLASS_NAMES[i]: round(float(preds[0][i]) * 100, 2) for i in range(len(CLASS_NAMES))}
        }

    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
