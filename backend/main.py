import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
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
MODEL_PATH = os.getenv("MODEL_PATH", "../lung_model.h5")
IMG_SIZE = 224
CLASS_NAMES = ["Normal", "Pneumonia", "Tuberculosis"]

# Load model globally
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def apply_heatmap(img_path_or_array, heatmap, intensity=0.8, res=224):
    heatmap = cv2.resize(heatmap, (res, res))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    img = img_path_or_array
    if isinstance(img, np.ndarray):
        img = np.uint8(255 * img)
        if img.shape[0] == 1: img = img[0]
    
    superimposed_img = heatmap * intensity + img
    superimposed_img = np.minimum(superimposed_img, 255).astype(np.uint8)
    
    _, buffer = cv2.imencode('.png', cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    return base64.b64encode(buffer).decode('utf-8')

# 11. ENSURE PREDICTION PIPELINE MATCHES TRAINING
CLASS_NAMES = ["Normal", "Pneumonia", "Tuberculosis"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 5. VERIFY IMAGE PREPROCESSING (Match Training)
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        
        # Normalization: image = image / 255.0
        img_array_exp = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0
        
        # Prediction
        preds = model.predict(img_array_exp)
        
        # 6. IMPROVE PREDICTION DEBUGGING (Print raw probabilities)
        print("\n--- Raw Prediction Probabilities ---")
        for i, name in enumerate(CLASS_NAMES):
            print(f"{name}: {preds[0][i]:.4f}")
            
        idx = np.argmax(preds[0])
        
        # Generate Bio-marker Heatmap (Grad-CAM)
        last_conv_layer = "top_activation"
        heatmap = get_gradcam_heatmap(model, img_array_exp, last_conv_layer, idx)
        heatmap_base64 = apply_heatmap(img_resized, heatmap)

        return {
            "prediction": CLASS_NAMES[idx],
            "confidence": round(float(preds[0][idx]) * 100, 2),
            "bio_marker": f"data:image/png;base64,{heatmap_base64}",
            "all_scores": {CLASS_NAMES[i]: round(float(preds[0][i]) * 100, 2) for i in range(len(CLASS_NAMES))}
        }

    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
