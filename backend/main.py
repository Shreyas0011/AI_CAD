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

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 1. Standardize to Grayscale then back to 3-channel (Medical Standard)
        gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        
        # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This is the gold standard for enhancing X-ray features
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        equalized = clahe.apply(gray)
        
        # 3. Noise Reduction (Bilateral Filter preserves edges)
        denoised = cv2.bilateralFilter(equalized, 9, 75, 75)
        
        # 4. Prepare for Model
        img_standardized = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
        img_resized = cv2.resize(img_standardized, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        
        img_array_exp = np.expand_dims(img_resized, axis=0) / 255.0
        
        # Prediction
        preds = model.predict(img_array_exp)
        idx = np.argmax(preds[0])
        
        # Find last conv layer for Grad-CAM
        last_conv_layer = "top_activation"
        
        # Generate Bio-marker Heatmap
        heatmap = get_gradcam_heatmap(model, img_array_exp, last_conv_layer, idx)
        heatmap_base64 = apply_heatmap(img_resized, heatmap)

        return {
            "prediction": CLASS_NAMES[idx],
            "confidence": round(float(preds[0][idx]) * 100, 2),
            "bio_marker": f"data:image/png;base64,{heatmap_base64}",
            "all_scores": {CLASS_NAMES[i]: round(float(preds[0][i]) * 100, 2) for i in range(len(CLASS_NAMES))}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
