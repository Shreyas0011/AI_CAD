# MP_Lung: AI-Powered Lung Disease Detection

This project uses deep learning (EfficientNetB0) to detect **Normal**, **Pneumonia**, and **Tuberculosis** from chest X-ray images.

## Project Structure

```
MP_Lung/
├── dataset/         # Dataset with train, val, test folders
├── backend/         # FastAPI backend
├── frontend/        # React + Vite + Tailwind frontend
├── train.py         # Model training script
├── requirements.txt # Python dependencies
└── README.md        # This file
```

## Setup & Training

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Dataset**:
   Organize your images in the following structure:
   ```
   dataset/
   ├── train/
   │   ├── Normal/
   │   ├── Pneumonia/
   │   └── Tuberculosis/
   ├── val/
   │   └── ...
   └── test/
       └── ...
   ```

3. **Train the Model**:
   ```bash
   python train.py
   ```
   This will save the trained model as `lung_model.h5`.

## Running the Application

### 1. Start Backend
```bash
cd backend
uvicorn main:app --reload
```
The API will be available at `http://localhost:8000`.

### 2. Start Frontend
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

## Features
- **Modern Medical UI**: Sleek dark-blue/cyan theme with glassmorphism.
- **Drag & Drop**: Easy X-ray image upload.
- **Instant Analysis**: Real-time prediction with confidence scores.
- **Responsive**: Works on desktop and mobile.
- **Secure**: Includes medical disclaimers and error handling.

## Medical Disclaimer
This system is an AI-assisted screening tool and not a replacement for professional medical diagnosis. Always consult a qualified healthcare provider for clinical decisions.
