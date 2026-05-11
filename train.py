import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Concatenate, GlobalMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image

# Enable Multi-Core & XLA for Supersonic Speed
tf.config.optimizer.set_jit(True)

# --- CONFIGURATION ---
CLASS_NAMES = ["normal", "pneumonia", "tuberculosis"]
IMG_SIZE = 128 # Supersonic Resolution
BATCH_SIZE = 128 # Maximum Batching
EPOCHS = 30
TRAIN_DIR = r'C:\Users\Shreyas\Downloads\archive\train'
TEST_DIR  = r'C:\Users\Shreyas\Downloads\archive\test'
MODEL_SAVE_PATH = 'lung_model.h5'

def apply_clahe(img):
    """Apply Contrast Limited Adaptive Histogram Equalization for X-ray enhancement."""
    img = np.uint8(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Check if image is grayscale or RGB
    if len(img.shape) == 2:
        return clahe.apply(img)
    else:
        # Convert to LAB to apply CLAHE on Lightness channel
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l2 = clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def medical_preprocessing(img):
    """Custom preprocessing for Medical X-Rays."""
    # 1. CLAHE Enhancement
    img = apply_clahe(img)
    # 2. Normalization
    img = img.astype('float32') / 255.0
    return img

from tensorflow.keras.applications import MobileNetV3Small

def get_fast_dataset():
    # Use modern tf.data pipeline for maximum CPU throughput
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        class_names=CLASS_NAMES,
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        class_names=CLASS_NAMES,
        shuffle=False
    )

    # Apply medical enhancement and prefetching
    def preprocess(image, label):
        # Image is already resized by image_dataset_from_directory
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    # Map preprocessing and enable Autotune Prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE).cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE).cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds

def build_highly_accurate_model(num_classes):
    print(f"\n--- Building Supersonic MobileNetV3 Backbone ({IMG_SIZE}x{IMG_SIZE}) ---")
    base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False

    x = base_model.output
    x = Concatenate()([GlobalAveragePooling2D()(x), GlobalMaxPooling2D()(x)])
    
    x = BatchNormalization()(x)
    x = Dense(512, activation='swish')(x) # Streamlined for CPU speed
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model, base_model
    
    # Deep Diagnostic Neck
    x = BatchNormalization()(x)
    x = Dense(1024, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(512, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model, base_model

def train_ultra_accurate():
    print(f"Training data  : {TRAIN_DIR}")
    print(f"Validation data: {TEST_DIR}")
    
    train_ds, val_ds = get_fast_dataset()
    
    model, base_model = build_highly_accurate_model(len(CLASS_NAMES))

    callbacks = [
        EarlyStopping(patience=7, restore_best_weights=True, monitor='val_accuracy', verbose=1),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-7, verbose=1)
    ]

    print("\n========================================")
    print("Phase 1: Training classification head")
    print("========================================")
    model.fit(
        train_ds,
        epochs=15,
        validation_data=val_ds,
        callbacks=callbacks
    )

    print("\n========================================")
    print("Phase 2: Fine-tuning Diagnostic Layers")
    print("========================================")
    base_model.trainable = True
    # Freeze everything except the top 40 layers for CPU speed
    for layer in base_model.layers[:-40]:
        layer.trainable = False
        
    model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks
    )

    print("\n========================================")
    print("Final Model Verification")
    print("========================================")
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"Final System Accuracy: {val_acc*100:.2f}%")
    print(f"Model saved as {MODEL_SAVE_PATH}")
    
    with open("training_done.txt", "w") as f:
        f.write(f"Accuracy: {val_acc*100:.2f}%")

if __name__ == "__main__":
    train_ultra_accurate()
