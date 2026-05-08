import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
DATASET_DIR = 'dataset'
MODEL_SAVE_PATH = 'lung_model.h5'

from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

def create_model(num_classes):
    # Load base model - EfficientNetB3 is more powerful than B0
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Initially freeze base model
    base_model.trainable = False
    
    # Add high-complexity custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Layer 1
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Layer 2
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Layer 3
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Use Nadam optimizer for better convergence
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model, base_model

import cv2

def medical_preprocessing(img):
    # Convert to grayscale for CLAHE then back to RGB
    img = img.astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(gray)
    # Return as 3-channel RGB for EfficientNet
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB).astype(np.float32)

def train():
    # Data Augmentation with Medical Preprocessing
    train_datagen = ImageDataGenerator(
        preprocessing_function=medical_preprocessing,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    num_classes = len(train_generator.class_indices)
    
    # Build Model
    model, base_model = create_model(num_classes)

    # Advanced Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor='val_accuracy'),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=4, min_lr=1e-7, monitor='val_loss'),
        tf.keras.callbacks.ModelCheckpoint('best_lung_model.h5', save_best_only=True)
    ]

    # Phase 1: Transfer Learning
    print("Phase 1: Initial training of deep dense head...")
    model.fit(train_generator, epochs=10, validation_data=val_generator, callbacks=callbacks)

    # Phase 2: Deep Fine-tuning (Unfreeze top 50 layers)
    print("Phase 2: Deep Fine-tuning EfficientNetB3...")
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    history = model.fit(
        train_generator,
        epochs=30, # Increased epochs for deep fine-tuning
        validation_data=val_generator,
        callbacks=callbacks
    )

    # Print accuracy
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    print(f"\nFinal Training Accuracy: {train_acc:.4f}")
    print(f"Final Validation Accuracy: {val_acc:.4f}")

    # Evaluate on Test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Final Test Accuracy: {test_acc:.4f}")

    # Save model
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
