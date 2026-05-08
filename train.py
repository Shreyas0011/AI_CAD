import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# 14. VERIFY PREPROCESSING CONSISTENCY
CLASS_NAMES = ["Normal", "Pneumonia", "Tuberculosis"]
IMG_SIZE = 300 # 2. INCREASE IMAGE RESOLUTION
BATCH_SIZE = 16 # Reduced batch size for B3 and higher resolution
EPOCHS = 50 # 3. TRAIN LONGER
DATASET_DIR = 'dataset'
MODEL_SAVE_PATH = 'lung_model.h5'

def verify_dataset_quality():
    print("\n--- 9. VERIFY DATASET QUALITY ---")
    corrupted = 0
    duplicates = set()
    
    for cls in CLASS_NAMES:
        path = os.path.join(DATASET_DIR, cls)
        if not os.path.exists(path): continue
        
        files = os.listdir(path)
        for f in files:
            f_path = os.path.join(path, f)
            # Check for corrupted files
            try:
                img = Image.open(f_path)
                img.verify()
                
                # Check for duplicates (Simple size/name check for demo)
                file_sig = (os.path.getsize(f_path), f)
                if file_sig in duplicates:
                    print(f"Warning: Potential duplicate detected: {f}")
                duplicates.add(file_sig)
                
            except Exception as e:
                print(f"Warning: Corrupted file detected: {f_path} ({e})")
                corrupted += 1
    
    if corrupted > 0:
        print(f"!!! Warning: {corrupted} corrupted files detected !!!")

def get_data_generators():
    # 5. ADD MORE ADVANCED AUGMENTATION
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20, # RandomRotation(0.2)
        zoom_range=0.2,    # RandomZoom(0.2)
        width_shift_range=0.2, # RandomTranslation
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2], # RandomContrast(0.2)
        fill_mode='nearest',
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        subset='validation'
    )

    return train_generator, val_generator

def calculate_class_weights(train_generator):
    # 4. USE CLASS WEIGHTS
    print("\n--- 10. DISPLAY CLASS DISTRIBUTION ---")
    labels = train_generator.classes
    class_indices = train_generator.class_indices
    unique, counts = np.unique(labels, return_counts=True)
    
    for i, count in zip(unique, counts):
        print(f"{CLASS_NAMES[i]}: {count} images")

    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    return dict(enumerate(weights))

def build_advanced_model(num_classes):
    # 1. USE A STRONGER MODEL (EfficientNetB3)
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # 6. ADD BATCH NORMALIZATION
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model, base_model

def evaluate_model(model, val_generator):
    # 11. GENERATE CONFUSION MATRIX & REPORT
    print("\n--- 11. GENERATE CONFUSION MATRIX ---")
    val_generator.reset()
    Y_pred = model.predict(val_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = val_generator.classes

    # Classification Report
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap='Blues')
    plt.title('V2 Advanced Confusion Matrix (Normal vs TB focus)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix_v2.png')
    print("Confusion matrix saved as confusion_matrix_v2.png")

def train_v2():
    verify_dataset_quality()
    train_gen, val_gen = get_data_generators()
    class_weights = calculate_class_weights(train_gen)
    
    model, base_model = build_advanced_model(len(CLASS_NAMES))

    # 3. & 7. Callbacks
    callbacks = [
        EarlyStopping(patience=7, restore_best_weights=True, monitor='val_accuracy'),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-8)
    ]

    print("\n--- Phase 1: Transfer Learning (EfficientNetB3) ---")
    model.fit(
        train_gen,
        epochs=15,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # 8. IMPROVE FINE-TUNING
    print("\n--- Phase 2: Ultra Fine-Tuning (Unfreeze Entire Network) ---")
    base_model.trainable = True
    # Freeze only very earliest layers
    for layer in base_model.layers[:50]:
        layer.trainable = False
        
    model.compile(optimizer=Adam(1e-6), loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks
    )

    print("\n--- Training Complete ---")
    evaluate_model(model, val_gen)

if __name__ == "__main__":
    train_v2()
