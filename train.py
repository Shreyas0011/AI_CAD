import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CLASS LABEL CONSISTENCY
CLASS_NAMES = ["Normal", "Pneumonia", "Tuberculosis"]
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
DATASET_DIR = 'dataset'
MODEL_SAVE_PATH = 'lung_model.h5'

def check_dataset_balance():
    print("\n--- 10. VERIFY DATASET BALANCE ---")
    counts = {}
    for cls in CLASS_NAMES:
        path = os.path.join(DATASET_DIR, cls)
        if os.path.exists(path):
            count = len(os.listdir(path))
            counts[cls] = count
            print(f"{cls}: {count} images")
        else:
            print(f"Warning: Directory {path} not found")
    
    if len(counts) > 0:
        max_c = max(counts.values())
        min_c = min(counts.values())
        if min_c < max_c * 0.5:
            print("!!! WARNING: Dataset is heavily imbalanced. Consider oversampling. !!!")

def get_data_generators():
    # 4. IMPROVE DATA AUGMENTATION & 5. NORMALIZATION
    train_datagen = ImageDataGenerator(
        rescale=1./255, # Normalization
        rotation_range=15, # RandomRotation(0.15)
        zoom_range=0.15,   # RandomZoom(0.15)
        horizontal_flip=True, # RandomFlip
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.9, 1.1], # RandomContrast equivalent
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES, # ENSURE CONSISTENT ORDER
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

    # Note: Test set should be separate, but here we use a subset or the same validation for evaluation
    return train_generator, val_generator

def build_model(num_classes):
    # 3. ENABLE FINE-TUNING - Phase 1: Frozen Base
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False 

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model, base_model

def plot_confusion_matrix(model, val_generator):
    # 7. ADD CONFUSION MATRIX
    print("\n--- Generating Confusion Matrix ---")
    val_generator.reset()
    Y_pred = model.predict(val_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = val_generator.classes

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as confusion_matrix.png")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

def train_pipeline():
    check_dataset_balance()
    train_gen, val_gen = get_data_generators()
    num_classes = len(CLASS_NAMES)
    
    print(f"\n--- 1. VERIFY CLASS LABEL CONSISTENCY ---")
    print(f"Class Indices: {train_gen.class_indices}")

    model, base_model = build_model(num_classes)

    # 2. IMPROVE MODEL TRAINING - Callbacks
    early_stop = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy')
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy')

    print("\n--- Phase 1: Training Head ---")
    history1 = model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen,
        callbacks=[early_stop, checkpoint]
    )

    # 3. ENABLE FINE-TUNING - Phase 2: Unfreeze
    print("\n--- Phase 2: Fine-Tuning ---")
    base_model.trainable = True
    # Freeze only early layers (first 100)
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    history2 = model.fit(
        train_gen,
        epochs=EPOCHS, # 25 epochs total for fine-tuning
        validation_data=val_gen,
        callbacks=[early_stop, checkpoint]
    )

    # 2. PRINT ACCURACIES
    train_acc = history2.history['accuracy'][-1]
    val_acc = history2.history['val_accuracy'][-1]
    print(f"\nFinal Training Accuracy: {train_acc:.4f}")
    print(f"Final Validation Accuracy: {val_acc:.4f}")

    # 7. EVALUATION
    plot_confusion_matrix(model, val_gen)
    
    print(f"\n--- 8. SAVE BEST MODEL ---")
    print(f"Best model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_pipeline()
