# ==========================================
# 🍎 FRUIT CLASSIFICATION + NUTRIENT DETECTION
# Using MobileNetV2 + CSV Mapping
# ==========================================

# STEP 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# STEP 2: Imports
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from tensorflow.keras.preprocessing import image

# ==========================================
# STEP 3: Load CSV (merged dataset)
# ==========================================
csv_path = '/content/drive/MyDrive/Fruit Dataset/merged_dataset.csv'
df = pd.read_csv(csv_path, low_memory=False)
print("✅ CSV Loaded:", df.shape)

# ==========================================
# STEP 4: Prepare Image Paths (recursive search)
# ==========================================
image_folder = '/content/drive/MyDrive/Fruit Dataset/Fruits_Dataset_Train/'

# Search all jpg/jpeg files recursively
all_images = glob.glob(os.path.join(image_folder, '**', '*.jp*'), recursive=True)
image_map = {os.path.basename(p): p for p in all_images}
print(f"🔍 Found {len(image_map)} total images in subfolders")

# Clean filenames and map to full paths
df['FileName'] = df['FileName'].astype(str).str.strip()
df['image_path'] = df['FileName'].map(image_map)

# Keep only rows with valid image paths
df_img = df[df['image_path'].notna() & df['fruit'].notna()].copy()
if len(df_img) == 0:
    raise ValueError("❌ No valid images found! Check folder and CSV mapping.")

print("✅ Valid image matches:", len(df_img))
print("Sample image path:", df_img['image_path'].iloc[0])

# ==========================================
# STEP 5: Encode Fruit Labels
# ==========================================
le = LabelEncoder()
df_img['label_enc'] = le.fit_transform(df_img['fruit'])
num_classes = len(le.classes_)
print("🍎 Classes:", le.classes_)

# ==========================================
# STEP 6: Train-Test Split
# ==========================================
train_df, test_df = train_test_split(
    df_img, test_size=0.2, stratify=df_img['fruit'], random_state=42
)
print("Train samples:", len(train_df), "| Test samples:", len(test_df))

# ==========================================
# STEP 7: Image Generators (Augmentation)
# ==========================================
IMG_SIZE = 224
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    x_col='image_path',
    y_col='fruit',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_gen = test_datagen.flow_from_dataframe(
    test_df,
    x_col='image_path',
    y_col='fruit',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ==========================================
# STEP 8: Build Model (MobileNetV2)
# ==========================================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Freeze base layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ==========================================
# STEP 9: Callbacks
# ==========================================
model_save_path = '/content/drive/MyDrive/Fruit Dataset/fruit_nutrient_mobilenetv2_best.h5'
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
]

# ==========================================
# STEP 10: Train Model
# ==========================================
EPOCHS = 20
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==========================================
# STEP 11: Accuracy & Loss Graphs
# ==========================================
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.legend()
plt.show()

# ==========================================
# STEP 12: Evaluate & Classification Report
# ==========================================
model.load_weights(model_save_path)
loss, acc = model.evaluate(test_gen)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")

# Predictions
y_pred = model.predict(test_gen)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_gen.classes

print("\n📊 Classification Report:\n", classification_report(y_true, y_pred_classes, target_names=list(test_gen.class_indices.keys())))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# ==========================================
# STEP 13: Save Final Model
# ==========================================
final_model_path = '/content/drive/MyDrive/Fruit Dataset/fruit_nutrient_mobilenetv2_final.h5'
model.save(final_model_path)
print(f"✅ Model saved at: {final_model_path}")

# ==========================================
# STEP 14: Predict Fruit + Show Nutrients
# ==========================================
def predict_fruit(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    class_name = le.classes_[class_idx]
    confidence = np.max(pred)

    print(f"\n🍏 Predicted Fruit: {class_name} ({confidence*100:.2f}%)")

    # Display Nutrients for this fruit
    nutrients = df[df['fruit'].str.lower() == class_name.lower()].iloc[0]
    print("\n🥗 Nutrient Info:")
    nutrient_cols = [c for c in df.columns if c not in ['FileName','fruit','image_path','label_enc']]
    for col in nutrient_cols[:8]:  # show top few nutrients
        print(f"{col}: {nutrients[col]}")
    return class_name

# Example:
# predict_fruit('/content/drive/MyDrive/Fruit Dataset/Fruits_Dataset_Test/apple/1.jpg')
