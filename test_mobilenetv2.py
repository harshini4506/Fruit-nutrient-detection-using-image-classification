# =====================================================
# 🍎 Standalone Fruit Classifier Test Script (100x100)
# Automatically picks first image in a folder
# =====================================================

import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import os

# -------------------------------
# 🧠 Paths - update these
# -------------------------------
MODEL_PATH = r"C:\Users\Dell\OneDrive\Desktop\Projects\ML\results_mobilenetv2\mobilenetv2_final.h5"
CSV_PATH = r"C:\Users\Dell\OneDrive\Desktop\Projects\ML\fruits_nutrients_values.csv"
TEST_IMAGE_FOLDER = r"C:\Users\Dell\OneDrive\Desktop\Projects\ML\Fruit360\fruits-360_100x100\fruits-360\Training\Apple 5"  # folder containing images

# -------------------------------
# 🔹 Load Model
# -------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# -------------------------------
# 🔹 Load Nutrient CSV to get fruit labels
# -------------------------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found at: {CSV_PATH}")
nutrient_df = pd.read_csv(CSV_PATH)

if "Fruit" not in nutrient_df.columns:
    raise ValueError("❌ 'Fruit' column not found in CSV.")

fruit_labels = sorted(nutrient_df["Fruit"].unique())
print(f"Fruit labels: {fruit_labels}")

# -------------------------------
# 🔹 Pick first image in folder
# -------------------------------
if not os.path.exists(TEST_IMAGE_FOLDER):
    raise FileNotFoundError(f"Folder not found: {TEST_IMAGE_FOLDER}")

images = [f for f in os.listdir(TEST_IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not images:
    raise FileNotFoundError(f"No image files found in folder: {TEST_IMAGE_FOLDER}")

TEST_IMAGE_PATH = os.path.join(TEST_IMAGE_FOLDER, images[0])
print(f"Using test image: {TEST_IMAGE_PATH}")

# -------------------------------
# 🔹 Preprocess Function (100x100)
# -------------------------------
def preprocess_image(pil_img, target_size=(100, 100)):
    img_resized = pil_img.resize(target_size)
    arr = np.array(img_resized).astype("float32") / 255.0  # normalize to [0,1]
    if arr.ndim == 2:  # grayscale -> RGB
        arr = np.stack((arr,) * 3, axis=-1)
    return np.expand_dims(arr, axis=0)

# -------------------------------
# 🔹 Load and preprocess test image
# -------------------------------
test_img = Image.open(TEST_IMAGE_PATH).convert("RGB")
test_array = preprocess_image(test_img, target_size=(100, 100))

# -------------------------------
# 🔹 Predict
# -------------------------------
pred = model.predict(test_array)
predicted_class = int(np.argmax(pred))
confidence = float(np.max(pred))
fruit_name = fruit_labels[predicted_class]

# -------------------------------
# 🔹 Show results
# -------------------------------
print(f"Predicted Fruit: {fruit_name}")
print(f"Confidence: {confidence*100:.2f}%")
