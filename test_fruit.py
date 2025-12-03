# ==========================================
# 🍏 FRUIT360 - SINGLE IMAGE PREDICTION
# ✅ Upload an image → Get fruit name + confidence
# ==========================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tkinter import Tk, filedialog

# ---- Paths ----
model_path = r"results\cnn_model.h5"
train_dir = r"C:\Users\Dell\OneDrive\Desktop\Projects\ML\Fruit360\fruits-360_100x100\fruits-360\Training"

# ---- Image setup ----
img_height, img_width = 100, 100

# ---- Load trained model ----
print("📦 Loading trained model...")
model = load_model(model_path)
print("✅ Model loaded successfully!")

# ---- Load class labels ----
class_labels = sorted([c for c in os.listdir(train_dir) if not c.startswith('.')])
print(f"📚 Loaded {len(class_labels)} fruit classes.\n")

# ---- Function: Predict fruit from image ----
def predict_fruit(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    predicted_class = class_labels[predicted_index]

    # Display image with prediction
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence*100:.2f}%")
    plt.axis('off')
    plt.show()

    # Print results
    print(f"🖼 Image: {os.path.basename(img_path)}")
    print(f"🍎 Predicted Fruit: {predicted_class}")
    print(f"📊 Confidence: {confidence*100:.2f}%")

# ---- Main: Upload image ----
if __name__ == "__main__":
    print("🔍 Select an image of a fruit to predict...\n")

    root = Tk()
    root.withdraw()  # hide main window
    img_path = filedialog.askopenfilename(
        title="Select Fruit Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    root.destroy()

    if img_path:
        print(f"📂 Selected file: {img_path}\n")
        predict_fruit(img_path)
    else:
        print("❌ No image selected. Please try again.")
