# ==========================================
# 🍏 FRUIT360 - IMAGE UPLOAD TEST SCRIPT
# ✅ Choose image → Get fruit name instantly
# ==========================================

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tkinter import Tk, filedialog

# ---- Paths ----
model_path = r"results\cnn_model.h5"
train_dir = r"C:\Users\Dell\OneDrive\Desktop\Projects\ML\Fruit360\fruits-360_100x100\fruits-360\Training"

# ---- Image setup ----
img_height, img_width = 100, 100

# ---- Load trained model ----
print("📦 Loading model...")
model = load_model(model_path)
print("✅ Model loaded successfully!")

# ---- Load class labels ----
class_labels = sorted(os.listdir(train_dir))
class_labels = [c for c in class_labels if not c.startswith('.')]
print(f"📚 Loaded {len(class_labels)} fruit classes.")

# ---- Function: Predict fruit ----
def predict_fruit(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    predicted_class = class_labels[predicted_class_index]

    # Show image with prediction
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence*100:.2f}%")
    plt.axis('off')
    plt.show()

    print(f"\n🖼 Image: {os.path.basename(img_path)}")
    print(f"🍎 Predicted Fruit: {predicted_class}")
    print(f"📊 Confidence: {confidence*100:.2f}%\n")

# ---- Main ----
if __name__ == "__main__":
    print("\n🔍 FRUIT DETECTION TEST (Upload Mode)")

    # Hide main Tkinter window
    root = Tk()
    root.withdraw()

    # Open file dialog
    img_path = filedialog.askopenfilename(
        title="Select a Fruit Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.jfif")]
    )

    if img_path:
        print(f"📂 Selected image: {img_path}")
        predict_fruit(img_path)
    else:
        print("⚠️ No image selected. Exiting.")
