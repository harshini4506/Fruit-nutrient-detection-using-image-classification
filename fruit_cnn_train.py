# ==========================================
# 🍎 FRUITS-360 CNN CLASSIFIER (Local - VS Code)
# ✅ Saves model, plots & classification report
# ==========================================

# ---- Imports ----
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix, classification_report

# ---- Paths ----
base_dir = r"C:\Users\Dell\OneDrive\Desktop\Projects\ML\Fruit360\fruits-360_100x100\fruits-360"
train_dir = os.path.join(base_dir, "Training")
test_dir = os.path.join(base_dir, "Test")

# ---- Output Directory ----
output_dir = os.path.join(os.getcwd(), "results")
os.makedirs(output_dir, exist_ok=True)

# ---- Image setup ----
img_height, img_width = 100, 100
batch_size = 32
validation_split = 0.2  # 20% of training data used for validation

# ---- Data Generators ----
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=validation_split
)

test_datagen = ImageDataGenerator(rescale=1./255)

# ---- Data Flow ----
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=1,
    class_mode=None,
    shuffle=False
)

# ---- Info ----
print(f"\n✅ Number of classes: {train_generator.num_classes}")
print(f"🖼 Training images: {train_generator.samples}")
print(f"🧪 Validation images: {val_generator.samples}")
print(f"🧩 Test images: {test_generator.samples}\n")

# ---- CNN Model ----
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),

    Dense(train_generator.num_classes, activation='softmax')
])

# ---- Compile ----
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ---- Summary ----
model.summary()

# ---- Train ----
epochs = 20
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator
)

# ---- Save Model (in results folder) ----
model_save_path = os.path.join(output_dir, "cnn_model.h5")
model.save(model_save_path)
print(f"\n💾 Model saved successfully at: {model_save_path}\n")

# ---- Plot Accuracy and Loss ----
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()

# Save accuracy/loss plot
acc_loss_plot_path = os.path.join(output_dir, "accuracy_loss_plot.png")
plt.savefig(acc_loss_plot_path)
print(f"📊 Accuracy/Loss plot saved at: {acc_loss_plot_path}")
plt.show()

# ---- Confusion Matrix ----
val_generator.reset()
preds = model.predict(val_generator)
pred_classes = np.argmax(preds, axis=1)
true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

cm = confusion_matrix(true_classes, pred_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix - Validation Data')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# Save confusion matrix
cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(cm_plot_path)
print(f"📈 Confusion matrix saved at: {cm_plot_path}")
plt.show()

# ---- Classification Report ----
report = classification_report(true_classes, pred_classes, target_names=class_labels)
print("\n📊 Classification Report:\n")
print(report)

# Save report to text file
report_path = os.path.join(output_dir, "classification_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"📝 Classification report saved at: {report_path}\n")

print("✅ Training completed successfully!")
