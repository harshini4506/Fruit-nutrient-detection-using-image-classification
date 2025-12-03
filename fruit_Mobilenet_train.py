# ==========================================
# 🍎 FRUITS-360 CLASSIFIER (MobileNetV2 + EarlyStopping)
# ✅ Auto-saves best model, plots & classification report
# ==========================================

# ---- Imports ----
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report

# ---- Paths ----
base_dir = r"C:\Users\Dell\OneDrive\Desktop\Projects\ML\Fruit360\fruits-360_100x100\fruits-360"
train_dir = os.path.join(base_dir, "Training")
test_dir = os.path.join(base_dir, "Test")

# ---- Output Directory ----
output_dir = os.path.join(os.getcwd(), "results_mobilenetv2")
os.makedirs(output_dir, exist_ok=True)

# ---- Image setup ----
img_height, img_width = 100, 100
batch_size = 32
validation_split = 0.2

# ---- Data Generators ----
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=validation_split
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

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

# ---- Load Pretrained Base Model ----
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(img_height, img_width, 3)
)
base_model.trainable = False  # freeze base layers initially

# ---- Build Transfer Learning Model ----
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ---- Compile ----
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ---- Callbacks ----
best_model_path = os.path.join(output_dir, "mobilenetv2_best_model.h5")

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(best_model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
]

# ---- Summary ----
model.summary()

# ---- Phase 1: Train top layers ----
epochs_stage1 = 10
history1 = model.fit(
    train_generator,
    epochs=epochs_stage1,
    validation_data=val_generator,
    callbacks=callbacks
)

# ---- Fine-tune last few layers ----
for layer in base_model.layers[-40:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs_stage2 = 10
history2 = model.fit(
    train_generator,
    epochs=epochs_stage2,
    validation_data=val_generator,
    callbacks=callbacks
)

# ---- Combine histories ----
history = {
    'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
    'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
    'loss': history1.history['loss'] + history2.history['loss'],
    'val_loss': history1.history['val_loss'] + history2.history['val_loss']
}

# ---- Save Final Model ----
final_model_path = os.path.join(output_dir, "mobilenetv2_final.h5")
model.save(final_model_path)
print(f"\n💾 Final model saved successfully at: {final_model_path}")
print(f"🏆 Best model auto-saved at: {best_model_path}\n")

# ---- Plot Accuracy and Loss ----
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('MobileNetV2 Accuracy per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('MobileNetV2 Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
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

cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(cm_plot_path)
print(f"📈 Confusion matrix saved at: {cm_plot_path}")
plt.show()

# ---- Classification Report ----
report = classification_report(true_classes, pred_classes, target_names=class_labels)
print("\n📊 Classification Report:\n")
print(report)

report_path = os.path.join(output_dir, "classification_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"📝 Classification report saved at: {report_path}\n")

print("✅ MobileNetV2 Training completed successfully!")
