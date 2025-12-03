# =====================================================
# 🍎 FRUIT CLASSIFIER + 🥗 DIET RECOMMENDER (Streamlit)
# Using trained MobileNetV2 model + Nutrient CSV
# =====================================================

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import itertools
import random
from PIL import Image
import os
from difflib import get_close_matches

# -------------------------------
# 🌟 App Configuration
# -------------------------------
st.set_page_config(
    page_title="Fruit Classifier & Diet Recommender",
    layout="wide",
    page_icon="🍎",
)

# -------------------------------
# 💅 Page Styling (Dark Theme)
# -------------------------------
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(180deg, #1a001f 0%, #000000 100%);
            color: white;
        }
        h1, h2, h3, h4 {
            color: #a64dff;
            text-shadow: 0px 0px 8px #6600cc;
        }
        .stButton>button {
            background-color: #6a0dad;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 10px 20px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #9932cc;
            transform: scale(1.05);
        }
        .stFileUploader {
            border: 2px dashed #6a0dad !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🍎 Fruit Classifier & 🥗 Diet Recommender")
st.markdown("### Upload a fruit image to identify it and get detailed nutritional insights!")

# -------------------------------
# 📁 Paths - update these if needed
# -------------------------------
MODEL_PATH = r"C:\Users\Dell\OneDrive\Desktop\Projects\ML\results\cnn_model.h5"
TRAIN_DIR = r"C:\Users\Dell\OneDrive\Desktop\Projects\ML\Fruit360\fruits-360_100x100\fruits-360\Training"
CSV_PATH = r"C:\Users\Dell\OneDrive\Desktop\Projects\ML\fruits_nutrients_values.csv"

# -------------------------------
# 🧠 Load Model, Labels, and Data (cached)
# -------------------------------
@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")
    model = tf.keras.models.load_model(path)
    return model

@st.cache_resource
def load_labels(train_dir=TRAIN_DIR):
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found at: {train_dir}")
    labels = sorted([c for c in os.listdir(train_dir) if not c.startswith('.')])
    return labels

@st.cache_data
def load_nutrients(path=CSV_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at: {path}")
    df = pd.read_csv(path)
    return df

# load resources with helpful UI feedback
try:
    with st.spinner("Loading model..."):
        model = load_model()
    with st.spinner("Loading class labels..."):
        class_labels = load_labels()
    with st.spinner("Loading nutrient data..."):
        nutrient_df = load_nutrients()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# verify 'Fruit' column
if "Fruit" not in nutrient_df.columns:
    st.error("❌ 'Fruit' column not found in the dataset.")
    st.stop()

# -------------------------------
# Helper: preprocess image
# -------------------------------
def preprocess_image(pil_img, target_size=(100, 100)):
    img_resized = pil_img.resize(target_size)
    arr = np.array(img_resized).astype("float32") / 255.0
    if arr.ndim == 2:  # grayscale -> RGB
        arr = np.stack((arr,)*3, axis=-1)
    return np.expand_dims(arr, axis=0)

# -------------------------------
# Helper: fuzzy match predicted fruit to nutrient CSV
# -------------------------------
def get_fruit_nutrient_info(predicted_name, nutrient_df):
    df_names = nutrient_df["Fruit"].str.lower().str.strip()
    predicted_clean = predicted_name.lower().strip()
    matches = get_close_matches(predicted_clean, df_names, n=1, cutoff=0.6)
    if matches:
        matched_index = df_names[df_names == matches[0]].index[0]
        fruit_info = nutrient_df.iloc[matched_index].drop(labels=["Fruit"], errors="ignore").to_frame(name=predicted_name)
        return fruit_info
    else:
        return None

# -------------------------------
# 📤 Upload Image Section
# -------------------------------
st.header("📸 Upload Fruit Image")
uploaded_file = st.file_uploader("Upload a fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        img = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Could not open image: {e}")
        img = None

    if img is not None:
        st.image(img, caption="Uploaded Image", width=400)

        # preprocess and predict
        try:
            img_array = preprocess_image(img, target_size=(100, 100))
            with st.spinner("Predicting..."):
                prediction = model.predict(img_array)

            predicted_class_index = int(np.argmax(prediction, axis=1)[0])
            confidence = float(np.max(prediction))
            fruit_name_model = class_labels[predicted_class_index]

            st.success(f"✅ Predicted Fruit: {fruit_name_model} with {confidence*100:.2f}% confidence")

            # get nutrient info
            fruit_info = get_fruit_nutrient_info(fruit_name_model, nutrient_df)
            if fruit_info is not None:
                st.subheader("🍽 Nutritional Information per 100g")
                st.dataframe(fruit_info.style.format("{:.2f}"))
            else:
                st.warning("Nutritional info not found for this fruit.")

        except Exception as e:
            st.error(f"Prediction error: {e}")

# -------------------------------
# 🥗 Diet Recommendation Section
# -------------------------------
st.markdown("---")
st.header("🥗 Personalized Diet Planner")

# nutrient columns (exclude Fruit)
cols = [c for c in nutrient_df.columns if c.lower() != "fruit"]
if not cols:
    st.error("No nutrient columns found in CSV (only 'Fruit' present).")
    st.stop()

st.markdown("### Select up to 3 nutrients and specify your target values")
nutrient_inputs = []
for i in range(1, 4):
    col1, col2 = st.columns([3, 2])
    with col1:
        nutrient = st.selectbox(
            f"Select nutrient {i}:",
            ["None"] + cols,
            key=f"nutrient_{i}",
            index=0
        )
    with col2:
        value = 0.0
        if nutrient != "None":
            value = st.number_input(
                f"Target for {nutrient}:",
                min_value=0.0,
                step=1.0,
                key=f"value_{i}"
            )
    if nutrient != "None":
        nutrient_inputs.append((nutrient, float(value)))

# -------------------------------
# 🧮 Generate Recommendations
# -------------------------------
if st.button("🔍 Recommend Fruit Sets"):
    df = nutrient_df.copy()
    if not nutrient_inputs:
        st.warning("⚠ Please select at least one nutrient.")
    else:
        try:
            fruits = df["Fruit"].tolist()
            combo_size = 3 if len(fruits) >= 3 else len(fruits)
            combos = list(itertools.combinations(fruits, combo_size))
            results = []

            # limit samples for performance
            sample_count = min(1500, len(combos))
            sampled = random.sample(combos, sample_count) if len(combos) > sample_count else combos

            for combo in sampled:
                subset = df[df["Fruit"].isin(combo)]
                diff_sum = 0.0
                for nutrient, target in nutrient_inputs:
                    if nutrient in df.columns:
                        total_value = subset[nutrient].astype(float).sum()
                        diff_sum += abs(total_value - target)
                    else:
                        diff_sum += abs(0.0 - target)
                results.append((combo, diff_sum))

            results.sort(key=lambda x: x[1])
            best_matches = results[:3]

            if best_matches:
                st.success(f"✅ Found {len(best_matches)} matching fruit sets!")
                st.write("### 🍇 Suggested Fruit Sets:")
                for i, (combo, score) in enumerate(best_matches):
                    st.write(f"Set {i+1}: {', '.join(combo)}  — (Total deviation: {round(score, 2)})")
                    subset = df[df["Fruit"].isin(combo)].set_index("Fruit")
                    nutrient_names = [n for n, _ in nutrient_inputs if n in df.columns]
                    if nutrient_names:
                        total_vals = subset[nutrient_names].astype(float).sum()
                        st.dataframe(total_vals.to_frame(name="Combined Value"))
                    else:
                        st.info("No valid nutrient columns selected.")
            else:
                st.error("No matching sets found. Try relaxing your target values.")
        except Exception as e:
            st.error(f"Error during recommendation: {e}")

# -------------------------------
# 🏁 Footer
# -------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#aaa;'>© 2025 Fruit Classifier & Diet Recommender | Built with ❤ using Streamlit</p>",
    unsafe_allow_html=True
)