import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import google.generativeai as genai

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fruit & Vegetable Recognition",
    page_icon="🍎",
    layout="centered"
)

# ---------------- GEMINI CONFIG ----------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

# ---------------- LOAD MODELS ----------------
fruit_model = load_model("fruit_model.h5")
veg_model = load_model("vegetable_model.h5")

# ---------------- CLASS LABELS ----------------
fruit_classes = [
    'Apple', 'Banana', 'Grapes', 'Mango', 'Orange',
    'Pineapple', 'Strawberry', 'Watermelon'
]

vegetable_classes = [
    'Beetroot', 'Cabbage', 'Carrot', 'Cauliflower',
    'Potato', 'Tomato', 'Onion', 'Spinach'
]

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- GEMINI NUTRIENT FUNCTION ----------------
def get_nutrients(food_name):
    prompt = f"""
    Give the nutritional content of {food_name}.

    Include:
    - Calories
    - Vitamins
    - Minerals
    - Health benefits

    Answer in bullet points, simple words.
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except:
        return "⚠️ Nutrient data temporarily unavailable."

# ---------------- UI ----------------
st.title("🍎 Fruit & Vegetable Recognition System")
st.write("Upload an image to identify the food and view its nutrients")

uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)

    # Predict using both models
    fruit_pred = fruit_model.predict(img_array)
    veg_pred = veg_model.predict(img_array)

    fruit_conf = np.max(fruit_pred)
    veg_conf = np.max(veg_pred)

    # Decide which model is more confident
    if fruit_conf > veg_conf:
        index = np.argmax(fruit_pred)
        label = fruit_classes[index]
        confidence = fruit_conf
        category = "Fruit"
    else:
        index = np.argmax(veg_pred)
        label = vegetable_classes[index]
        confidence = veg_conf
        category = "Vegetable"

    # ---------------- DISPLAY RESULT ----------------
    st.success(f"### 🧠 Detected: {label} ({category})")
    st.write(f"**Confidence:** {confidence:.2f}")
    st.progress(float(confidence))

    # ---------------- GEMINI OUTPUT ----------------
    with st.spinner("🤖 Fetching nutrient information..."):
        nutrients = get_nutrients(label)

    st.subheader("🥗 Nutritional Information")
    st.write(nutrients)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("CNN Models + Gemini AI | College Mini Project")
