import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
import os

# Load the model
model = load_model("food_model.h5")
class_names = sorted(os.listdir("data/train"))

# Load nutrition data
nutrition_df = pd.read_csv("food_nutrition.csv")  # Ensure correct path

# Sample fun facts (you can expand this)
tips = {
    "Bread": "🥖 Great source of carbs, but try whole grain for more fiber.",
    "Dessert": "🍰 Limit sugary desserts to occasional treats to avoid sugar spikes.",
    "Vegetable-Fruit": "🥦 Eat a variety of colorful veggies and fruits every day.",
    "Soup": "🍜 Light soups can be a good low-calorie meal option.",
    "Egg": "🥚 High in protein and nutrients — great for breakfast!",
    "Meat": "🍗 Choose lean meats to cut down saturated fats.",
    "Rice": "🍚 Pair with fiber-rich foods to slow digestion.",
    "Seafood": "🐟 Rich in omega-3 fats — good for your heart.",
    "Fried food": "🍟 Best enjoyed in moderation due to high fat content.",
    "Dairy product": "🧀 Good source of calcium, but watch saturated fats.",
    "Noodles-Pasta": "🍝 Opt for whole wheat pasta for better fiber."
}

# UI Setup
st.set_page_config(page_title="Food Recognition & Calorie Detector", layout="centered")
st.title("🍽️ Food Recognition & Calorie Detector")
st.markdown("Upload a food image and get its predicted class along with calories, protein, vitamin C, and extra health insights.")

# File upload
file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
if file:
    image = Image.open(file).resize((224, 224))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    prediction = model.predict(img_array)[0]
    label = class_names[np.argmax(prediction)]

    st.subheader(f"🍴 **Predicted Food:** :green[{label}]")

    info = nutrition_df[nutrition_df["food"].str.lower() == label.lower()]
    if not info.empty:
        # Basic nutrition
        calories = info["calories"].values[0]
        protein = info["protein"].values[0]
        vitamin_c = info["vitamin_c"].values[0]

        # Optional fields if exist
        fat = info.get("fat", pd.Series([None])).values[0]
        carbs = info.get("carbs", pd.Series([None])).values[0]
        fiber = info.get("fiber", pd.Series([None])).values[0]
        sugar = info.get("sugar", pd.Series([None])).values[0]

        st.markdown(f"**Calories:** {calories:.0f} kcal")
        st.markdown(f"**Protein:** {protein:.2f} g")
        st.markdown(f"**Vitamin C:** {vitamin_c:.2f} mg {'🟢' if vitamin_c >= 15 else '🟡' if vitamin_c >= 5 else '🔴'}")

        if fat: st.markdown(f"**Fat:** {fat:.1f} g")
        if carbs: st.markdown(f"**Carbs:** {carbs:.1f} g")
        if fiber: st.markdown(f"**Fiber:** {fiber:.1f} g")
        if sugar: st.markdown(f"**Sugar:** {sugar:.1f} g")

        # Verdict
        verdict = "💪 Healthy Choice" if calories < 200 and sugar < 10 else "⚠️ High Calorie / Sugar"
        st.success(f"**Nutritional Verdict:** {verdict}")

        # Fun Fact
        if label in tips:
            st.info(f"**Tip:** {tips[label]}")

    else:
        st.warning("⚠️ Nutrition info not found in the database.")
