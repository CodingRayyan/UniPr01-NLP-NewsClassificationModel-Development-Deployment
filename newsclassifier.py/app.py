import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import streamlit as st
import pickle

# Load the model and TF-IDF vectorizer
with open("newsclassifier_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vec.pkl", "rb") as tfidf_file:
    tfidf = pickle.load(tfidf_file)

# ✅ Correct background image code
st.markdown(
    """
    <style>
    body {
        background-image: url("stbg.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📰 News Headline Classification")
st.subheader("Categories Available: ['ENTERTAINMENT', 'TRAVEL', 'WELLNESS', 'POLITICS', 'STYLE & BEAUTY']")
st.write("Rayyan, Wajahat, and Sami amazed the crowd with their elegant outfits at the gala. (Predicted Category: STYLE & BEAUTY)")
st.write("Enter a news headline and get its category prediction!")

# User Input
sentence = st.text_input("Enter a headline:")

if st.button("Predict Category"):
    if sentence:
        # Transform input using TF-IDF
        sentence_vector = tfidf.transform([sentence])

        # Predict category
        prediction = model.predict(sentence_vector)[0]

        st.success(f"**Predicted Category:** {prediction}")
    else:
        st.warning("⚠️ Please enter a headline before clicking Predict!")
