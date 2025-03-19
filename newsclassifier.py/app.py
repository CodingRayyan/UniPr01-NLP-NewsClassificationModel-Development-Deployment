import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import streamlit as st
import pickle

# Load the model and TF-IDF vectorizer
with open("newsclassifier_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vec.pkl", "rb") as tfidf_file:
    tfidf = pickle.load(tfidf_file)

st.title("📰 News Headline Classification\nCategories Available: ['ENTERTAINMENT', 'TRAVEL', 'WELLNESS', 'POLITICS', 'STYLE & BEAUTY']")
st.write("Example: Rayyan, Wajahat and Sami wore their best suits at the event and were looking great! (Predicted Category: STYLE & BEAUTY)")
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
