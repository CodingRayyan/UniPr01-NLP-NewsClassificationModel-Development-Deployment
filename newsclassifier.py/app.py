import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import streamlit as st
import pickle

with open("newsclassifier_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vec.pkl", "rb") as tfidf_file:
    tfidf = pickle.load(tfidf_file)

st.title("📰 News Headline Classification")
st.write("Categories Available: ['ENTERTAINMENT', 'TRAVEL', 'WELLNESS', 'POLITICS', 'STYLE & BEAUTY']")
st.write("Rayyan, Wajahat, and Sami amazed the crowd with their elegant outfits at the gala. (Predicted Category: STYLE & BEAUTY)")
st.write("Enter a news headline and get its category prediction!")

sentence = st.text_input("Enter a headline:")

if st.button("Predict Category"):
    if sentence:
        
        sentence_vector = tfidf.transform([sentence])

        prediction = model.predict(sentence_vector)[0]

        st.success(f"**Predicted Category:** {prediction}")
    else:
        st.warning("⚠️ Please enter a headline before clicking Predict!")
