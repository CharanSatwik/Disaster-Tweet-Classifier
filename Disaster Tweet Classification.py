import streamlit as st
import joblib
import re

# Load the trained model and vectorizer
model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'www\S+', '', text)  # Remove URLs starting with www
    text = re.sub(r'\@\w+', '', text)   # Remove mentions
    text = re.sub(r'\#\w+', '', text)   # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = re.sub(r'\d+', '', text)     # Remove numbers
    text = text.strip()
    return text

# Prediction function
def predict_single_tweet(tweet):
    # Clean the tweet text
    cleaned_tweet = clean_text(tweet)
    
    # Transform the tweet using the loaded vectorizer
    tweet_vector = vectorizer.transform([cleaned_tweet])
    
    # Predict using the loaded model
    prediction = model.predict(tweet_vector)[0]
    
    # Return the prediction
    return "Disaster Tweet" if prediction == 1 else "Not a Disaster Tweet"

# Streamlit app
st.title("Disaster Tweet Classifier")
st.write("This app predicts whether a given tweet is related to a disaster or not.")

# Text input
tweet_input = st.text_area("Enter a tweet to classify:")

# Predict button
if st.button("Predict"):
    if tweet_input.strip():  # Ensure input is not empty
        prediction = predict_single_tweet(tweet_input)
        st.write(f"The tweet is classified as: **{prediction}**")
    else:
        st.write("Please enter a valid tweet.")
