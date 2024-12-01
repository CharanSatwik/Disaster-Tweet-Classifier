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

# Predict function
def predict_single_tweet(tweet):
    # Clean the tweet text
    cleaned_tweet = clean_text(tweet)
    
    # Transform the tweet using the loaded vectorizer
    tweet_vector = vectorizer.transform([cleaned_tweet])
    
    # Predict using the loaded model
    prediction = model.predict(tweet_vector)[0]
    
    # Return the prediction
    return "Disaster Tweet" if prediction == 1 else "Not a Disaster Tweet"

# Example tweet for prediction
tweet = "I bought a new phone"
result = predict_single_tweet(tweet)
print(f"The tweet is classified as: {result}")
