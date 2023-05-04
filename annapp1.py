import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
import base64

nltk.download('stopwords')

# Load the trained model and preprocessing objects
classifier = load_model('trained_model.h5')
cv = pickle.load(open('count-Vectorizer.pkl','rb'))
sc = pickle.load(open('Standard-Scaler.pkl','rb'))

# Function to perform sentiment analysis
def predict_sentiment(input_review):
    input_review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=input_review)
    input_review = input_review.lower()
    input_review_words = input_review.split()
    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')
    input_review_words = [word for word in input_review_words if not word in stop_words]
    ps = PorterStemmer()
    input_review = [ps.stem(word) for word in input_review_words]
    input_review = ' '.join(input_review)
    input_X = cv.transform([input_review]).toarray()
    input_X = sc.transform(input_X)
    pred = classifier.predict(input_X)
    pred = (pred > 0.5)
    if pred[0][0]:
        return "Positive review"
    else:
        return "Negative review"

def main():
    st.set_page_config(page_title='Student Sentiment Analysis', page_icon=':books:')

    # Get the number of reviews to collect from the user
    num_reviews = st.number_input('How many reviews would you like to collect?', min_value=1, max_value=10)

    # Collect the reviews from the user
    reviews = []
    for i in range(num_reviews):
        review = st.text_input(f'Enter review {i+1}:')
        reviews.append(review)

    # If the user has submitted reviews, show the results and analytics
    if st.button('Submit'):
        # Perform sentiment analysis and show the results
        results = []
        for review in reviews:
            result = predict_sentiment(review)
            results.append(result)
            st.success(f"Review: {review}")
            st.success(f"Sentiment: {result}")

        # Show analytics using a bar chart
        df = pd.DataFrame({'Reviews': [f'Review {i+1}' for i in range(num_reviews)], 'Sentiment': results})
        df_counts = df['Sentiment'].value_counts()
        fig, ax = plt.subplots()
        ax.bar(df_counts.index, df_counts.values, color=['blue', 'yellow'])
        ax.set_title('Sentiment Analysis Results')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        st.pyplot(fig)

    # Define a dictionary of user credentials
    user_credentials = {
        "admin": "password123",
        "user": "password456",
    }

    # Render the login form
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Log in"):
        if username in user_credentials and password == user_credentials[username]:
            session_state = st.session_state.get(authenticated=True, username=username)
            st.success("Logged in!")
        else:
            st.error("Incorrect username or password")

    # If the user is authenticated as the admin, show the reviews and analytics
    if 'session_state' in locals() and session_state.username == "admin":
        st.write("Reviews and Analytics")
        # Show a table of the reviews
        df_reviews = pd.DataFrame({'Reviews': reviews})
        st.write(df_reviews)

        # Show a bar chart of the sentiment analysis results
        fig, ax = plt.subplots()
        ax.bar(df_counts.index, df_counts.values, color=['blue', 'yellow'])
        ax.set_title('Sentiment Analysis Results')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        st.pyplot(fig)



# Run the app
if __name__=='__main__':
    main()
