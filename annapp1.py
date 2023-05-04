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
import sqlite3

nltk.download('stopwords')

# Load the trained model and preprocessing objects
classifier = load_model('trained_model.h5')
cv = pickle.load(open('count-Vectorizer.pkl','rb'))
sc = pickle.load(open('Standard-Scaler.pkl','rb'))

# Create a connection to the database
conn = sqlite3.connect('reviews.db')
c = conn.cursor()

# Create a table to store the reviews
c.execute('''CREATE TABLE IF NOT EXISTS reviews
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              course_experience TEXT,
              instructor TEXT,
              material TEXT,
              sentiment TEXT)''')
conn.commit()

# Function to perform sentiment analysis
def predict_sentiment(input_review):
    if not input_review:
        return "Unknown"
    input_review = re.sub(pattern='[^a-zA-Z\s]', repl=' ', string=input_review)
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
    st.title('Student sentiment analysis')

    # Check if user is an admin
    is_admin = st.sidebar.checkbox('Admin access')

    if not is_admin:
        # Create a form to collect reviews from multiple users
        with st.form(key='review_form'):
            review1 = st.text_area('How was the course experience?')
            review2 = st.text_area('Tell us about the instructor?')
            review3 = st.text_area('Was the material provided useful?')
            submitted = st.form_submit_button('Submit')

            # Store the reviews in the database
            if submitted:
                sentiment1 = predict_sentiment(review1)
                sentiment2 = predict_sentiment(review2)
                sentiment3 = predict_sentiment(review3)

                c.execute("INSERT INTO reviews (course_experience, instructor, material, sentiment) VALUES (?, ?, ?, ?)",
                          (review1, review2, review3, sentiment1))
                conn.commit()
                st.success('Thank you for submitting your reviews.')

    else:
        # Get all the reviews from the database
        reviews_df = pd.read_sql_query("SELECT * FROM reviews", conn)

        # Show individual review analytics
        st.header('Individual Review Analytics')
        for column in reviews_df.columns[:-1]:
            st.subheader(column)
            df_counts = reviews_df[column].apply(predict_sentiment).value_counts()
            st.bar_chart(df_counts)

        # Show overall analytics
        st.header('Overall Analytics')
        df_counts = reviews_df['sentiment'].value_counts()
        st.bar_chart(df_counts)

if __name__ == '__main__':
    main()

