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
c.execute(DROP reviews)
# Create a table to store the reviews
c.execute('''CREATE TABLE IF NOT EXISTS reviews
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              course_experience TEXT,
              sentiment1 TEXT,
              instructor TEXT,
              sentiment2 TEXT,
              material TEXT,
              sentiment3 TEXT)''')

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
              c.execute("INSERT INTO reviews (course_experience, sentiment1, instructor, sentiment2, material, sentiment3) VALUES (?, ?, ?, ?, ?, ?)", (review1, sentiment1, review2, sentiment2, review3, sentiment3))

              c.commit()



    else:
        # Get all the reviews from the database
        reviews_df = pd.read_sql_query("SELECT * FROM reviews", conn)

        # Check if there are any reviews to display
        if len(reviews_df) == 0:
            st.warning('No reviews to display.')
        else:
            # Show overall analytics
            st.header('Overall Analytics')
            df_counts = reviews_df['sentiment'].value_counts()
            st.bar_chart(df_counts)

            # Show sentiment-wise analytics
            st.header('Sentiment-wise Analytics')
            df_counts1 = reviews_df[reviews_df['sentiment']=='positive']['sentiment'].value_counts()
            df_counts2 = reviews_df[reviews_df['sentiment']=='negative']['sentiment'].value_counts()
            st.bar_chart(pd.concat([df_counts1, df_counts2], axis=0))

            # Show reviews table
            st.header('Reviews Table')
            st.dataframe(reviews_df)

            # Allow admin to delete all reviews
            if st.button('Delete all reviews'):
              c.execute("DELETE FROM reviews")
              conn.commit()
              st.success('All reviews have been deleted.')


if __name__ == '__main__':
    main()









