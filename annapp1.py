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
conn = sqlite3.connect('reviews1.db')
c = conn.cursor()

# Create a table to store the reviews
c.execute('''CREATE TABLE IF NOT EXISTS reviews1
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
              c.execute("INSERT INTO reviews1 (course_experience, sentiment1, instructor, sentiment2, material, sentiment3) VALUES (?, ?, ?, ?, ?, ?)", (review1, sentiment1, review2, sentiment2, review3, sentiment3))

              conn.commit()
              st.success('Thank you for submitting your reviews.')


    else:
        # Get all the reviews from the database
        reviews_df = pd.read_sql_query("SELECT * FROM reviews1", conn)
        num_pos_reviewsfor1 = len(reviews_df[reviews_df['sentiment1'] == 'Positive review'])
        num_pos_reviewsfor2 = len(reviews_df[reviews_df['sentiment1'] == 'Positive review'])
        num_pos_reviewsfor3 = len(reviews_df[reviews_df['sentiment1'] == 'Positive review'])
        num_neg_reviewsfor1 = len(reviews_df[reviews_df['sentiment1'] == 'Negative review'])
        num_neg_reviewsfor2 = len(reviews_df[reviews_df['sentiment1'] == 'Negative review'])
        num_neg_reviewsfor3 = len(reviews_df[reviews_df['sentiment1'] == 'Negative review'])
        totalnum_pos_reviews = len(reviews_df[reviews_df['sentiment1'] == 'Positive review']) + \
                  len(reviews_df[reviews_df['sentiment2'] == 'Positive review']) + \
                  len(reviews_df[reviews_df['sentiment3'] == 'Positive review'])
        totalnum_neg_reviews = len(reviews_df[reviews_df['sentiment1'] == 'Negative review']) + \
                  len(reviews_df[reviews_df['sentiment2'] == 'Negative review']) + \
                  len(reviews_df[reviews_df['sentiment3'] == 'Negative review'])


        # Check if there are any reviews to display
        if len(reviews_df) == 0:
            st.warning('No reviews to display.')
        else:
            # Show overall analytics
           # Show sentiment-wise analytics
          st.header('Reviews Table')
          st.dataframe(reviews_df)
          st.write(f"Total Number of positive reviews for question 1: {num_pos_reviewsfor1}")
          st.write(f"Total Number of positive reviews for question 2: {num_pos_reviewsfor2}")
          st.write(f"Total Number of positive reviews for question 3: {num_pos_reviewsfor3}")
          
          st.write(f"Total Number of negative reviews for question 1: {num_neg_reviewsfor1}")
          st.write(f"Total Number of negative reviews for question 2: {num_neg_reviewsfor2}")
          st.write(f"Total Number of negative reviews for question 3: {num_neg_reviewsfor3}")
      
          st.write(f"Total Number of positive reviews: {totalnum_pos_reviews}")
          st.write(f"Total Number of negative reviews: {totalnum_neg_reviews}")


            # Allow admin to delete all reviews
          if st.button('Delete all reviews'):
             c.execute("DELETE FROM reviews1")
             conn.commit()
             st.success('All reviews have been deleted.')
          st.header('Sentiment-wise Analytics')
          df_counts1 = reviews_df[reviews_df['sentiment1']=='Positive review']['sentiment1'].value_counts()
          df_counts2 = reviews_df[reviews_df['sentiment1']=='Negative review']['sentiment1'].value_counts()
          df_counts3 = reviews_df[reviews_df['sentiment2']=='Positive review']['sentiment2'].value_counts()
          df_counts4 = reviews_df[reviews_df['sentiment2']=='Negative review']['sentiment2'].value_counts()
          df_counts5 = reviews_df[reviews_df['sentiment3']=='Positive review']['sentiment3'].value_counts()
          df_counts6 = reviews_df[reviews_df['sentiment3']=='Negative review']['sentiment3'].value_counts()
          st.bar_chart(pd.concat([df_counts1, df_counts2, df_counts3, df_counts4, df_counts5, df_counts6], axis=0))



if __name__ == '__main__':
    main()












