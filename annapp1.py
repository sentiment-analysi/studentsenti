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

# Function to show analytics for all reviews
def show_analytics(reviews):
    results = {}
    for i, review in enumerate(reviews):
        result = predict_sentiment(review)
        results[f'Review {i+1}'] = result
    df = pd.DataFrame({'Reviews': list(results.keys()), 'Sentiment': list(results.values())})
    df_counts = df['Sentiment'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(df_counts.index, df_counts.values, color=['blue', 'yellow'])
    ax.set_title('Sentiment Analysis Results')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    st.pyplot(fig)



def main():
    st.title('Student sentiment analysis')

    # Create a form to collect reviews from multiple users
    with st.form(key='review_form'):
        review1 = st.text_area('How was the course experience?')
        review2 = st.text_area('Tell us about the instructor?')
        review3 = st.text_area('Was the material provided useful?')
        submitted = st.form_submit_button('Submit')

        # Store the reviews in a Pandas DataFrame or a database
        if submitted:
            reviews_df = pd.DataFrame({
                'Course experience': [review1],
                'Instructor': [review2],
                'Material': [review3]
            })
            st.success('Thank you for submitting your reviews!')

    # Only show the analytics if there are reviews to analyze
    if 'reviews_df' in locals():
        # Perform sentiment analysis on the reviews and show the results
        results = {}
        for col in reviews_df.columns:
            results[col] = predict_sentiment(reviews_df[col].iloc[0])
            st.success(f"{col}: {results[col]}")

        # Show analytics using a bar chart
        df = pd.DataFrame({'Reviews': list(results.keys()), 'Sentiment': list(results.values())})
        df_counts = df['Sentiment'].value_counts()
        fig, ax = plt.subplots()
        ax.bar(df_counts.index, df_counts.values, color=['blue', 'yellow'])
        ax.set_title('Sentiment Analysis Results')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        st.pyplot(fig)

if __name__=='__main__':
    main()
