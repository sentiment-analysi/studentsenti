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

# Main function to run the app Main function to run the app
def main():
    st.title('Student sentiment analysis')

    # Create empty lists to store user reviews
    reviews1 = []
    reviews2 = []
    reviews3 = []

    # Get the number of users and reviews per user
    num_users = st.number_input('Number of users:', min_value=1, max_value=10, step=1)
    num_reviews = st.number_input('Number of reviews per user:', min_value=1, max_value=10, step=1)

    # Get the user inputs
    for i in range(num_users):
        st.subheader(f'Reviews for User {i+1}')
        for j in range(num_reviews):
            review1 = st.text_input(f'Review {j+1} for course experience:')
            review2 = st.text_input(f'Review {j+1} for instructor:')
            review3 = st.text_input(f'Review {j+1} for useful material:')
            reviews1.append(review1)
            reviews2.append(review2)
            reviews3.append(review3)

    # Perform sentiment analysis for all reviews
    results1 = [predict_sentiment(review) for review in reviews1]
    results2 = [predict_sentiment(review) for review in reviews2]
    results3 = [predict_sentiment(review) for review in reviews3]

    # Show the overall sentiment analysis results
    st.subheader('Sentiment Analysis Results')
    st.write('Course experience:', dict(Counter(results1)))
    st.write('Instructor:', dict(Counter(results2)))
    st.write('Useful material:', dict(Counter(results3)))

    # Show analytics using a bar chart
    results = {'Course experience': results1, 'Instructor': results2, 'Useful material': results3}
    df = pd.DataFrame({'Reviews': list(results.keys()), 'Sentiment': list(map(lambda x: dict(Counter(x)), results.values()))})
    df_counts = df.explode('Sentiment').groupby(['Reviews', 'Sentiment']).size().reset_index(name='Count')
    fig, ax = plt.subplots()
    ax.bar(df_counts['Sentiment'], df_counts['Count'], color=['blue', 'yellow'])
    ax.set_title('Sentiment Analysis Results')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    ax.legend(df_counts['Reviews'])
    st.pyplot(fig)


# Run the app
if __name__=='__main__':
    main()
