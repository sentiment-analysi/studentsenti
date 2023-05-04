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
    st.set_page_config(page_title='Student sentiment analysis', page_icon=':books:', layout='wide')

    # Get the number of users and reviews per user
    num_users = st.sidebar.number_input('Enter the number of users:', min_value=1, step=1)
    reviews_per_user = st.sidebar.number_input('Enter the number of reviews per user:', min_value=1, step=1)

    # Initialize an empty dataframe to store the reviews
    reviews_df = pd.DataFrame(columns=['User', 'Review', 'Sentiment'])

    # Allow each user to submit their reviews
    for i in range(num_users):
        st.subheader(f'User {i+1}')
        for j in range(reviews_per_user):
            review = st.text_input(f'Enter review {j+1}:')
            if st.button('Submit review'):
                sentiment = predict_sentiment(review)
                reviews_df = reviews_df.append({'User': f'User {i+1}', 'Review': review, 'Sentiment': sentiment}, ignore_index=True)
                st.success('Review submitted successfully!')

    # Show the admin panel
    st.sidebar.markdown('## Admin Panel')
    st.sidebar.write(reviews_df)

    # Show the overall sentiment analysis results
    st.markdown('## Sentiment analysis results')
    sentiment_counts = reviews_df['Sentiment'].value_counts()
    st.write('Sentiment distribution:', sentiment_counts)

    # Show the sentiment analysis results as a bar chart
    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.index, sentiment_counts.values, color=['blue', 'yellow'])
    ax.set_title('Sentiment Analysis Results')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    st.pyplot(fig)



# Run the app
if __name__=='__main__':
    main()
