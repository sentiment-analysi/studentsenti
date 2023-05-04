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
    st.title('Student sentiment analysis')

    # Get the number of reviews to collect from the user
    n_reviews = st.number_input('How many reviews would you like to collect?', min_value=1, max_value=10, step=1)

    # Collect the reviews from the user
    reviews = []
    for j in range(n_reviews):
        review = st.text_input(f'Enter review {j+1}:', key=f'review_{j+1}')
        reviews.append(review)

    # Perform sentiment analysis and show the results
    if st.button('Predict'):
        results = [predict_sentiment(review) for review in reviews]
        st.write('Sentiment Analysis Results:')
        for i, result in enumerate(results):
            st.write(f'Review {i+1}: {result}')

        # Show analytics using a bar chart
        df = pd.DataFrame({'Reviews': [f'Review {i+1}' for i in range(n_reviews)], 'Sentiment': results})
        df_counts = df['Sentiment'].value_counts()
        fig, ax = plt.subplots()
        ax.bar(df_counts.index, df_counts.values, color=['blue', 'yellow'])
        ax.set_title('Sentiment Analysis Results')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        st.pyplot(fig)



# Run the app
if __name__=='__main__':
    main()
