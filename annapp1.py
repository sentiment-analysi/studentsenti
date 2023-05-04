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
    st.set_page_config(page_title='Student Sentiment Analysis', page_icon=':books:', layout='wide')
    st.title('Student sentiment analysis')

    # Login form for admin
    with st.form(key='login_form'):
        username = st.text_input('Username')
        password = st.text_input('Password', type='password')
        login = st.form_submit_button('Login')

    # If login successful, show the review form
    if login:
        if username == 'admin' and password == 'password':
            st.success('Logged in successfully!')
            st.sidebar.markdown('## Analytics')
            st.sidebar.markdown('View sentiment analysis results for all reviews submitted:')
            show_analytics = st.sidebar.checkbox('Show analytics')
            if show_analytics:
                # Load reviews from the database or DataFrame and perform sentiment analysis
                # Display the sentiment analysis results using a bar chart
                pass
            else:
                # Create a form to collect reviews from multiple users
                with st.form(key='review_form'):
                    review1 = st.text_area('How was the course experience?')
                    review2 = st.text_area('Tell us about the instructor?')
                    review3 = st.text_area('Was the material provided useful?')
                    submitted = st.form_submit_button('Submit')

                    # Store the reviews in a database or Pandas DataFrame
                    if submitted:
                        reviews_df = pd.DataFrame({
                            'Course experience': [review1],
                            'Instructor': [review2],
                            'Material': [review3]
                        })
                        st.success('Thank you for submitting your reviews!')
        else:
            st.error('Invalid username or password.')


if __name__=='__main__':
    main()
