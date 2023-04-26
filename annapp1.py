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
import base64
import PyPDF2

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

# Function to create a PDF report
def create_downloadable_report(results):
    # Create a new PDF file
    pdf = PyPDF2.PdfFileWriter()
    # Add a new page to the PDF file
    page = PyPDF2.pdf.PageObject.createBlankPage(None, 72*11, 72*8.5)
    pdf.addPage(page)
    # Write the analysis results to the page
    content = "Sentiment Analysis Results:\n\n"
    for key, value in results.items():
        content += f"{key}: {value}\n"
    page.mergeTextFields(content)
    # Convert the PDF file to bytes
    pdf_bytes = io.BytesIO()
    pdf.write(pdf_bytes)
    pdf_bytes.seek(0)
    # Encode the bytes in base64 format for download
    b64 = base64.b64encode(pdf_bytes.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="sentiment_analysis_report.pdf">Download report</a>'
    return href

# Main function to run the app
def main():
    st.title('Student sentiment analysis')

    # Get the user inputs
    review1 = st.text_input('How was the course experience?')
    review2 = st.text_input('Tell us about the instructor?')
    review3 = st.text_input('Was the material provided useful?')

    # Perform sentiment analysis and show the results
    if st.button('Predict'):
        result1 = predict_sentiment(review1)
        result2 = predict_sentiment(review2)
        result3 = predict_sentiment(review3)
        st.success(f"Course experience: {result1}")
        st.success(f"Instructor: {result2}")
        st.success(f"Material: {result3}")
        
        # Show analytics using a bar chart
        results = {'Course experience': result1, 'Instructor': result2, 'Useful material': result3}
        df = pd.DataFrame({'Reviews': list(results.keys()), 'Sentiment': list(results.values())})
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

