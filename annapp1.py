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
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
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
def generate_pdf(predictions):
    buffer = io.BytesIO()
    # create a new PDF with Reportlab
    p = canvas.Canvas(buffer, pagesize=letter)

    # Write prediction details in PDF
    y = 700
    for key, value in predictions.items():
        p.drawString(100, y, f'{key}: {value}')
        y -= 50

    # Draw the bar chart in PDF
    chart_width = 400
    chart_height = 300
    x = 100
    y -= 50
    fig = plt.gcf()
    fig.set_size_inches(chart_width/80, chart_height/80)
    imgdata = io.BytesIO()
    fig.savefig(imgdata, format='png')
    imgdata.seek(0)
    p.drawImage(imgdata, x, y - chart_height, width=chart_width, height=chart_height)
    p.showPage()
    p.save()

    # get the value of BytesIO buffer and write PDF to file
    pdf_value = buffer.getvalue()
    buffer.close()
    return pdf_value

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
        
    if st.button('Download report'):
        results = {'Course experience': result1, 'Instructor': result2, 'Useful material': result3}
        pdf_data = generate_pdf(results)
        st.download_button(
            label="Download Report",
            data=pdf_data,
            file_name='sentiment_analysis_report.pdf',
            mime='application/pdf'
        )
# Run the app
if __name__=='__main__':
    main()
