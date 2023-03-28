# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 18:25:56 2023

@author: EXSON
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import streamlit as st
nltk.download('stopwords')



classifier = load_model('trained_model.h5')
cv=pickle.load(open('count-Vectorizer.pkl','rb'))
sc=pickle.load(open('Standard-Scaler.pkl','rb'))

def predict_sentiment1(input_review):
        input_review = re.sub(pattern='[^a-zA-Z]',repl=' ', string=input_review)
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
        pred = loaded_model.predict(input_X)
        pred = (pred > 0.5)
        if pred[0][0]:
            print("Positive review")
        else:
            print("Negative review")
    
    
    
    
    
def main():
    
    st.title('Student sentiment analysis')
    
    
    review1=st.text_input('How was the course experience?')
    review2=st.text_input('Tell us about the instructor?')
    review3=st.text_input('Was the material provided useful?')
    
    
    result1=''
    result2=''
    result3=''
    
    
    if st.button('predict'):
        result1= predict_sentiment1(review1)
        result2= predict_sentiment1(review2)
        result3= predict_sentiment1(review3)
      
        
    st.success(result1)
    st.success(result2)
    st.success(result3)
    
   
    

if __name__=='__main__':
    main()
