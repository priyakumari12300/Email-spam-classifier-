import streamlit as st  
import pickle 
import nltk
from nltk.corpus import stopwords
from string import punctuation
import nltk
nltk.download('stopwords')
import sklearn
nltk.download('punkt_tab')

stop=stopwords.words('english')
stop_word_list=list(punctuation)+stop
from nltk.stem import PorterStemmer
p=PorterStemmer()

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:] # cloning list to variable
    y.clear()
    for i in text:
        if i not in stop_word_list:
            
            y.append(p.stem(i))
    return ' '.join(y)

tfidf = pickle.load(open('vectorization.pkl','rb'))
model= pickle.load(open('model.pkl','rb'))

st.title('Email Spam Classifier') 
input_email=st.text_area('Enter the mail')


if st.button('Predict'):
    transformed_email=transform_text(input_email)
    vectorize=tfidf.transform([transformed_email])
    result=model.predict(vectorize)[0]

    if result==1:
        st.header('spam')
    else:
       st.header('not Spam')

