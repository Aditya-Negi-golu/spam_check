import streamlit as st
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model (1).pkl', 'rb'))

st.title("SPAM CHECKER")

input_text = st.text_area("Enter your text message")
if st.button('Predict'):
    # preprocess
    trans_text = transform_text(input_text)
    # vectorize
    vector_text = tfidf.transform([trans_text])
    # predict
    result = model.predict(vector_text)[0]
    if result == 1:
        st.header('SPAM ! Be aware')
    else:
        st.header('NOT SPAM ')
