import streamlit
import streamlit as slt
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def text_processing(text):
    # Converting text to Lower Case
    text = text.lower()

    # Tokenize i.e breaking down a string into words
    text = list(nltk.word_tokenize(text))
    temp = []

    # Removing Special Characters, stopwords and punctuations
    for word in text:
        if word.isalnum() and word not in stopwords.words('english') and word not in string.punctuation:
            temp.append(word)
    text = temp[:]
    temp.clear()

    # Stemming i.e bring the word into it's base form
    for word in text:
        temp.append(ps.stem(word))

    text = temp[:]
    return " ".join(text)



tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('mnb.pkl', 'rb'))


slt.title("SMS/Spam Classifier")
input_msg = slt.text_area("Enter the Message")


if slt.button('Predict'):
    #Step 1: Preprocess Message

    transformed_msg = text_processing(input_msg)

    #Step 2: Vectorize
    vectorized_msg = tfidf.transform([transformed_msg])

    #Step 3: Predict
    result = model.predict(vectorized_msg)[0]

    #Step 4: Display

    if result == 1:
        slt.header('SPAM')
    else:
        slt.header('NOT SPAM')
