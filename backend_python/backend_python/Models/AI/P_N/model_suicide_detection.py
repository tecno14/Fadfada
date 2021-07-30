import joblib
import pickle
import numpy
import re

from nltk import ngrams
# from nltk.tokenize import sent_tokenize, word_tokenize  
from sklearn.feature_extraction.text import CountVectorizer
# import keras
# from tensorflow.keras.models import load_model
# from ar_corrector.corrector import Corrector
# corr = Corrector()

# ------------------------------------------------

a_file = open("./dict_P_N.pkl", "rb")
dict_P_N = pickle.load(a_file)

feature_path_P = './featureP.pkl'
loaded_vec_P = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open(feature_path_P, "rb")))
# Load TfidfTransformer
tfidftransformer_path_P = './tfidftransformerP.pkl'
tfidftransformer_P = pickle.load(open(tfidftransformer_path_P, "rb"))

model_P_N = pickle.load(open("./SVM.sav", 'rb'))

# ------------------------------------------------

def prper_data_P(text,n=3):
    """
    n : number of gram
    """
    # n=3
    
    Link=numpy.zeros((2,200))
    vec_emotion_P_N=vec_emotion_tfidf=list()

    # vec_emotion= numpy.zeros((4,tfidf_array.shape[1]))
    sentence = text
    tfidf = tfidftransformer_P.transform(loaded_vec_P.transform([sentence]))
    tfidf=tfidf.toarray()

    for k in range(n):

        term_ngrams = ngrams(sentence.split(), (k+1))
        term_ngrams=list(term_ngrams)
        for kk in range(len(term_ngrams)):
            if n !=0:
                word=""
                for kkk in range(len(term_ngrams[kk])):
                    if kkk == 0:
                        word = term_ngrams[kk][kkk]
                    else:
                        word = word +" "+ term_ngrams[kk][kkk]

            else:
                word=term_ngrams[kk][0]

            index_sentence=loaded_vec_P.vocabulary_.get(word)

            if index_sentence == None:
                T_F=0
            else: 
                T_F=tfidf[0][index_sentence]


            E_P_N=dict_P_N.get(word)


            if E_P_N ==None:
                E_P_N=0


            vec_emotion_tfidf.append(T_F)
            vec_emotion_P_N.append(E_P_N)

    for e1 in range(len(vec_emotion_tfidf)):
        if e1>199:
            break
        else:
            Link[0][e1]=vec_emotion_tfidf[e1]

    for e1 in range(len(vec_emotion_P_N)):
        if e1>199:
            break
        else:
            Link[1][e1]=vec_emotion_P_N[e1]

    vec_emotion_tfidf=numpy.hstack(( Link[0], Link[1]))
    vec_emotion_tfidf=numpy.array(vec_emotion_tfidf)
    return vec_emotion_tfidf

# ------------------------------------------------

def model_P(text):
    text=prper_data_P(text)
    predictions = model_P_N.predict([text])
    return predictions

# ------------------------------------------------

# usage:
# text = "بدي موت"
# res = model_P(text)
# print(res[0])