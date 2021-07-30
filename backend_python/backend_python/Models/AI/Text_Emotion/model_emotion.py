import joblib
import pickle
import numpy
import re

from nltk import ngrams
from nltk.tokenize import sent_tokenize, word_tokenize  
from sklearn.feature_extraction.text import CountVectorizer
# import keras
from tensorflow.keras.models import load_model
# from ar_corrector.corrector import Corrector
# corr = Corrector()

# ------------------------------------------------

a_file = open("./dict_Emotion_N.pkl", "rb")
dict_Emotion_N = pickle.load(a_file)

a_file = open("./dict_Emotion_N.pkl", "rb")
dict_Emotion_HS = pickle.load(a_file)

a_file = open("./dict_Emotion_N.pkl", "rb")
dict_Emotion_AF = pickle.load(a_file)

feature_path = './feature.pkl'
loaded_vec = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open(feature_path, "rb")))

# Load TfidfTransformer
tfidftransformer_path = './tfidftransformer.pkl'
tfidftransformer = pickle.load(open(tfidftransformer_path, "rb"))

Model_Emotion = load_model('./model.h5')
# Model_Emotion = keras.models.load_model('./model.h5') 

# ------------------------------------------------

def clean_data(text):
  search = ["أ","إ","آ","ة","_","-","/",".","،"," و "," يا ",'"',"ـ","'","ى",
              "\\",'\n', '\t','&quot;','?','؟','!']
  replace = ["ا","ا","ا","ه"," "," ","","",""," و"," يا",
               "","","","ي","",' ', ' ',' ',' ? ',' ؟ ', ' ! ']
  longation = re.compile(r'(.)\1+')
  subst = r"\1\1"
  text = re.sub(longation, subst, text)
  text = re.sub(r"[^\w\s]", '', text)
  text = re.sub(r"[a-zA-Z]", '', text)
  text = re.sub(r"\d+", ' ', text)
  text = re.sub(r"\n+", ' ', text)
  text = re.sub(r"\t+", ' ', text)
  text = re.sub(r"\r+", ' ', text)
  text = re.sub(r"\s+", ' ', text)
  text = re.sub("أ+", "ا", text)
  text = re.sub("اا+", "ا", text)
  text = re.sub("وو+", "و", text)
  text = re.sub("يي+", "ي", text)
  text = re.sub("هه+", "ه", text)
  text = re.sub("🤣🤣+","🤣", text)
  text = re.sub("😂😂+","😂", text)
  text = re.sub("😭😭+","😭", text)
  text = re.sub("😱😱+","😱", text)
  text = re.sub("😡😡+","😡", text)
  text = re.sub("😀😀+","😀", text)


  for i in range(0, len(search)):
     text = text.replace(search[i], replace[i])
    
  text = text.strip()
 # text=corr.contextual_correct(text)
  stop_word = open('./stop word.txt',encoding="utf8")
  stop_word=stop_word.read()
  remov=list()
  terms= word_tokenize(text)


  for i in range(len(terms)):
        if terms[i] in stop_word:
            print("({}) removed".format(terms[i]))
            remov.append(terms[i])
            
            
  for i in remov:
    terms.remove(i)
  cancot=""
  for i in terms:
        if cancot=="":
            cancot=i
        else:
            cancot=cancot+" "+i
  text=cancot          
  return text

# ------------------------------------------------

def prper_data(text, n=3):
    """
    n : number of gram
    """
    
    Link=numpy.zeros((4,151))
    vec_emotion_tfidf=vec_emotion_N=vec_emotion_AF=vec_emotion_HS=Cnn_F=list()

    sentence = clean_data(text)
    tfidf = tfidftransformer.transform(loaded_vec.transform([sentence]))
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
            
            index_sentence=loaded_vec.vocabulary_.get(word)
            
            if index_sentence ==None:
                T_F=0
            else:
                T_F=tfidf[0][index_sentence]  
                
            E_N=dict_Emotion_N.get(word)
            E_AF=dict_Emotion_AF.get(word)
            E_HS=dict_Emotion_HS.get(word)

            
            if E_N ==None:
                E_N=0
            if E_AF ==None:
                E_AF=0
            if E_HS ==None:
                E_HS=0  
                
            vec_emotion_tfidf.append(T_F)
            vec_emotion_N.append(E_N)
            vec_emotion_AF.append(E_AF)
            vec_emotion_HS.append(E_HS)
            
            
    for e1 in range(len(vec_emotion_tfidf)):
        Link[0][e1]=vec_emotion_tfidf[e1]
    
    for e1 in range(len(vec_emotion_N)):
        Link[1][e1]=vec_emotion_N[e1]
        
    for e1 in range(len(vec_emotion_AF)):
        Link[2][e1]=vec_emotion_AF[e1]
        
        
    for e1 in range(len(vec_emotion_HS)):
        Link[3][e1]=vec_emotion_HS[e1]
        
    Link = Link.reshape(1, Link.shape[0], Link.shape[1], 1)  
    
    return Link

# ------------------------------------------------

def model_emotion(text):
    text=prper_data(text)
    predictions = Model_Emotion.predict(text)
    max_index = numpy.argmax(predictions[0])
    emotion_detection = ('angry', 'sad', 'fear', 'neutral', 'happy')
    emotion_prediction = emotion_detection[max_index]
    return emotion_prediction 

# ------------------------------------------------

# usage:
# text="انا بكرهك"
# res = model_emotion(text)
# print(res)
