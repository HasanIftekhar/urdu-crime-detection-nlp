import streamlit as st
import speech_recognition as sr
import pandas as pd
from googletrans import Translator
from glob import glob
import speech_recognition as sr
from tinytag import TinyTag
import re
import pandas as pd
import csv
import codecs
import urduhack
from urduhack.preprocessing import normalize_whitespace
from urduhack.preprocessing import remove_punctuation
from urduhack.preprocessing import remove_accents
from urduhack.preprocessing import replace_urls
from urduhack.preprocessing import replace_emails
from urduhack.preprocessing import replace_phone_numbers
from urduhack.preprocessing import replace_numbers
from urduhack.preprocessing import replace_currency_symbols
from urduhack.preprocessing import remove_english_alphabets
from urduhack.tokenization import word_tokenizer
from typing import FrozenSet
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import re
import json
from urduhack.config import LEMMA_LOOKUP_TABLE_PATH
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import KeyedVectors
import pandas as pd
import nltk
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
model = KeyedVectors.load_word2vec_format('urduvec_140M_100K_300d.bin', binary=True)
import random

def Text_preprocessing(text2):



    Tokenized_sentence=word_tokenizer(text2)
    String_for_preprocessing = ' '.join([str(elem) for elem in Tokenized_sentence])
    normalized_text = normalize_whitespace(String_for_preprocessing)
    punctuation_Removed=remove_punctuation(normalized_text)
    Removed_accents=remove_accents(punctuation_Removed)
    Replaced_urls=replace_urls(Removed_accents)
    Replaced_emails=replace_emails(Replaced_urls)
    Replaced_phonenumbers=replace_phone_numbers(Replaced_emails)
    Replaced_numbers=replace_numbers(Replaced_phonenumbers)
    Replaced_currency=replace_currency_symbols(Replaced_numbers)
    Removed_english_alpha=remove_english_alphabets(Replaced_currency)
    Preprocessed_text=Removed_english_alpha


    my_file = open("stopwords-ur.txt", "r",encoding='utf-8')
    content = my_file.read()
    Stopwords=content.split()
    input_sentence = re.findall(r'[\u0600-\u06ff]+', Preprocessed_text)
    Stopwords_removed = [word for word in input_sentence if not word in Stopwords]
    print("Original Text :", text2)
    # print("Text after Preprocessing & Stopwords removal:",Stopwords_removed)
    lemma = lemma_lookup(Stopwords_removed)     #LEMMATIZATION FUCNTION CALL'''
    return lemma #returning Final Text

_WORD2LEMMA = None


def lemma_lookup(text2, lookup_path: str = LEMMA_LOOKUP_TABLE_PATH) -> list:
    tokens = text2
    global _WORD2LEMMA
    if _WORD2LEMMA is None:
        with open(lookup_path, "r", encoding="utf-8") as file:
            _WORD2LEMMA = json.load(file)

    return [_WORD2LEMMA[word] if word in _WORD2LEMMA else word for word in tokens]

def Sentiment_score(Final_text):

    P_file = open('poisitive-words.ur.txt', "r", encoding='utf-8')   #Opening Positive urdu lexicons
    P_content = P_file.read()
    P_list = P_content.split("\n")
    N_file = open('negative-words.ur.txt', "r", encoding='utf-8')  #Opening Negative urdu lexicons
    N_content = N_file.read()
    N_list = N_content.split("\n")
    P_score = 0
    N_score = 0

    for i in range(len(Final_text)):
        for j in range(len(P_list)):
            if Final_text[i] == P_list[j]:
                P_score += 1
                print("word found:", Final_text[i], " At index:", i)   #comparing our Cleaned text with Positive lexicon and assigning the sentence score as +1





    index_list = []
    for i in range(len(Final_text)):
         for j in range(len(N_list)):
            if Final_text[i] == N_list[j]:
                N_score += -1
                index_list.append(i)
                print("word found:", Final_text[i], "At index:", i)  #comparing our Cleaned text with negative lexicon and assigning the sentence score -1

    Total_score=P_score+N_score

    if Total_score >=0 :
        return "Sentence is Positive/neutral",Total_score/10 #Printing Sentence Sentiment
    if Total_score < 0:
            return "Sentence is Negative",Total_score/10 #Printing Sentence Sentiment
xlist= []
ylist = []
def tsne_plot(model,text):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []
    sum3 = 0

    for word in text:
        try:
            labels.append(word)
            tokens.append(model[word])
        except:
            pass



    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])


    sumx=x[0]+x[1] #for average distance of first points
    sumy=y[0]+y[1]

    avgx = sumx/2
    avgy=sumy/2

    for i in range(2,len(x)): # Sentence had 20 points. Convert to 1 by taking average.

        sum1x=avgx+ x[i]
        sum1y=avgy+y[i]


        avgx= sum1x/2
        avgy=sum1y/2



    xlist.append(avgx)  #list with x coordinates per sentence
    ylist.append(avgy)  #list with y coordinates per sentence




import pandas





text=''
text2=''
st.markdown("<h1 style='text-align: center; color: White;'>Welcome To Crime Detection</h1>", unsafe_allow_html=True)
audiofile= st.file_uploader("Pick an audio File",accept_multiple_files=True )
if audiofile:

    if st.button("Convert Audio To Text"):
        r = sr.Recognizer()
        for file in range(0, len(audiofile), 1):
            with sr.AudioFile(audiofile[file]) as source:
                audio = r.listen(source)
                text = r.recognize_google(audio, language='ur-pk')

                st.title("Audio Demo")
                st.audio(audiofile[file])
                st.title("Speech To text Conversion")
                st.write(text)

            with open('STT.txt', 'w', encoding='utf8') as f1:
                f1.write(text)

            with open('STT.txt', 'r', encoding='utf8') as f2:
                text2 = f2.readline()

            st.title("Text After PreProcessing")
            final_text = Text_preprocessing(text2)
            st.write(final_text)
            st.title("Sentiment Score of the Sentence")
            score=Sentiment_score(final_text)
            print(score)
            st.write(score)
            tsne_plot(model, final_text)



        #
        # df = pandas.DataFrame(data={"X_List": xlist, "Y_list": ylist})
        # df.to_csv("Front.csv", sep=',', index=False)


        fig, ax = plt.subplots()

        # dataset = pd.read_csv('Front.csv')
        # x = dataset.iloc[:, [0, 1]].values
        # for i in range(1, 11):
        #     kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        #     kmeans.fit(x)
        # kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
        # y_predict = kmeans.fit_predict(x)
        # plt.scatter(x[y_predict == 0, 0], x[y_predict == 0, 1], s=100, c='blue',
        #             label='Cluster 1')  # for first cluster
        # plt.scatter(x[y_predict == 1, 0], x[y_predict == 1, 1], s=100, c='green',
        #             label='Cluster 2')  # for second cluster
        # plt.scatter(x[y_predict == 2, 0], x[y_predict == 2, 1], s=100, c='red',
        #             label='Cluster 3')  # for third cluster
        # plt.scatter(x[y_predict == 3, 0], x[y_predict == 3, 1], s=100, c='cyan',
        #             label='Cluster 4')  # for fourth cluster
        # plt.scatter(x[y_predict == 4, 0], x[y_predict == 4, 1], s=100, c='magenta',
        #             label='Cluster 5')  # for fifth cluster
        # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow',
        #             label='Centroid')
        # plt.title('Clusters of Sentences')
        # plt.xlabel('Positivity of sentence')
        # plt.ylabel('Negativity of sentence')
        # plt.legend()
        # plt.show()

        st.title("Sentence Plot")
        plt.scatter(xlist,ylist)

        st.pyplot(fig)





























