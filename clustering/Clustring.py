from __future__ import print_function, unicode_literals, division
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from urduhack.config import LEMMA_LOOKUP_TABLE_PATH
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
sns.set()
import re
import json
from sklearn.cluster import KMeans

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
                #print("word found:", Final_text[i], " At index:", i)   #comparing our Cleaned text with Positive lexicon and assigning the sentence score as +1





    index_list = []
    for i in range(len(Final_text)):
         for j in range(len(N_list)):
            if Final_text[i] == N_list[j]:
                N_score += -1
                index_list.append(i)
                #print("word found:", Final_text[i], "At index:", i)  #comparing our Cleaned text with negative lexicon and assigning the sentence score -1

    Total_score=P_score+N_score
    return Total_score


    # if Total_score >=0 :
    #     print("Sentence is Positive/neutral",Total_score) #Printing Sentence Sentiment
    # if Total_score < 0:
    #         print("Sentence is Negative",Total_score) #Printing Sentence Sentiment



def Text_preprocessing(sentence):


    '''STEP 1
    Sentence Tokenization
    '''
    Tokenized_sentence=word_tokenizer(sentence)

    '''STEP 2
        Sentence Preprocessing
        1.Normalzie WhiteSpace
        2.Remove Punctuation
        3.Remove accents
        4.Replace urls
        5.Replace emails
        6.Replace numbers
        7.Replace phone numbers
        8.Replace currancy symbols
        9.Remove English alphabhets
        #tokenized sentence into string for preprocessing
        ALL DONE BY USING URDUHACK API
    '''
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

    '''STEP 3
        Stop words removal
        REMOVING STOPWORDS BY COMPARING SENTENCES WORDS WITH STOPWORDS FILe
    '''
    my_file = open("stopwords-ur.txt", "r",encoding='utf-8')
    content = my_file.read()
    Stopwords=content.split()
    input_sentence = re.findall(r'[\u0600-\u06ff]+', Preprocessed_text)
    Stopwords_removed = [word for word in input_sentence if not word in Stopwords]
    #print("Original Text :", sentence)
    # print("Text after Preprocessing & Stopwords removal:",Stopwords_removed)
    lemma = lemma_lookup(Stopwords_removed)     #LEMMATIZATION FUCNTION CALL'''
    return lemma #returning Final Text





'''STEP 4
    Lemmatization
'''

_WORD2LEMMA = None


def lemma_lookup(text, lookup_path: str = LEMMA_LOOKUP_TABLE_PATH) -> list:
    tokens = text
    global _WORD2LEMMA
    if _WORD2LEMMA is None:
        with open(lookup_path, "r", encoding="utf-8") as file:
            _WORD2LEMMA = json.load(file)

    return [_WORD2LEMMA[word] if word in _WORD2LEMMA else word for word in tokens]


dataset = pd.read_csv('2DREP.csv')
x = dataset.iloc[:, [0, 1]].values
print(x)
wcss_list = []  # Initializing the list for the values of WCSS

# Using for loop for iterations from 1 to 10.
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss_list.append(kmeans.inertia_)


plt.plot(range(1, 11), wcss_list)
plt.title('The Elobw Method Graph')
plt.xlabel('Number of clusters(k)')
plt.ylabel('wcss_list')
plt.show()

kmeans = KMeans(n_clusters=4, init='k-means++', random_state= 42)
y_predict= kmeans.fit_predict(x)

plt.scatter(x[y_predict == 0, 0], x[y_predict == 0, 1], s = 100, c = 'blue', label = 'Cluster 1') #for first cluster
plt.scatter(x[y_predict == 1, 0], x[y_predict == 1, 1], s = 100, c = 'green', label = 'Cluster 2') #for second cluster
plt.scatter(x[y_predict== 2, 0], x[y_predict == 2, 1],  s = 100, c = 'red', label = 'Cluster 3') #for third cluster
plt.scatter(x[y_predict == 3, 0], x[y_predict == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4') #for fourth cluster
plt.scatter(x[y_predict == 4, 0], x[y_predict == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5') #for fifth cluster
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroid')
plt.title('Clusters of Sentences')
# plt.xlabel('Positivity of sentence')
# plt.ylabel('Negativity of sentence')
plt.legend()
plt.show()




cluster1x=x[y_predict == 0,0] #returning x cordiantes of each clusters into a list
cluster2x=x[y_predict == 1,0]
cluster3x=x[y_predict == 2,0]
cluster4x=x[y_predict == 3,0]
indexes1x=[]
indexes2x=[]
indexes3x=[]
indexes4x=[]





import pandas
df = pandas.read_csv('2DREP.csv')
for i in range (0,len(df)):
    for j in range(len(cluster1x)):
        if df['X_List'][i]==cluster1x[j]:
            indexes1x.append(i)

for i in range (0,len(df)):
    for j in range(len(cluster2x)):
        if df['X_List'][i]==cluster2x[j]:
            indexes2x.append(i)

for i in range (0,len(df)):
    for j in range(len(cluster3x)):
        if df['X_List'][i]==cluster3x[j]:
            indexes3x.append(i)

for i in range (0,len(df)):
    for j in range(len(cluster4x)):
        if df['X_List'][i]==cluster4x[j]:
            indexes4x.append(i)

print("Cluster 1 Indexes:",indexes1x)
print("Cluster 2 Indexes:",indexes2x)
print("Cluster 3 Indexes:",indexes3x)
print("Cluster 4 Indexes:",indexes4x)




df1 = pandas.read_csv('SpeechToText.csv')
res = pd.DataFrame()




Cluster1_sent_avg=0
print("Cluster 1 Sentences")
for i in range (0,len(df1)):
    for j in range(len(indexes1x)):
        if i==indexes1x[j]:
            print(df1['0'][i])


            Final_text=Text_preprocessing(df1['0'][i])

            Cluster1_sent_avg=Sentiment_score(Final_text)
            print(Cluster1_sent_avg)


print("Avergae Sentiment Score of cluster 1:",Cluster1_sent_avg/i)


Cluster2_sent_avg=0
print("Cluster 2 Sentences")
for i in range (0,len(df1)):
    for j in range(len(indexes2x)):
        if i==indexes2x[j]:


            print(df1['0'][i])
            Final_text = Text_preprocessing(df1['0'][i])
            Cluster2_sent_avg += Sentiment_score(Final_text)
print("Avergae Sentiment Score of cluster 2:",Cluster2_sent_avg/i)



Cluster3_sent_avg=0
print("Cluster 3 Sentences")
for i in range (0,len(df1)):
    for j in range(len(indexes3x)):
        if i==indexes3x[j]:
            print(df1['0'][i])
            Final_text = Text_preprocessing(df1['0'][i])
            Cluster3_sent_avg += Sentiment_score(Final_text)
print("Avergae Sentiment Score of cluster 3:",Cluster3_sent_avg/i)



Cluster4_sent_avg=0
print("Cluster 4 Sentences")
for i in range (0,len(df1)):
    for j in range(len(indexes4x)):
        if i==indexes4x[j]:
            print(df1['0'][i])
            Final_text = Text_preprocessing(df1['0'][i])
            Cluster4_sent_avg += Sentiment_score(Final_text)
print("Avergae Sentiment Score of cluster 4:",Cluster4_sent_avg/i)
#







