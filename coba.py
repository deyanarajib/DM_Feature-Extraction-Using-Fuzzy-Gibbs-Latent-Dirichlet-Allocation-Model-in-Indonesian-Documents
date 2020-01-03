rawdata = []
for j in range(0,8):
    x = open(str(j+1)+'.txt','r').read()
    rawdata.append(x.replace('\n',' '))

import nltk
from nltk.tokenize import word_tokenize as token
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string, numpy as np

ST = StemmerFactory()
stemmer = ST.create_stemmer()
SW = StopWordRemoverFactory()
stop_word = SW.get_stop_words()

doc=[]
for i in rawdata:
    temp=[]
    for j in token(i):
        word = stemmer.stem(str.lower(j))
        #if word not in stop_word and len(word) > 2 and not word.startswith(tuple(string.punctuation)+tuple([str(k) for k in range(10)])+tuple('¿')):
        temp.append(word)
    doc.append(temp)

dictionary=[]
for i in doc:
    for j in i:
        if j not in dictionary:
            dictionary.append(j)

maxloop = 100


docidx = []
for i in doc:
    temp=[]
    for word in i:
        temp.append(dictionary.index(word))
    docidx.append(temp)

#Inisialisasi
N = 2
    
Z = []
for i in range(len(doc)):
    temp=[]
    for j in range(len(doc[i])):
        temp.append(np.random.multinomial(6, [1/N]*2, size=1))
    Z.append(temp)




