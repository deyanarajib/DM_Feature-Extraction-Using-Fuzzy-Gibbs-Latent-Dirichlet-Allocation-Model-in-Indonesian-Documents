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

#rawdata
print('rawdata')
print(rawdata)

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

#dictionary
print('dictionary')
print(dictionary)

maxloop = 100


docidx = []
for i in doc:
    temp=[]
    for word in i:
        temp.append(dictionary.index(word))
    docidx.append(temp)

#docidx    
print('docidx')
print(docidx)

#Inisialisasi
N = 2
    
Z = []
for i in range(len(doc)):
    temp=[]
    for j in range(len(doc[i])):
        temp.append(np.random.randint(N))
    Z.append(temp)
#z
print('Z')
print(Z)

flatZ = np.hstack(Z)
flatd = np.hstack(doc)

W=[]
for i in dictionary:
    idx = np.where(flatd==i,True,False)
    temp=[]
    for j in range(N):
        temp.append(list(flatZ[idx]).count(j))
    W.append(temp)
W = np.transpose(W)

#w
print('W')
print(W)

Y=[]
for i in Z:
    temp=[]
    for j in range(N):
        temp.append(i.count(j))
    Y.append(temp)

#y
print('Y')
print(Y)

alfa,beta = 1,1
V = len(dictionary)

for h in range(0,50):
    for i in range(len(doc)):
        for j in range(len(doc[i])):
            to = Z[i][j]
            wid = docidx[i][j]
            Y[i][to] = Y[i][to]-1
            W[to][wid] = W[to][wid]-1  
#        left = (W[to][wid]+beta)/(sum(W[to])+V*beta)
#        right = (Y[i][to]+alfa)/(sum(Y[i])+N*alfa)
            left = (np.array(W)[:,wid]+beta)/(sum(W[to])+V*beta)
            right = (np.array(Y)[i]+alfa)/(sum(Y[i])+N*alfa)
            prob = left*right 
            Znew = np.argmax(prob)
        
            Z[i][j] = Znew
            to = Z[i][j]
            Y[i][to] = Y[i][to]+1
            W[to][wid] = W[to][wid]+1
#import rajib

print('----------------')

print('Z baru')
print(Z)
print('W baru')
print(W)
print('Y baru')
print(Y)


Pwt = []
for i in range(len(W)):
    temp=[]
    for j in range(len(W[0])):
        result = (W[i][j]+beta)/(sum(W[i])+V*beta)
        temp.append(result)
    Pwt.append(temp)
#pwt
print('Pwt')
print(Pwt)

Pdt = []
for i in range(len(Y)):
    temp=[]
    for j in range(len(Y[0])):
        result = (Y[i][j]+alfa)/(sum(Y[i])+N*alfa)
        temp.append(result)
    Pdt.append(temp)

#pdt
print('Pdt')
print(Pdt)
