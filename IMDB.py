# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:55:08 2019

@author: user
"""

f=open('reviews.txt')
raw_reviews=f.readlines()
print(raw_reviews[2])
f.close()
g=open('labels.txt')
raw_labels=g.readlines()
print(raw_labels[2])
g.close()

tokens=list(map(lambda x: set(x.split(' ')),raw_reviews))
# Makes a single dictionary 
print(len(tokens)) # 25000
print(tokens[4]) # 25000

vocab=set()
for sent in tokens:
    for word in sent:
        if(len(word)>0):
            vocab.add(word)

print(len(vocab))
vocabs=list(vocab)
print(len(vocabs))

print(len(raw_reviews))
print(len(raw_labels))

wordindex=dict()
for i,word in enumerate(vocabs):
    wordindex[word]=i

input_dataset=list()
for sent in tokens:
    sent_indices=list()
    for word in sent:
        try:
            sent_indices.append(wordindex[word])
        except:
            ''
    input_dataset.append(sent_indices)

print(len(wordindex.keys())) # 74074
print(len(input_dataset))

target_dataset=list()
for label in raw_labels:
    if label=='positive\n':
        target_dataset.append(1)
    else:
        target_dataset.append(0)
     
print(len(vocabs))

import numpy as np
import sys
hidden_size=100
alpha=0.01
iterations=2
np.random.seed(1)
def sigmoid(x):
    return (1/(1+np.exp(-x)))

w0=0.2*np.random.random((len(vocabs),hidden_size))-0.1
w1=0.2*np.random.random((hidden_size,1))-0.1
print(np.sum(w0[1]))
correct,total=0,0
for j in range(iterations):
    for i in range(len(input_dataset)-1000):
        x,y=(input_dataset[i],target_dataset[i])
        l1=sigmoid(np.sum(w0[x],axis=0))
        l2=sigmoid(np.dot(l1,w1))
        l2_delta=l2-y
        l1_delta=l2_delta.dot(w1.T)
        w0[x]-=(l1_delta*alpha)
        w1-=np.outer(l1.T,l2_delta)
        if np.abs(l2_delta)<0.5:
            correct+=1
        total+=1
        if(i%10==0):
            progress=str(i/float(len(input_dataset)))
            sys.stdout.write('\rIter:'+str(j)+' Review:'+str(i)+'Progress:'+progress[0:5]+' Train Acc.:'+str((correct/total)*100)+'%')
correct,total=0,0
for i in range(len(input_dataset)-1000,len(input_dataset)):
    x,y=input_dataset[i],target_dataset[i]
    l1=sigmoid(np.sum(w0[x],axis=0))
    l2=sigmoid(np.dot(l1,w1))
    if np.abs(l2-y)<0.5:
        correct+=1
    total+=1
print('\nTest_accuracy : ',str((correct/total)*100)+'%')


import math
from collections import Counter
def get(target='beautiful'):
    target_index=wordindex[target]
    scores=Counter()
    for word,i in wordindex.items():
        raw_diff=w0[i]-w0[target_index]
        sq=raw_diff*raw_diff
        scores[word]=-math.sqrt(sum(sq))
        # Counter with most common gives the words or values which are higher.
    return scores.most_common(10)

print('All similar words are : ',get())


import random
np.random.seed(1)
tokens=list(map(lambda x: set(x.split(' ')),raw_reviews))
wordcnt=Counter()
b=0
for sent in tokens:
    for word in sent:
        b+=1
        wordcnt[word]-=1
print(wordcnt['beautiful'])
print(wordcnt.most_common(10)[0][0])
vocab=list(map(lambda x: x[0],wordcnt.most_common()))
word2index={}
print(vocab[0])
print(len(vocab))
for i,word in enumerate(vocab):
    word2index[word]=i
print(len(word2index))
input_dataset=list()
concatenated=list()
c=0
for sent in tokens:
    sent_indices=list()
    for word in sent:
        c+=1
        try:
            sent_indices.append(word2index[word])
            concatenated.append(word2index[word])
        except:
            ''
    input_dataset.append(sent_indices)
print(c)
print(b)
print(len(word2index.keys()))
concatenated=np.array(concatenated)
print(concatenated.shape)
print(input_dataset[1])
random.shuffle(input_dataset)
print(input_dataset[1])
def similar(target='beautiful'):
    target_index=wordindex[target]
    scores=Counter()
    for word,i in wordindex.items():
        raw_diff=w0[i]-w0[target_index]
        sq=raw_diff*raw_diff
        scores[word]=-math.sqrt(sum(sq))
        # Counter with most common gives the words or values which are higher.
    return scores.most_common(10)
hidden_size=100
iterations,alpha=2,0.05
negative,window=5,2
w0=(np.random.rand(len(vocab),hidden_size)-0.5)*0.2
w1=np.random.rand(len(vocab),hidden_size)*0
l2_target=np.zeros(negative+1)
l2_target[0]=1
print(len(input_dataset*iterations)) # 50000
for rev_i,review in enumerate(input_dataset*iterations):
    for target_i in range(len(review)):
        target_samples=[review[target_i]]+list(concatenated[(np.random.rand(negative)*len(concatenated)).astype('int').tolist()])
        left_context=review[max(0,target_i-window):target_i]
        right_context=review[target_i+1:min(len(review),target_i+window)]
        l1=np.mean(w0[left_context+right_context],axis=0)
        l2=sigmoid(np.dot(l1,w1[target_samples].T))
        l2_delta=l2-l2_target
        l1_delta=l2_delta.dot(w1[target_samples])
        w0[left_context+right_context]-=alpha*l1_delta
        w1[target_samples]-=alpha*(np.outer(l2_delta,l1))
    if(rev_i%250==0):
        sys.stdout.write('\rProgress:'+str(float(rev_i/len(input_dataset*iterations)))+ '    Similar Words:'+str(similar('terrible')))
    sys.stdout.write('\rProgress:'+str(float(rev_i/len(input_dataset*iterations))))  
    
def analogy(positive=['terrible','good'],negative=['bad']):
    norms=np.sum(w0*w0,axis=1) 
    print(norms.shape)  # (74075,)
    norms.resize(norms.shape[0],1)
    print(norms.shape) # (74075, 1)
    normed_w=w0*norms
    print(normed_w.shape)  # (74075, 100)
    query=np.zeros(len(w0[0]))
    for word in positive:
        query+=normed_w[word2index[word]]
    for word in negative:
        query-=normed_w[word2index[word]]
    scores=Counter()
    for word,index in word2index.items():
        raw_diff=w0[index]-query
        sq=raw_diff*raw_diff
        scores[word]=-math.sqrt(sum(sq))
    return scores.most_common(10)

print(analogy())
print(analogy(['elizabeth','he'],['she']))
    