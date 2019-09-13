# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 22:07:14 2019

@author: user
"""
import numpy as np
from collections import Counter
import random
import math
import sys

f=open('reviews.txt')
raw_reviews=f.readlines()
print(raw_reviews[2])
f.close()
g=open('labels.txt')
raw_labels=g.readlines()
g.close()

tokens=list(map(lambda x: set(x.split(' ')),raw_reviews))
# Makes a single dictionary 
print(len(tokens)) # 25000
print(tokens[4]) 

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

target_dataset=list()
for label in raw_labels:
    if label=='positive\n':
        target_dataset.append(1)
    else:
        target_dataset.append(0)
        
hidden_size=100
alpha=0.01
iterations=1
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
            sys.stdout.write('\rIter:'+str(j)+' Review:'+str(i)+' Progress:'+progress[0:5]+' Train Acc.:'+str((correct/total)*100)+'%')
correct,total=0,0
for i in range(len(input_dataset)-1000,len(input_dataset)):
    x,y=input_dataset[i],target_dataset[i]
    l1=sigmoid(np.sum(w0[x],axis=0))
    l2=sigmoid(np.dot(l1,w1))
    if np.abs(l2-y)<0.5:
        correct+=1
    total+=1
print('\nTest_accuracy : ',str((correct/total)*100)+'%')

norms=np.sum(w0*w0,axis=1)
norms.resize(norms.shape[0],1)
norms_w=w0*norms
def make_sent_vect(words):
    indices=list(map(lambda x: wordindex[x],filter(lambda x : x in wordindex,words)))
    return np.mean(norms_w[indices],axis=0)

reviews2vect=[]
for sent in tokens:
    reviews2vect.append(make_sent_vect(sent))
reviews2vect=np.array(reviews2vect)
print(reviews2vect.shape) # (25000, 100)

def most_similar_reviews(review):
    v=make_sent_vect(review)
    scores=Counter()
    print(v.shape) # (100,)
    print(reviews2vect.dot(v).shape) # (25000,)
    for i,val in enumerate(reviews2vect.dot(v)):
        scores[i]=val
    most_similar=list()
    for idx,score in scores.most_common(3):
        most_similar.append(raw_reviews[idx][0:45])
    return most_similar,idx

print(most_similar_reviews(['amazing','good'])[0],'\n.....Review No.-',most_similar_reviews(['ammazing','good'])[1])


def softmax(x):
    x=np.atleast_2d(x)
    temp=np.exp(x)
    return (temp/(np.sum(temp,axis=1,keepdims=True)))

word_vects={}
word_vects['yankees'] = np.array([[0.,0.,0.]])
word_vects['bears'] = np.array([[0.,0.,0.]])
word_vects['braves'] = np.array([[0.,0.,0.]])
word_vects['red'] = np.array([[0.,0.,0.]])
word_vects['socks'] = np.array([[0.,0.,0.]])
word_vects['lose'] = np.array([[0.,0.,0.]])
word_vects['defeat'] = np.array([[0.,0.,0.]])
word_vects['beat'] = np.array([[0.,0.,0.]])
word_vects['tie'] = np.array([[0.,0.,0.]])

sent2output=np.random.rand(3,len(word_vects))
identity=np.eye(3)
l0=word_vects['red']
l1=l0.dot(identity)+word_vects['socks']
l2=l1.dot(identity)+word_vects['defeat']
pred=softmax(l2.dot(sent2output))
print(pred)
print(l0,l1.shape,l2)
y = np.array([1,0,0,0,0,0,0,0,0])
# Back-Propagation :-
alpha=0.01
pred_delta=(pred-y)
l2_delta=pred_delta.dot(sent2output.T)
defeat_delta=l2_delta*1
l1_delta=l2_delta.dot(identity.T)
socks_delta=l1_delta*1
l0_delta=l1_delta.dot(identity)
word_vects['red']-=l0_delta*alpha
word_vects['socks']-=socks_delta*alpha
word_vects['defeat']-=defeat_delta*alpha
identity-=np.outer(l0,l1_delta)*alpha
identity-=np.outer(l1,l2_delta)*alpha
sent2output=np.outer(l2,pred_delta)*alpha
print(identity)



f = open('tasksv11/en/qa1_single-supporting-fact_train.txt','r')
raw = f.readlines()
f.close()
tokens=list()
for line in raw:
    tokens.append(line.lower().replace('\n','').split(' ')[1:])

print(len(tokens))
print(tokens[0:3])
vocab=set()
for sent in tokens:
    for word in sent:
        vocab.add(word)
vocab=list(vocab)
print(len(vocab))
word2index={}
for i,word in enumerate(vocab):
    word2index[word]=i

def word2indices(sent):
    idx=[]
    for word in sent:
        idx.append(word2index[word])
    return idx

def softmax(x):
    e_x=np.exp(x-np.max(x))
    return (e_x/np.sum(e_x,axis=0))

embed_size=10
embed=(np.random.rand(len(vocab),embed_size)-0.5)*0.1
start=np.zeros(embed_size)
recurrent=np.eye(embed_size)
decoder=(np.random.rand(embed_size,len(vocab))-0.5)*0.1
one_hot=np.eye(len(vocab))
print(start)
print(decoder.shape) # 10,83
print(one_hot)

# Forward Propagation :-
def predict(sent):
    layers=list()
    layer={}
    layer['hidden']=start
    layers.append(layer)
    loss=0
    #print(layers[-1]['hidden'])
    for target_i in range(len(sent)):
        layer={}
        layer['pred']=softmax(layers[-1]['hidden'].dot(decoder)) # (83, )
        #print(layer['pred'])
        #print(layer['pred'][sent[target_i]])
        loss+=-np.log(layer['pred'][sent[target_i]])
        #print(loss)
        #print(embed[sent[target_i]])
        layer['hidden']=layers[-1]['hidden'].dot(recurrent)+embed[sent[target_i]]
        layers.append(layer)
    return layers,loss
sent=word2indices(tokens[4])
print(sent) # [15, 4, 49, 10, 54]
predict(word2indices(tokens[4]))

for iter in range(30000):
    alpha=0.001
    sent=word2indices(tokens[iter%len(tokens)][1:])
    #print(tokens[1])
    #print(sent)
    layers,loss=predict(sent)
   # print(len(layers)) 
    for idx in reversed(range(len(layers))):
        layer=layers[idx]
        target=sent[idx-1]
        if idx>0:
            layer['output_delta']=layer['pred']-one_hot[target]
            new_hidden_delta=layer['output_delta'].dot(decoder.T)
            if (idx==len(layers)-1):
                layer['hidden_delta']=new_hidden_delta
            else:
                layer['hidden_delta']=new_hidden_delta+layers[idx+1]['hidden_delta'].dot(recurrent.T)
        else:
            layer['hidden_delta']=layers[idx+1]['hidden_delta'].dot(recurrent.T)
    start-=(layers[0]['hidden_delta']*alpha)/float(len(sent))       
    for idx,layer in enumerate(layers[1:]):
        decoder-=(np.outer(layers[idx]['hidden'],layer['output_delta'])*alpha)/float(len(sent))
        target=sent[idx]
        embed[target]-=(layers[idx]['hidden_delta']*alpha)/float(len(sent))
        recurrent-=(np.outer(layers[idx]['hidden'],layer['hidden_delta'])*alpha)/float(len(sent))
    if(iter % 1000 == 0):
        print("Perplexity:" + str(np.exp(loss/len(sent))))
        
sent_index=11
l,_=predict(word2indices(tokens[sent_index]))
print(tokens[sent_index])
for i,layer in enumerate(l[1:-1]):
    input=tokens[sent_index][i]
    target=tokens[sent_index][i+1]
    pred=vocab[layer['pred'].argmax()]
    print("Prev Input:" + input + (' ' * (12 - len(input))) +"True:" + target + (" " * (15 - len(target))) + "Pred:" + pred)
