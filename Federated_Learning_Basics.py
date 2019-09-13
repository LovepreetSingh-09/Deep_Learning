# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 19:57:00 2019

@author: user
"""
import sys,random,math
from collections import Counter
import numpy as np
import codecs
import copy

class Tensor(object):
    def __init__(self,data,autograd=False,creators=None,creation_op=None,id=None,grad=None):
        self.data=np.array(data)
        self.creators=creators
        self.creation_op=creation_op
        self.grad=None
        self.autograd=autograd
        self.children={}
        if id is None:
            self.id=np.random.randint(0,100000)
        else:
            self.id=id
        if creators is not None:
            for c in creators:
                if self.id not in c.children:
                    c.children[self.id]=1
                else:
                    c.children[self.id]+=1
    
    def children_accounted_for(self):
        for i,cnt in self.children.items():
            if cnt!=0:
                return False
        return True
    
    def backward(self,grad=None,grad_origin=None):
        if self.autograd:
            if grad is None:
                grad=Tensor(np.ones_like(self.data))
            if grad_origin is not None:
                if self.children[grad_origin.id]==0:
                    raise Exception
                else:
                    self.children[grad_origin.id]-=1
            
            assert grad.autograd==False
            
            if self.grad is None:
                self.grad=grad
            else:
                self.grad+=grad
            
            if self.creators is not None and (self.children_accounted_for() or grad_origin is None):
                if self.creation_op=='add':
                    self.creators[0].backward(self.grad,self)
                    self.creators[1].backward(self.grad,self)
                
                if self.creation_op=='sub':
                    self.creators[0].backward(Tensor(self.grad.data),self)
                    self.creators[1].backward(Tensor(self.grad.__neg__().data),self)
                
                if self.creation_op=='mul':
                    new=self.grad*self.creators[1]
                    self.creators[0].backward(new,self)
                    bew=self.grad*self.creators[0]
                    self.creators[1].backward(bew,self)
                
                if self.creation_op=='neg':
                    self.creators[0].backward(self.grad.__neg__())
                
                if self.creation_op=='mm':
                    act=self.creators[0]
                    l=self.creators[1]
                    b=self.grad.mm(l.transpose())
                    c=self.grad.transpose().mm(act).transpose()
                    act.backward(b)
                    l.backward(c)
                
                if self.creation_op=='transpose':
                    self.creators[0].backward(self.grad.transpose())
                
                if self.creation_op=='tanh':
                    ones=Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad*(ones-(self*self)))
                
                if self.creation_op=='sigmoid':
                    ones=Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (self *(ones-self)))
               
                if("sum" in self.creation_op):
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.expand(dim,self.creators[0].data.shape[dim]))

                if("expand" in self.creation_op):
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.sum(dim))
                    
                if self.creation_op=='index_select':
                    d=np.zeros_like(self.creators[0].data)
                    indices_=self.index_select_indices.data.flatten()
                    gr=grad.data.reshape(len(indices_),-1)
                    for i in range(len(indices_)):
                        d[indices_[i]]+=gr[i]
                    self.creators[0].backward(Tensor(d))
                
                if self.creation_op=='cross_entropy':
                    dis=self.softmax_output-self.target_dist
                    self.creators[0].backward(Tensor(dis))
                    
    def __add__(self,other):
        if self.autograd and other.autograd:
            return Tensor(self.data+other.data,creators=[self,other],creation_op='add',autograd=True)
        return Tensor(self.data + other.data)
    
    def __neg__(self):
        if self.autograd:
            return Tensor(self.data*-1,autograd=True,creators=[self],creation_op='neg')
        return Tensor(self.data*-1)
    
    def __sub__(self,other):
        if self.autograd and other.autograd:
            return Tensor(self.data-other.data,creators=[self,other],creation_op='sub',autograd=True)
        return Tensor(self.data-other.data)
    
    def __mul__(self,other):
        if self.autograd and other.autograd:
            return Tensor(self.data*other.data,creators=[self,other],creation_op='mul',autograd=True)
        return Tensor(self.data*other.data)
    
    def sum(self,dim):
        if self.autograd:
            return Tensor(self.data.sum(dim),creators=[self],creation_op='sum_'+str(dim),autograd=True)
        return Tensor(self.data.sum(dim))
        
    def mm(self,x):
        if self.autograd:
            return Tensor(self.data.dot(x.data),creators=[self,x],creation_op='mm',autograd=True)
        return Tensor(self.data.dot(x.data))
        
    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(),creators=[self],creation_op='transpose',autograd=True)
        return Tensor(self.data.transpose())
    
    def expand(self, dim,copies):
        trans_cmd = list(range(0,len(self.data.shape)))
        trans_cmd.insert(dim,len(self.data.shape))
        new_data = self.data.repeat(copies).reshape(list(self.data.shape) + [copies]).transpose(trans_cmd)
        if(self.autograd):
            return Tensor(new_data,autograd=True,creators=[self],creation_op="expand_"+str(dim))
        return Tensor(new_data)
    
    def sigmoid(self):
        if self.autograd:
            return Tensor(1/(1+np.exp(-self.data)),autograd=True,creators=[self],creation_op='sigmoid')
        return Tensor(1/(1+np.exp(-self.data)))
    
    def tanh(self):
        if self.autograd:
            return Tensor(np.tanh(self.data),autograd=True,creators=[self],creation_op='tanh')
        return Tensor(np.tanh(self.data))
    
    def softmax(self):
        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp,axis=len(self.data.shape)-1, keepdims=True)
        return softmax_output
    
    def index_select(self,indices):
        if self.autograd:
            new=Tensor(self.data[indices.data],autograd=True,creators=[self],creation_op='index_select')
            new.index_select_indices=indices
            return new
        return Tensor(self.data[indices.data])
       
    def cross_entropy(self,target_indices):
        temp=np.exp(self.data)
        softmax_output=temp/np.sum(temp,axis=len(self.data.shape)-1,keepdims=True)
        t=target_indices.data.flatten()
        p=softmax_output.reshape(len(t),-1)
        target_dist=np.eye(p.shape[1])[t]
        loss=-(np.log(p)*(target_dist)).sum(1).mean()
        if self.autograd:
            out=Tensor(loss,autograd=True,creators=[self],creation_op='cross_entropy')
            out.softmax_output=softmax_output
            out.target_dist=target_dist
            return out
        return Tensor(loss)
        
    def __repr__(self):
        return str(self.data.__repr__())
    
    def __str__(self):
        return str(self.data.__str__())
                
class SGD(object):
    def __init__(self,parameters,alpha=0.01):
        self.parameters=parameters
        self.alpha=alpha
        
    def zero(self):
        for w in self.parameters:
            w.grad.data*=0
        
    def step(self,zero=True):
        for w in self.parameters:
           # print(w.grad)
            w.data-=w.grad.data*self.alpha
            if zero:
                w.grad.data*=0
                
class Layer(object):
    def __init__(self):
        self.parameters=list()
    def get_parameters(self):
        return self.parameters
    
class Linear(Layer):
    def __init__(self,n_inputs,n_outputs):
        super().__init__()
        w=np.random.randn(n_inputs,n_outputs)*np.sqrt(2.0/(n_inputs))
        self.weights=Tensor(w,autograd=True)
        self.bias=Tensor(np.zeros(n_outputs),autograd=True)
        self.parameters.append(self.weights)
        self.parameters.append(self.bias)
    
    def forward(self,input):
        return input.mm(self.weights)+self.bias.expand(0,len(input.data))

class Sequential(Layer):
    def __init__(self,layers=list()):
        super().__init__()
        self.layers=layers
        
    def add(self,layer):
        self.layers.append(layer)
    
    def forward(self,input):
        for l in self.layers:
            input=l.forward(input)
        return input
    
    def get_parameters(self):
        params=list()
        for l in self.layers:
            params+=l.get_parameters()
        return params

class Tanh(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self,input):
        return input.tanh()

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self,input):
        return input.sigmoid()

class MSELoss(Layer):
    def __init__(self):
        super().__init__()
    def forward(self,pred,target):
        dif=pred-target
        return (dif*dif).sum(0)

class Embedding(Layer):
    def __init__(self,vocab_size,dim):
        super().__init__()
        self.vocab=vocab_size
        self.dim=dim
        self.weight=Tensor((np.random.rand(vocab_size,dim)-0.5)/dim,autograd=True)
        self.parameters.append(self.weight)
        
    def forward(self,input):
        return self.weight.index_select(input)

class CrossEntropyLoss(Layer):
    def __init__(self):
        super().__init__()
    def forward(self,input,target):
        return input.cross_entropy(target)
    
class RCNN(Layer):
    def __init__(self,n_inputs,n_hidden,n_outputs,activation='sigmoid'):
        super().__init__()
        self.n_inputs=n_inputs
        self.n_hidden=n_hidden
        self.n_outputs=n_outputs
        if activation == 'sigmoid':
            self.activation=Sigmoid()
        elif activation == 'tanh':
            self.activation=Tanh()
        else:
            raise Exception('No Or Wrong Activation')
        self.w_ih=Linear(n_inputs,n_hidden)
        self.w_hh=Linear(n_hidden,n_hidden)
        self.w_ho=Linear(n_hidden,n_outputs)
        self.parameters+=self.w_ih.get_parameters()
        self.parameters+=self.w_hh.get_parameters()
        self.parameters+=self.w_ho.get_parameters()
        
    def forward(self,input,hidden):
        prev_hidden=self.w_hh.forward(hidden)
        combined=self.w_ih.forward(input)+prev_hidden
        new=self.activation.forward(combined)
        output=self.w_ho.forward(new)
        return output,new
    
    def init_hidden(self,batch_size=1):
        return Tensor(np.zeros((batch_size,self.n_hidden)),autograd=True)


with codecs.open('spam.txt','r',encoding='utf-8',errors='ignore') as f:
    raw=f.readlines()

vocab=set()
spam=list()
for row in raw:
    spam.append(row[:-2].split(' '))
    for word in spam[-1]:
        vocab.add(word)

print(len(raw))
       
with codecs.open('ham.txt','r',encoding='utf-8',errors='ignore') as f:
    raw=f.readlines()

print(len(raw))
ham=list()
for row in raw:
    ham.append(row[:-2].split(' '))
    for word in ham[-1]:
        vocab.add(word)

vocab.add('<udx>')
vocab=list(vocab)
print(len(vocab))

w2i={}
for i,word in enumerate(vocab):
    w2i[word]=i

def indices(input,l=500):
    index=list()
    for line in input:
        if len(line)<500:
            line=list(line)+(['<udx>']*(l-len(line)))
            idx=[]
            for word in line:
                idx.append(w2i[word])
            index.append(idx)
    return index

s=indices(spam)
h=indices(ham)
train_spam=s[:-1000]
train_ham=h[:-1000]
test_spam=s[-1000:]
test_ham=h[-1000:]
train_data=list()
train_target=list()
test_data=list()
test_target=list()

for i in range(max(len(train_ham),len(train_spam))):
    train_data.append(train_spam[i%len(train_spam)])
    train_target.append([1])
    train_data.append(train_ham[i%len(train_ham)])
    train_target.append([0])

for i in range(1000):
    test_data.append(test_spam[i])
    test_target.append([1])
    test_data.append(test_ham[i])
    test_target.append([0])

model=Embedding(vocab_size=len(vocab),dim=1)
model.weight.data*=0

def train(model,input_data,target_data,batch_size=150,iterations=5):
    criterion = MSELoss()
    optim = SGD(parameters=model.get_parameters(), alpha=0.003)
    n_batches=int(len(input_data)/batch_size)
    print(n_batches)
    for i in range(iterations):
        total_loss=0
        for batch_i in range(n_batches):
            model.weight.data[w2i['<udx>']]*=0
            input=Tensor(input_data[batch_i*batch_size:(batch_i+1)*batch_size],autograd=True)
            target=Tensor(target_data[batch_i*batch_size:(batch_i+1)*batch_size],autograd=True)
            pred=model.forward(input).sum(1).sigmoid()
            loss=criterion.forward(pred,target)
            total_loss+=loss.data[0]/batch_size
            loss.backward()
            optim.step()
            sys.stdout.write("\r\tLoss:" + str(total_loss / (batch_i+1)))
        print()
    return model
        
  

def test(model,test_data,target):
    model.weight.data[w2i['<udx>']]*=0
    input=Tensor(test_data,autograd=True)
    target=Tensor(target,autograd=True)
    pred=model.forward(input).sum(1).sigmoid()
    return ((pred.data>0.5)==target.data).mean()

for i in range(3):
    model=train(model,train_data, train_target, iterations=1)
    print("% Correct on Test Set: " + str(test(model,test_data, test_target)*100))
    
  
bob=[train_data[:1000],train_target[:1000]]
alice=[train_data[1000:2000],train_target[1000:2000]]
sue=[train_data[2000:],train_target[2000:]]

model=Embedding(len(vocab),dim=1)
model.weight.data*=0

for i in range(3):
    bob_model=train(copy.deepcopy(model),bob[0],bob[1],iterations=1)
    alice_model=train(copy.deepcopy(model),alice[0],alice[1],iterations=1)
    sue_model=train(copy.deepcopy(model),sue[0],sue[1],iterations=1)
    model.weight.data=(bob_model.weight.data+alice_model.weight.data+sue_model.weight.data)/3
    print('Accuracy: ',test(model,test_data,test_target))
    
email=['my','computer','password','is','pizza']
bobs_input=np.array([[w2i[word] for word in email]])
print(bobs_input)
bobs_output=np.array([[0]])
model=Embedding(vocab_size=len(vocab),dim=1)
model.weight.data*=0

bobs_model=train(copy.deepcopy(model),bobs_input,bobs_output,iterations=1,batch_size=1)
for i,val in enumerate(bobs_model.weight.data):
    if val!=0:
        print(vocab[i])

import phe
public_key,private_key=phe.generate_paillier_keypair(n_length=1024)
a=public_key.encrypt(5)
b=public_key.encrypt(4)
z=a+b
print(z)
c=6+4
c=public_key.encrypt(c)
print(c)
z=private_key.decrypt(c)
print(z)

public_key,private_key=phe.generate_paillier_keypair(n_length=128)

def train_encrypted(model,inpu_data,target_data,public_key):
    encrypted_weights=list()
    new_model=train(copy.deepcopy(model),inpu_data,target_data,batch_size=1)
    print(new_model.weight.data.shape)
    for val in new_model.weight.data[:,0]:
#        print(val) [:,0] To convert the array into scaler unit for encoding because encryption can only be done on scaler units.
        encrypted_weights.append(public_key.encrypt(val))
    ew=np.array(encrypted_weights).reshape(new_model.weight.data.shape)
    return ew

model=Embedding(vocab_size=len(vocab),dim=1)
model.weight.data*=0

bob_encrypted_model=train_encrypted(model,bob[0],bob[1],public_key)
alice_encrypted_model=train_encrypted(model,alice[0],alice[1],public_key)
sue_encrypted_model=train_encrypted(model,sue[0],sue[1],public_key)

aggregated_encrypted_model=bob_encrypted_model+alice_encrypted_model+sue_encrypted_model
raw_values=list()
print(aggregated_encrypted_model.flatten())
for val in aggregated_encrypted_model.flatten():
    raw_values.append(private_key.decrypt(val))

print(raw_values)
model.weight.data=np.array(raw_values).reshape(model.weight.data.shape)/3
print("\t% Correct on Test Set: " + str(test(model, test_data, test_target)*100))

