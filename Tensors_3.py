# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:35:47 2019

@author: user
"""
import numpy as np

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
        return ((pred-target)*(pred-target)).sum(0)

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


import sys,random,math
from collections import Counter
import numpy as np

f = open('tasksv11/en/qa1_single-supporting-fact_train.txt','r')
raw = f.readlines()
f.close()

tokens = list()
for line in raw[0:1000]:
    tokens.append(line.lower().replace("\n","").split(" ")[1:]) # To Remove the Line Number
print(tokens[1])
print(raw[1])
new_tokens=list()
for line in tokens:
    new_tokens.append(['-']*(6-len(line))+line)
tokens=new_tokens
print(tokens[2])
print(len(tokens))
vocab=set()
for line in tokens:
    for word in line:
        vocab.add(word)
vocab=list(vocab)
print(len(vocab))  
word2index={}
for i,word in enumerate(vocab):
    word2index[word]=i

def word2indices(sentence):
    idx=list()
    for word in sentence:
        idx.append(word2index[word])
    return idx

indices=list()
for line in tokens:
    idx=list()
    for word in line:
        idx.append(word2index[word])
    indices.append(idx)

data=np.array(indices)

embed = Embedding(vocab_size=len(vocab),dim=16)
model = RCNN(n_inputs=16, n_hidden=16, n_outputs=len(vocab))

criterion = CrossEntropyLoss()
optim = SGD(parameters=model.get_parameters() + embed.get_parameters(), alpha=0.05)

for iter in range(1000):
    batch_size = 100
    total_loss = 0
    hidden = model.init_hidden(batch_size=batch_size)
    for i in range(5):
        input = Tensor(data[0:batch_size,i], autograd=True)
       # print(input.data.shape) # (100,)
        rnn_input=embed.forward(input=input)
        #print(rnn_input.data.shape) # (100, 16)
        output,hidden=model.forward(input=rnn_input,hidden=hidden)
        #print(output.data.shape)
    target=Tensor(data[0:batch_size,i+1],autograd=True)
   # print(i)
    # print(target.data.shape)
    loss = criterion.forward(output, target)
    loss.backward()
    optim.step()
    total_loss += loss.data
    if(iter % 200 == 0):
        p_correct = (target.data == np.argmax(output.data,axis=1)).mean()
        print("Loss:",total_loss / (len(data)/batch_size),"% Correct:",p_correct)


batch_size = 1
hidden = model.init_hidden(batch_size=batch_size)
for t in range(5):
    input = Tensor(data[0:batch_size,t], autograd=True)
    rnn_input = embed.forward(input=input)
    output, hidden = model.forward(input=rnn_input, hidden=hidden)

target = Tensor(data[0:batch_size,t+1], autograd=True)    
loss = criterion.forward(output, target)

ctx = ""
for idx in data[0:batch_size][0][0:-1]:
    ctx += vocab[idx] + " "
print("Context:",ctx)
print("True:",vocab[target.data[0]])
print("Pred:", vocab[output.data.argmax()])


