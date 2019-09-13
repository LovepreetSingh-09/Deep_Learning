# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 19:53:09 2019

@author: user
"""

import numpy as np
from collections import Counter
import random
import math
import sys

class Tensor(object):
    def __init__(self,data,creators=None,creators_op=None):
        self.data=np.array(data)
        self.creators=creators
        self.creators_op=creators_op
        self.grad=None
        
    def __add__(self,other):
        return Tensor(self.data+other.data,creators=[self,other],creators_op='add')
    
    def backward(self,grad):
        self.grad=grad
        if (self.creators_op=='add'):
            self.creators[0].backward(grad)
            self.creators[1].backward(grad)
            
    def __repr__(self):
        return str(self.data.__repr__())
    
    def __str__(self):
        return str(self.data.__str__())

x = Tensor([1,2,3,4,5])
y = Tensor([2,2,2,2,2])

z = x + y
z.backward(Tensor([1,1,1,1,1]))
print(z.grad) # [1, 1, 1, 1, 1]
print(x.grad) # [1, 1, 1, 1, 1]
print(y.grad) # [1, 1, 1, 1, 1]
print(z.creators) # [[1, 2, 3, 4, 5], [2, 2, 2, 2, 2]]
print(z.creators_op)   # add

a = Tensor([1,2,3,4,5])
b = Tensor([2,2,2,2,2])
c = Tensor([5,4,3,2,1])
d = Tensor([-1,-2,-3,-4,-5])

e = a + b
f = c + d
g = e + f

g.backward(Tensor(np.array([1,1,1,1,1])))
print(a.grad) # [1 1 1 1 1]


d = a + b
e = b + c
f = d + e
f.backward(Tensor(np.array([1,1,1,1,1])))

b.grad.data == np.array([2,2,2,2,2])
print(b.grad) # array([1, 1, 1, 1, 1])

# Upgrading Autograd to Support Multiple Tensors :-

class Tensor(object):
    
    def __init__(self,data,creators=None,creators_op=None,autograd=False,id=None,grad=None):
        self.data=data
        self.creators=creators
        self.creators_op=creators_op
        self.autograd=autograd
        self.grad=grad
        if id is None:
            self.id=np.random.randint(0,100000)
        else:
            self.id=id
        self.children={}
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
                grad=(np.ones_like(self.data))
            if grad_origin is not None:
                if self.children[grad_origin.id]==0:
                    raise Exception("Can't backpropagate more than once")
                else:
                    self.children[grad_origin.id]-=1
                    
        if self.grad is None:
            self.grad=grad
        else:
            self.grad+=grad
        
        # grads must not have grads of their own
        assert grad.autograd==False 
        
        if (self.creators is not None and (self.children_accounted_for() or grad_origin is None)):
            if self.creators_op=='add':
                self.creators[0].backward(self.grad,self)
                self.creators[1].backward(self.grad,self)
            
            
    def __add__(self,other):
        if self.autograd and other.autograd:
            return Tensor(self.data+other.data,creators=[self,other],creators_op='add',autograd=True)
        return (self.data+other.data)   
    
    def __repr__(self):
        return str(self.data.__repr__())
    
    def __str__(self):
        return str(self.data.__str__())

a = Tensor([1,2,3,4,5], autograd=True)
b = Tensor([2,2,2,2,2], autograd=True)
c = Tensor([5,4,3,2,1], autograd=True)

d = a + b
e = b + c
f = d + e

f.backward(Tensor(np.array([1,1,1,1,1])))
print(b.grad) # [2 2 2 2 2]
print(b.grad == np.array([2,2,2,2,2])) # True
print(b.children.items()) #  dict_items([(58649, 0), (51049, 0)])
print(f.children.items())
x = Tensor(np.array([[1,2,3], [4,5,6]]))
x.sum(0)
x.expand(dim=2, copies=4)

dim=4
copies=4
c=np.array([[3,5,3,5,8]])
print(len(c.shape))
trans_cmd = list(range(0,len(c.shape)))
print(c.shape)
print(trans_cmd)
trans_cmd.insert(dim,len(c.shape))
print(trans_cmd)
print(c.repeat(copies))
print((list(c.shape)+ [copies]))
print(c.repeat(copies).reshape(list(c.shape)+ [copies]))
new_data = c.repeat(copies).reshape(list(c.shape) + [copies]).transpose(trans_cmd)
print(new_data)
print(c.shape)
print(c.transpose(0,1))       

print(('sum_10'.split("_")[1]))
np.random.seed(0)

class Tensor(object):
    def __init__(self,data,creators=None,creators_op=None,id=None,grad=None,autograd=False):
        self.data=np.array(data)
        self.creators=creators
        self.creators_op=creators_op
        self.autograd=autograd
        self.grad=None
        self.children={}
        if id is None:
            self.id=np.random.randint(0,10000)
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
                    raise Exception('Cant Backpropagate more')
                else:
                    self.children[grad_origin.id]-=1
            
            if self.grad is None:
                self.grad=grad
            else:
                self.grad+=grad
            
            assert grad.autograd == False
            
            if self.creators is not None and (self.children_accounted_for()or grad_origin is None):
                if self.creators_op=='add':
                    self.creators[0].backward(self.grad,self)
                    self.creators[1].backward(self.grad,self)
                    
                if self.creators_op=='neg':
                    self.creators[0].backward(self.grad.__neg__())
                    
                if self.creators_op=='sub':
                    self.creators[0].backward(Tensor(self.grad.data),self)
                    self.creators[1].backward(Tensor(self.grad.__neg__().data),self)
                
                if self.creators_op=='mm':
                    act=self.creators[0]
                    fi=self.creators[1]
                    w=self.grad.mm(fi.transpose())
                    b=self.grad.transpose().mm(act).transpose()
                    self.creators[0].backward(w)
                    self.creators[1].backward(b)
                
                if self.creators_op=='transpose':
                    self.creators[0].backward(self.grad.transpose())
                    
                if self.creators_op=='mul':
                    c=self.grad*self.creators[1]
                    self.creators[0].backward(c,self)                    
                    a=self.grad*self.creators[0]
                    self.creators[1].backward(a,self)
                  
                if("sum" in self.creators_op):
                    dim = int(self.creators_op.split("_")[1])
                    self.creators[0].backward(self.grad.expand(dim,self.creators[0].data.shape[dim]))

                if("expand" in self.creators_op):
                    dim = int(self.creators_op.split("_")[1])
                    self.creators[0].backward(self.grad.sum(dim))
                
                if self.creators_op=='sigmoid':
                    one=Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad*(self*(one-self)))
                
                if self.creators_op=='tanh':
                    one=Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad*(one-(self*self)))
                    
    def __add__(self,other):
        if self.autograd and other.autograd:
            return Tensor(self.data+other.data,creators=[self,other],creators_op='add',autograd=True)
        return Tensor(self.data + other.data)
    
    def __neg__(self):
        if self.autograd:
            return Tensor(self.data*-1,autograd=True,creators=[self],creators_op='neg')
        return Tensor(self.data*-1)
    
    def __sub__(self,other):
        if self.autograd and other.autograd:
            return Tensor(self.data-other.data,creators=[self,other],creators_op='sub',autograd=True)
        return Tensor(self.data-other.data)
    
    def __mul__(self,other):
        if self.autograd and other.autograd:
            return Tensor(self.data*other.data,creators=[self,other],creators_op='mul',autograd=True)
        return Tensor(self.data*other.data)
    
    def sum(self,dim):
        if self.autograd:
            return Tensor(self.data.sum(dim),creators=[self],creators_op='sum_'+str(dim),autograd=True)
        return Tensor(self.data.sum(dim))
        
    def mm(self,x):
        if self.autograd:
            return Tensor(self.data.dot(x.data),creators=[self,x],creators_op='mm',autograd=True)
        return Tensor(self.data.dot(x.data))
        
    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(),creators=[self],creators_op='transpose',autograd=True)
        return Tensor(self.data.transpose())
    
    def expand(self, dim,copies):
        trans_cmd = list(range(0,len(self.data.shape)))
        trans_cmd.insert(dim,len(self.data.shape))
        new_data = self.data.repeat(copies).reshape(list(self.data.shape) + [copies]).transpose(trans_cmd)
        if(self.autograd):
            return Tensor(new_data,autograd=True,creators=[self],creators_op="expand_"+str(dim))
        return Tensor(new_data)
    
    def sigmoid(self):
        if self.autograd:
            return Tensor(1/(1+np.exp(-self.data)),autograd=True,creators=[self],creators_op='sigmoid')
        return Tensor(1/(1+np.exp(-self.data)))
    
    def tanh(self):
        if self.autograd:
            return Tensor(np.tanh(self.data),autograd=True,creators=[self],creators_op='tanh')
        return Tensor(np.tanh(self.data))
        
    def __repr__(self):
        return str(self.data.__repr__())
    
    def __str__(self):
        return str(self.data.__str__())
    
a = Tensor([1,2,3,4,5], autograd=True)
b = Tensor([3,5,3,5,8], autograd=True)
c = Tensor([5,4,3,2,1], autograd=True)

d = a + (-b)
e = (-b) + c
f = d + e

l=Tensor(np.array([1,1,1,1,1]))
f.backward(Tensor(np.array([1,1,1,1,1])))
print(b.children) # {1315: 1, 3475: 1}
print(b.grad.data == np.array([-2,-2,-2,-2,-2])) # [ True  True  True  True  True]
print(b) # [3 5 3 5 8]
print(-b.data) # [-3 -5 -3 -5 -8]

data = Tensor(np.array([[0,0],[0,1],[1,0],[1,1]]), autograd=True)
target = Tensor(np.array([[0],[1],[0],[1]]), autograd=True)

w = list()
class SGD(object):
    def __init__(self,parameters,alpha=0.01):
        self.parameters=parameters
        self.alpha=alpha
        
    def zero(self):
        for w in self.parameters:
            w.grad.data*=0
    
    def step(self,zero=True):
        for w in self.parameters:
            w.data-=w.grad.data*self.alpha
            if zero:
                w.grad.data=0

w.append(Tensor(np.random.rand(2,3), autograd=True))
w.append(Tensor(np.random.rand(3,1), autograd=True))
optim = SGD(parameters=w, alpha=0.1)
for i in range(10):
    pred=data.mm(w[0]).mm(w[1])
    loss=((pred-target)*(pred-target)).sum(0)
    loss.backward(Tensor(np.ones_like(loss.data)))
    for w_ in w:
        w_.data-=w_.grad.data*0.1
        w_.grad.data*=0
    print(loss.data)

print(target.grad) # [[-0. ] [ 4.16788677] [-4.8459507 ] [-0.67806393]]
print(np.ones_like(loss.data)) # [1.]
print(pred.grad) # [[ 0.        ] [-0.08326002] [ 0.09685373] [ 0.01359371]]

class layer(object):
    def __init__(self):
        self.parameters=list()
    def get_parameters(self):
        return self.parameters
    
class Linear(layer):
    def __init__(self,n_inputs,n_outputs):
        super().__init__()
        w=np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0/(n_inputs))
        self.weight=Tensor(w,autograd=True)
        self.bias=Tensor(np.zeros(n_outputs),autograd=True)
        self.parameters.append(self.weight)
        self.parameters.append(self.bias)
        
    def forward(self,input):
        return input.mm(self.weight)+self.bias.expand(0,len(input.data))


class SGD(object):
    def __init__(self,parameters,alpha=0.01):
        self.parameters=parameters
        self.alpha=alpha
        
    def zero(self):
        for w in self.parameters:
            w.grad.data*=0
    
    def step(self,zero=True):
        for w in self.parameters:
            w.data-=w.grad.data*self.alpha
            if zero:
                w.grad.data=0
                

a=Linear(2,3)
b=Linear(3,1)
a.get_parameters()


class Sequential(layer):
    def __init__(self,layers=list()):
        super().__init__()
        self.layers=layers
        
    def add(self,layers):
        self.layers.append(layers)
    
    def get_parameters(self):
        params=list()
        for l in self.layers:
            params+=l.get_parameters()
        return params
    
    def forward(self,input_):
        for l in self.layers:
            input_=l.forward(input_)
        return(input_)

class MSELoss(layer):
    def __init__(self):
        super().__init__()
    def forward(self,pred,target):
        return ((pred-target)*(pred-target)).sum(0)
  
data = Tensor(np.array([[0,0],[0,1],[1,0],[1,1]]), autograd=True)
target = Tensor(np.array([[0],[1],[0],[1]]), autograd=True)          
model=Sequential([Linear(2,3),Linear(3,1)])  
print(model.get_parameters()[0].grad)
criterion=MSELoss()
optim=SGD(parameters=model.get_parameters(),alpha=0.05)
print()
for i in range(10):
    pred=model.forward(data)
    #loss=((pred-target)*(pred-target)).sum(0)
    loss=criterion.forward(pred,target)
    loss.backward(Tensor(np.ones_like(loss.data)))
    optim.step()
    print(loss)
    
class Sigmoid(layer):
    def __init__(self):
        super().__init__()
    def forward(self,input):
        return input.sigmoid()

class Tanh(layer):
    def __init__(self):
        super().__init__()
    def forward(self,input):
        return input.tanh()
np.random.seed(0)
data = Tensor(np.array([[0,0],[0,1],[1,0],[1,1]]), autograd=True)
target = Tensor(np.array([[0],[1],[0],[1]]), autograd=True)
model = Sequential([Linear(2,3), Tanh(), Linear(3,1), Sigmoid()])
criterion = MSELoss()
optim = SGD(parameters=model.get_parameters(), alpha=1)
for i in range(10):
    # Predict
    pred = model.forward(data)
    # Compare
    loss = criterion.forward(pred, target)
    # Learn
    loss.backward(Tensor(np.ones_like(loss.data)))
    optim.step()
    print(loss)
model.get_parameters()
        










