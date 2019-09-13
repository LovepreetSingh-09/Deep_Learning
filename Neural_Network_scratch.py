# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 20:19:01 2019

@author: user
"""

import numpy as np
ab=[2,9.5,9.9]
bc=[3,0.8,0.8]
cd=[4,1.3,0.1]
inp=np.array([ab[0],bc[0],cd[0]])
print(inp)
weights=np.array([[2,3,4],[2,3,1],[2,3,4]])
gp=[0.1,1,0.1]
print(weights[0])
alpha=0.1
def w_sum(c,d):
    out=0
    for i in range(len(c)):
        out+=(c[i]*d[i])
    return out

def neural_network(a,b):
    output=list(range(len(b)))
    for i in range(len(b)):
        output[i]=w_sum(a,b[i])
    return output

def err(a,b):
    error=list(range(len(a)))
    for i in range(len(a)):
        error[i]=(a[i]-b[i])**2
    return error
    
def delta_(a,b):
    delta=list(range(len(a)))
    for i in range(len(a)):
        delta[i]=(a[i]-b[i])
    return delta

def w_d(a,b):
    weight_delta=[[None]*len(a)]*len(b)
    for i in range(len(a)):
        for j in range(len(b)):
            weight_delta[i][j]=a[i]-b[j]
    return weight_delta

for x in range(1):
    pred=neural_network(inp,weights)
    print(pred)
    error=err(pred,gp)
    print(error)
    delta=delta_(pred,gp)
    print(delta)
    weight_delta=w_d(delta,inp)
    print(weight_delta)
    for i in range(len(delta)):
        for j in range(len(inp)):
            weights[i][j]=weights[i][j]-(alpha*weight_delta[i][j])
        
    print(weights)
    print(error)
    print('\n\n',delta)

for x in range(1):
    print(inp,'\n\n',weights)
    pred=weights.dot(inp)
    print(pred)
    error=err(pred,gp)
    print(error)
    delta=delta_(pred,gp)
    print(delta)
    weight_delta=w_d(delta,inp)
    print(weight_delta)
    for i in range(len(delta)):
        for j in range(len(inp)):
            weights[i][j]=weights[i][j]-(alpha*weight_delta[i][j])
        
    print(weights)
    print(error)
    print('\n\n',delta)

print(inp.dot(weights))
print(list(range(len(weights))))
print(inp*weights)
hidden_weights=np.array([[0.4,0.5,0.3],[0.5,0.7,0.1],[0.4,0.7,0.4]])
for x in range(1):
    pred=weights.dot(inp)
    pred_f=hidden_weights.dot(pred)
    print(pred,'\n\n',pred_f)
    
weights = np.array([0.5,0.48,-0.7])
streetlights = np.array( [[ 1, 0, 1 ],[ 0, 1, 1 ], [ 0, 0, 1 ],[ 1, 1, 1 ], [ 0, 1, 1 ],[ 1, 0, 1 ] ] ) 
walk_vs_stop = np.array( [ 0, 1, 0, 1, 1, 0 ] )
for i in range(60):
    error=0
    for ri in range(len(streetlights)):
        inp=streetlights[ri]
        gp=walk_vs_stop[ri]
        pred=inp.dot(weights)
        err_1=(pred-gp)**2
        delta=pred-gp
        error+=err_1
        weights=weights-(alpha*inp*delta)
    print(error)
    print(weights)
    
def relu(x):
    return (x>0)*x

def relu2deriv(x):
    return (x>0)

alpha=0.2
hidden_size=4
streetlights = np.array( [[ 1, 0, 1 ],[ 0, 1, 1 ], [ 0, 0, 1 ],[ 1, 1, 1 ] ] ) 
np.random.seed(1)
w0=2*np.random.random((3,hidden_size))-1
w1=2*np.random.random((hidden_size,1))-1
walk_vs_stop = np.array( [ [ 1, 1, 0, 0 ] ]).T
print(walk_vs_stop[0:1].shape)
for i in range(60):
    error=0
    for j in range(len(streetlights)):
        l0=streetlights[j:j+1]
        l1=relu(np.dot(l0,w0))
        l2=np.dot(l1,w1)
        error=error+np.sum((l2-walk_vs_stop[j:j+1])**2)
        l1_delta=l2-walk_vs_stop[j:j+1]
        l0_delta=np.dot(l1_delta,w1.T)*relu2deriv(l1)
        w0=w0-(alpha*np.dot(l0.T,l0_delta))
        w1=w1-(alpha*np.dot(l1.T,l1_delta))
    if(i%10==9):
        print(str(error))

alpha=0.1
streetlights = np.array( [[ 1, 0, 1 ],[ 0, 1, 1 ], [ 0, 0, 1 ],[ 1, 1, 1 ], [ 0, 1, 1 ],[ 1, 0, 1 ] ] ) 
a0=streetlights[0:1]
b0=np.array([[1,3,1,2],[1,2,5,1],[7,9,2,1]])
b1=np.array([[1],[2],[1],[2]])
a1=relu(np.dot(a0,b0))
print(a1) # [[ 8 12  3  3]]
a2=np.dot(a1,b1)
print(a2) # [[41]]
d=50
e=np.sum((a2-d)**2) 
print(e)  # 81
a1_d=a2-d
print(a1_d) # [[-9]]
a0_d=np.dot(a1_d,b1.T)*relu2deriv(a1)
print(a0_d) # [[ -9 -18  -9 -18]]
b1=b1-(alpha*np.dot(a1.T,a1_d))
print(alpha*np.dot(a1.T,a1_d)) # [[ -7.2] [-10.8] [ -2.7] [ -2.7]]
print(b1) # [[ 8.2] [12.8] [ 3.7] [ 4.7]]
b0=b0-(alpha*np.dot(a0.T,a0_d))
print(alpha*np.dot(a0.T,a0_d)) # [[-0.9 -1.8 -0.9 -1.8] [-0.  -0.  -0.  -0. ] [-0.9 -1.8 -0.9 -1.8]]
print(b0) # [[ 1.9  4.8  1.9  3.8] [ 1.   2.   5.   1. ] [ 7.9 10.8  2.9  2.8]]
print(b0,b1)
print(np.sum(b0))
a=np.array([[[1,2,3,4],[1,3,5,7]],[[1,3,6,9],[2,4,7,9]]])
b=np.array([[[5,6,8,4],[1,3,4,3]],[[5,6,6,8],[3,7,9,0]]])
print(a.shape) # (2, 2, 4)
# It makes n rows of 3rd dimension with element having shape(2,2) 
# Thus, makes a 4D array of  shape (4, 1, 2, 2)
c=a.reshape(-1,1,2,2)
d=b.reshape(-1,1,2,2)
print(c)
print(d)
cd=list()
cd.append(c)
cd.append(d)
print(cd)
m=np.concatenate(cd,axis=1)
print(m)
print(m.shape) # (4, 2, 2, 2)
b1=np.array([[1,2],[2,3],[1,4],[2,5]])
a=np.array([1,2])
print(np.dot(b1,a))
print(np.outer(a,b1))
print(np.inner(a,b1))
print(np.max(b1))
print(b1.reshape(4,-1))
