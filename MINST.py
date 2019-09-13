# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 20:34:44 2019

@author: user
"""
import numpy as np
from sklearn.datasets.base import get_data_home
print(get_data_home())
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
# or #
mnist = fetch_openml('mnist_784', version=1)
print(mnist.keys())
mnist.data=mnist.data/255
image,image_test,labels,labels_test=train_test_split(mnist.data,mnist.target,random_state=0)
print(image.shape)
print(labels.shape)
print(mnist.categories)
print(image[0])
def relu(x):
    return (x>=0)*x
def relu2deriv(x):
    return x>=0

hidden_size=40
alpha=0.005
np.random.seed(1)
labels=np.array(labels).astype(int)
labels=labels.reshape(-1,1)
print(labels[0:1].shape)
w0=0.2*np.random.random((784,hidden_size))-0.1
w1=0.2*np.random.random((hidden_size,1))-0.1
print(w1)
i=0
for iteration in range(50):
    error=0
    correct=0
    for i in range(len(image)):
        l0=image[i:i+1]
        l1=relu(np.dot(l0,w0))
        l2=np.dot(l1,w1)
        error+=np.sum((l2-labels[i:i+1])**2)
        # print(error)
        correct+=int(np.argmax(l2)==np.argmax(labels[i:i+1]))
        # print(l1)
        # print(labels[i:i+1])
        l2_delta=l2-labels[i:i+1]
        # print(l2_delta)
        l1_delta=np.dot(l2,w1.T)*relu2deriv(l1)
        # print(l1_delta)
        # print(l1_delta,l2_delta)
        w0=w0+(alpha*np.dot(l0.T,l1_delta))
        w1=w1+(alpha*np.dot(l1.T,l2_delta))
        # print(w0)
        # print(w1)
    print(error/len(image))
        
import sys 
import numpy as np
from keras.datasets import mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()
images,labels=(x_train[0:1000].reshape(1000,28*28)/255,y_train[0:1000])

one_hot_labels=np.zeros((len(labels),10))
# i= row no. ;  l= value
for i,l in enumerate(labels):
    one_hot_labels[i][l]=1
labels=one_hot_labels
# It gives 1 value to the row and column where the actual value is in the label
print(labels)

test_images=x_test.reshape(len(x_test),28*28)/255
test_labels=np.zeros((len(y_test),10))
for i,l in enumerate(y_test):
    test_labels[i][l]=1
print(test_labels.shape)
    
np.random.seed(1)
alpha=0.005
hidden_size=40
pixels_per_image=784
w0=0.2*np.random.random((pixels_per_image,hidden_size)) -0.1
w1=0.2*np.random.random((hidden_size,10)) -0.1
print(w0)
for iteration in range(350):
    error=0.0
    correct=0
    for i in range(len(images)):
        l0=images[i:i+1]
       # print(l2)
        l1=relu(np.dot(l0,w0))
        l2=np.dot(l1,w1)
        error=error+np.sum((labels[i:i+1]-l2)**2)
        #print(error)
        correct+=int(np.argmax(l2)==np.argmax(labels[i:i+1]))
       # print(correct)
        # print(l1)
        # print(labels[i:i+1])
        l2_delta=(l2-labels[i:i+1])
        # print(l2_delta)
        l1_delta=np.dot(l2_delta,w1.T)*relu2deriv(l1)
        #print(l1_delta)
        # print(l1_delta,l2_delta)
        w0=w0-(alpha*l0.T.dot(l1_delta))
        w1=w1-(alpha*l1.T.dot(l2_delta))
        # print(w0)
        # print(w1)
    sys.stdout.write('\r'+ 'I:'+str(iteration)+  'Error: '+str(error/float(len(images)))[0:5] + 'Correct: '+str(correct/float(len(images))))
    
error=0.0
cnt=0
for i in range(len(test_images)):
    l0=test_images[i:i+1]
    l1=relu(np.dot(l0,w0))
    l2=np.dot(l1,w1)
    error=error+np.sum((l2-test_labels[i:i+1])**2)
    cnt+=int(np.argmax(l2)==np.argmax(test_labels[i:i+1]))
sys.stdout.write('\r'+ 'I:'+str(i)+  'Error: '+str(error/float(len(test_images)))[0:5] + 'Correct: '+str(cnt/float(len(test_images))))


# Dropout and Batch Gradient Descent :-
hidden_size=100
batch_size=100
w0=0.2*np.random.random((784,hidden_size))-0.1
w1=0.2*np.random.random((hidden_size,10))-0.1
alpha=0.1
i=0
print(w0.shape,w1.shape)
for j in range(300):
    error=0
    correct=0
    for i in range(int(len(images)/batch_size)):
        batch_start,batch_end=(i*batch_size,(i+1)*batch_size)
        l0=images[batch_start:batch_end]
        l1=relu(np.dot(l0,w0))
        dropout=np.random.randint(2,size=l1.shape)
        l1*=dropout*2  # Becoz the dropout we used is 50 % 0 and 1 value So, to compensate for the weighted sum for l2 we use 1/0.5=2
        l2=np.dot(l1,w1)
        error+=np.sum((labels[batch_start:batch_end]-l2)**2)
        for k in range(batch_size):
            correct+=int(np.argmax(l2[k:k+1])==np.argmax(labels[batch_start+k:batch_start+k+1]))
        l2_delta=(labels[batch_start:batch_end]-l2)/batch_size
        l1_delta=np.dot(l2_delta,w1.T)*relu2deriv(l1)
        l1_delta*=dropout
        w0+=alpha*np.dot(l0.T,l1_delta)
        w1+=alpha*np.dot(l1.T,l2_delta)
    if(j%10==0):
        test_error=0
        test_correct=0
        for i in range(len(test_images)):
             l0=test_images[i:i+1]
             l1=relu(np.dot(l0,w0))
             l2=np.dot(l1,w1)
             test_error=test_error+np.sum((l2-test_labels[i:i+1])**2)
             test_correct+=int(np.argmax(l2)==np.argmax(test_labels[i:i+1]))
        sys.stdout.write('\r'+ 'I:'+ str(j) +  ' Test Error: '+ str(test_error/float(len(test_images)))[0:5] + ' Correct: '+ str(test_correct/float(len(test_images))) + '  Training  Error: '+ str(error/float(len(images)))[0:5] + ' Correct: '+ str(correct/float(len(images))))

def tanh(x):
    return np.tanh(x)

def tanh2deriv(x):
    return (1-(x**2))

def softmax(x):
    temp=np.exp(x)
    return (temp/(np.sum(temp,axis=1,keepdims=True)))
    
alpha=2
i=0
hidden_size=100
batch_size=100
w0=0.02*np.random.random((784,hidden_size))-0.01
w1=0.2*np.random.random((hidden_size,10))-0.1
for j in range(300):
    correct=0
    for i in range(int(len(images)/batch_size)):
        batch_start,batch_end=(i*batch_size,(i+1)*batch_size)
        l0=images[batch_start:batch_end]
        l1=tanh(np.dot(l0,w0))
        dropout=np.random.randint(2,size=l1.shape)
        l1*=dropout*2  # Becoz the dropout we used is 50 % 0 and 1 value So, to compensate for the weighted sum for l2 we use 1/0.5=2
        l2=softmax(np.dot(l1,w1))
        for k in range(batch_size):
            correct+=int(np.argmax(l2[k:k+1])==np.argmax(labels[batch_start+k:batch_start+k+1]))
        l2_delta=(labels[batch_start:batch_end]-l2)/(batch_size*l2.shape[0])
        l1_delta=np.dot(l2_delta,w1.T)*tanh2deriv(l1)
        l1_delta*=dropout
        w0+=alpha*np.dot(l0.T,l1_delta)
        w1+=alpha*np.dot(l1.T,l2_delta)
    if(j%10==0):
        test_error=0
        test_correct=0
        for i in range(len(test_images)):
             l0=test_images[i:i+1]
             l1=tanh(np.dot(l0,w0))
             l2=np.dot(l1,w1)
             test_correct+=int(np.argmax(l2)==np.argmax(test_labels[i:i+1]))
        sys.stdout.write('\r'+ 'I:'+ str(j) + ' Correct: '+ str(test_correct/float(len(test_images))) + ' Correct: '+ str(correct/float(len(images))))
