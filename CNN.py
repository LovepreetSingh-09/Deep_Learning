# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 20:33:30 2019

@author: user
"""

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
print(labels[256:384])
test_images=x_test.reshape(len(x_test),28*28)/255
test_labels=np.zeros((len(y_test),10))
for i,l in enumerate(y_test):
    test_labels[i][l]=1
print(test_labels.shape) # (10000, 10)
print(test_images.shape) # (10000, 784)
print(range(len(test_images)))

np.random.seed(1)
num_label=10
pixels_per_image=784
alpha,iterations=(2,300)
input_rows=28
input_cols=28
kernel_rows=3
kernel_cols=3
num_kernels=16
batch_size=128
hidden_size=((input_rows-kernel_rows)*(input_cols-kernel_cols)*num_kernels)
w1=0.2*np.random.random((hidden_size,num_label))-0.1
print(w1.shape) # (10000, 10)
kernels=0.02*np.random.random((kernel_rows*kernel_cols,num_kernels))-0.01
def tanh(x):
    return np.tanh(x)

def tanh2deriv(x):
    return (1-(x**2))

def softmax(x):
    temp=np.exp(x)
    return (temp/np.sum(temp,axis=1,keepdims=True))

def get_image_section(layer,row_from,row_to,col_from,col_to):
    section=layer[:,row_from:row_to,col_from:col_to]
    return section.reshape(-1,1,row_to-row_from,col_to-col_from)
i=0
for j in range(iterations):
    cnt=0
    for i in range(int(len(images)/batch_size)):
        batch_start,batch_end=((i*batch_size),((i+1)*batch_size))
        l0=images[batch_start:batch_end]
        l0=l0.reshape(l0.shape[0],28,28)
       # print(l0.shape) # (128, 28, 28)
        sects=list()
        # The following double loop run for 25*25=625 times
        for row_start in range(l0.shape[1]-kernel_rows):
            for col_start in range(l0.shape[2]-kernel_cols):
                sect=get_image_section(l0,row_start,row_start+kernel_rows,col_start,col_start+kernel_cols)
               # print(sect.shape) # (128, 1, 3, 3)
                sects.append(sect)
       # print(sects[1].shape) # (128, 1, 3, 3)
        extended_input=np.concatenate(sects,axis=1)
        es=extended_input.shape
      #  print(es) # (128, 625, 3, 3)
        flattened_input=extended_input.reshape(es[0]*es[1],-1)
      #  print(flattened_input.shape) # (80000, 9)
        kernel_output=np.dot(flattened_input,kernels)
        #print(kernel_output.shape) # (80000, 16)
        l1=tanh(kernel_output.reshape(es[0],-1))
       # print(kernel_output.reshape(es[0],-1).shape) # (128, 10000)
        dropout=np.random.randint(2,size=l1.shape)
        l1*=dropout*2
        l2=softmax(np.dot(l1,w1)) # (128, 10)
        for k in range(batch_size):
            labelset=labels[batch_start+k:batch_start+k+1]
            inc=int(np.argmax(l2[k:k+1])==np.argmax(labelset))
            cnt+=inc
        l2_delta=(labels[batch_start:batch_end]-l2)/(batch_size*l2.shape[0])
        l1_delta=np.dot(l2_delta,w1.T)*tanh2deriv(l1)
        l1_delta*=dropout
        w1+=alpha*np.dot(l1.T,l2_delta)
        l1_delta_reshape=l1_delta.reshape(kernel_output.shape)
       # print(l1_delta_reshape.shape) # (80000, 16)
        k_update=flattened_input.T.dot(l1_delta_reshape)
       # print(k_update.shape) # (9, 16)
        kernels-=alpha*k_update
    test_cnt=0
    for i in range(len(test_images)):
        l0=test_images[i:i+1]
        l0=l0.reshape(l0.shape[0],28,28)
        sects=list()
        for row_start in range(l0.shape[1]-kernel_rows):
            for col_start in range(l0.shape[2]-kernel_cols):
                sect=get_image_section(l0,row_start,row_start+kernel_rows,col_start,col_start+kernel_cols)
                sects.append(sect)
        t_extended_input=np.concatenate(sects,axis=1)
      #  print(t_extended_input.shape)
        es=t_extended_input.shape
      #  print(es[0],es[1])
        t_flattened_input=t_extended_input.reshape(es[0]*es[1],-1)
       # print(t_flattened_input.shape)
        kernel_output=np.dot(t_flattened_input,kernels)
        l1=tanh(kernel_output.reshape(es[0],-1))
        l2=softmax(np.dot(l1,w1))
        test_cnt+=int(np.argmax(l2)==np.argmax(test_labels[i:i+1]))
    sys.stdout.write('\r'+ 'I:'+ str(j) + ' Test_acc: '+ str(test_cnt/float(len(test_images))) + ' Train_acc: '+ str(cnt/float(len(images))))

