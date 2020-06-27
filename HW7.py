%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from sklearn.metrics import mean_squared_error

#load data
image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
data_path = "C:\\Users\\wang\\dataset\\mnist\\"
train_data = np.loadtxt(data_path + "mnist_train.csv", 
                        delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv", 
                       delimiter=",") 


#get predictor and response from data
train_x=[i[1:] for i in train_data]
test_x=[i[1:] for i in test_data]
train_y=[i[0] for i in train_data]
test_y=[i[0] for i in test_data]

#make the range of pixels between 0 and 1
normalized=[]
for item in range(len(train_x[:500])):
    normalized.append(tuple(i/255 for i in train_x[item]))


#make the value of pixels only -1 and 1
binarized=[]
for item in range(500):
    binarized.append(tuple(-1 if i<0.5 else 1 for i in normalized[item]))
    
    
#16 is about 2% of 784 bits,flip 1 and -1
noised=np.array(binarized)
for j in range(500):
    sampled=random.sample(list(enumerate(noised[j])), 16)
    for idx,num in sampled:
        if noised[j][idx]==1:
            noised[j][idx]=-1
        else:
            noised[j][idx]=1
            
            
#find the corner node
corner_node=[[0,0],[0,27],[27,0],[27,27]]


#find the edge node
edge_node=[]
for j in range(1,27):
    edge_node.append([j,0])
    edge_node.append([0,j])
    edge_node.append([27,j])
    edge_node.append([j,27])
    
    
#update the probability for center nodes
def center_update():
    for i in range(26):
        for j in range(26):
            x=i+1
            y=j+1
            a_center=0.2*((2*pi_init[x-1,y]-1)+(2*pi_init[x+1,y]-1)+(2*pi_init[x,y-1]-1)+(2*pi_init[x,y+1]-1))+0.5*image[x,y]
            b_center=-0.2*((2*pi_init[x-1,y]-1)+(2*pi_init[x+1,y]-1)+(2*pi_init[x,y-1]-1)+(2*pi_init[x,y+1]-1))-0.5*image[x,y]
            pi_init[i+1,j+1]=np.exp(a_center)/(np.exp(a_center)+np.exp(b_center))
            
            
#update the probability for edge nodes
def edge_update():
    for item in edge_node:
        x=item[0]
        y=item[1]
        if x==0:
            a_center=0.2*((2*pi_init[x+1,y]-1)+(2*pi_init[x,y-1]-1)+(2*pi_init[x,y+1]-1))+0.5*image[x,y]
            b_center=-0.2*((2*pi_init[x+1,y]-1)+(2*pi_init[x,y-1]-1)+(2*pi_init[x,y+1]-1))-0.5*image[x,y]
        elif x==27:
            a_center=0.2*((2*pi_init[x-1,y]-1)+(2*pi_init[x,y-1]-1)+(2*pi_init[x,y+1]-1))+0.5*image[x,y]
            b_center=-0.2*((2*pi_init[x-1,y]-1)+(2*pi_init[x,y-1]-1)+(2*pi_init[x,y+1]-1))-0.5*image[x,y]
        elif y==0:
            a_center=0.2*((2*pi_init[x-1,y]-1)+(2*pi_init[x+1,y]-1)+(2*pi_init[x,y+1]-1))+0.5*image[x,y]
            b_center=-0.2*((2*pi_init[x-1,y]-1)+(2*pi_init[x+1,y]-1)+(2*pi_init[x,y+1]-1))-0.5*image[x,y]
        else:
            a_center=0.2*((2*pi_init[x-1,y]-1)+(2*pi_init[x+1,y]-1)+(2*pi_init[x,y-1]-1))+0.5*image[x,y]
            b_center=-0.2*((2*pi_init[x-1,y]-1)+(2*pi_init[x+1,y]-1)+(2*pi_init[x,y-1]-1))-0.5*image[x,y]
        pi_init[x,y]=np.exp(a_center)/(np.exp(a_center)+np.exp(b_center))
        
        
        
#update the probability for corner nodes
def corner_update():
    for item in corner_node:
        x=item[0]
        y=item[1]
        if (x==0) & (y==0):
            a_center=0.2*((2*pi_init[x+1,y]-1)+(2*pi_init[x,y+1]-1))+0.5*image[x,y]
            b_center=-0.2*((2*pi_init[x+1,y]-1)+(2*pi_init[x,y+1]-1))-0.5*image[x,y]
        elif (x==0) & (y==27):
            a_center=0.2*((2*pi_init[x+1,y]-1)+(2*pi_init[x,y-1]-1))+0.5*image[x,y]
            b_center=-0.2*((2*pi_init[x+1,y]-1)+(2*pi_init[x,y-1]-1))-0.5*image[x,y]
        elif (x==27) & (y==0):
            a_center=0.2*((2*pi_init[x-1,y]-1)+(2*pi_init[x,y+1]-1))+0.5*image[x,y]
            b_center=-0.2*((2*pi_init[x-1,y]-1)+(2*pi_init[x,y+1]-1))-0.5*image[x,y]
        else:
            a_center=0.2*((2*pi_init[x-1,y]-1)+(2*pi_init[x,y-1]-1))+0.5*image[x,y]
            b_center=-0.2*((2*pi_init[x-1,y]-1)+(2*pi_init[x,y-1]-1))-0.5*image[x,y]
        pi_init[x,y]=np.exp(a_center)/(np.exp(a_center)+np.exp(b_center))
        
        
        
pred_image=[]
acc=[]
for p in range(500):
    image=noised[p].reshape(28,28)
    #generate initial value for pi, the probability that the pixel is 1
    pi_init=[None]*784
    for j in range(784):
        pi_init[j]=random.random()
    #reshape pi_inint into matrix
    pi_init=np.array(pi_init).reshape(28,28)
    
    recursion = 0
    while((recursion == 0) or (diff > 1e-8)):
        old_pi=pi_init.copy()
        center_update()
        edge_update()
        corner_update()
        diff=mean_squared_error(pi_init,old_pi)
        recursion += 1
    #generate prediction based on node probability
    predict=[]
    for i in range(28):
        for j in range(28):
            if pi_init[i,j]>0.5:
                predict.append(1)
            else:
                predict.append(-1)
    pred_image.append(predict)
    #calculate accuracy

    correct=0
    for pred,label in zip(predict,binarized[p]):
        if pred==label:
            correct+=1
    acc.append(correct)
    
    
#fraction of accurate pixel prediction
sum(acc)/(500*784)


#original
for i in max_list:
    fig = plt.figure
    plt.imshow(np.array(binarized[i]).reshape(28,28), cmap='gray')
    plt.show()
    
    
    
#noised
for i in max_list:
    fig = plt.figure
    plt.imshow(np.array(noised[i]).reshape(28,28), cmap='gray')
    plt.show()
    
    

#denoised
for i in max_list:
    fig = plt.figure
    plt.imshow(np.array(pred_image[i]).reshape(28,28), cmap='gray')
    plt.show()
