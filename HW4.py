%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,recall_score,confusion_matrix


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


# get one 10 by 10 patch for each training image
patch_data=[]
for item in range(0,60000):
    patch_list=[]
    for x_slide in range(0,4):
        for y_slide in range(0,4):
            a=6*x_slide
            b=6*y_slide
            patch=[]
            #print(a,b)
            for i in range(0,10):
                for j in range(10):
                    patch.append(28*(i+b)+a+j)
            patch_list.append(patch)
    index=patch_list[random.randint(0,15)]
    patch_data.append(train_x[item][index])
   
   
#subsample 6000 patch out of 60000
sub_patch=random.sample(population=patch_data, k=6000) 


#first level cluster
cl1=KMeans(n_clusters =50)
cl1.fit(sub_patch)

#use first level cluster to predict all 60000 patches
cluster_labels = cl1.predict(patch_data)

# creat dictionary, key :cluster center, value patch from training 
dic50={}
keyList = list(range(0,50)) 
for i in keyList: 
    dic50[i] = []
    
for i in list(range(0,60000)):
    key=cluster_labels[i]
    dic50[key] .append(i)
    
# second level cluster
cl2_input=[]
cl2_model=[]
for j in range(0,50):
    cl2_input.append([patch_data[i] for i in dic50[j]])
    cl2_model.append(KMeans(n_clusters =30))
    cl2_model[j].fit(cl2_input[j]) 
    
    
# create padding for train
pad_train_x=train_x.copy()
for i in range(0,60000):
    a=pad_train_x[i].reshape((28,28))
    b=np.pad(a,((1,1),(1,1)),'constant')
    pad_train_x[i]=b.reshape((900,))
    
    
# create padding for test
pad_test_x=test_x.copy()
for i in range(0,10000):
    a=pad_test_x[i].reshape((28,28))
    b=np.pad(a,((1,1),(1,1)),'constant')
    pad_test_x[i]=b.reshape((900,))
    
    
# sample patch from padded train
patch_pad_train=[]
for item in range(0,60000):
    patch_list=[]
    for x_slide in range(0,4):
        for y_slide in range(0,4):
            x=6*x_slide+1
            y=6*y_slide+1    
            for a in range(-1,2):
                for b in range(-1,2):
                    patch=[]
                    for i in range(0,10):
                        for j in range(10):
                            patch.append(30*(i+(x+a))+(y+b)+j)
                    patch_list.append(pad_train_x[item][patch])
    patch_pad_train.append(patch_list)
    
    
# sample patch from padded test
patch_pad_test=[]
for item in range(0,10000):
    patch_list=[]
    for x_slide in range(0,4):
        for y_slide in range(0,4):
            x=6*x_slide+1
            y=6*y_slide+1    
            for a in range(-1,2):
                for b in range(-1,2):
                    patch=[]
                    for i in range(0,10):
                        for j in range(10):
                            patch.append(30*(i+(x+a))+(y+b)+j)
                    patch_list.append(pad_test_x[item][patch])
    patch_pad_test.append(patch_list)
    
    
    
# predict for all 60000*144 train patches using first level model
train_label=[]
for j in range(0,60000):

    label=cl1.predict(patch_pad_train[j])
    label_list=[]
    for i in range(0,144):

        a=label[i]
        b=cl2_model[a].predict(patch_pad_train[j][i].reshape(1,-1))[0]
        label_list.append(a*50+b+1)
                          
    train_label.append(label_list)
 
 
 
 
# predict for all 10000*144 test patches using first level model
test_label=[]
for j in range(0,10000):

    label=cl1.predict(patch_pad_test[j])
    label_list=[]
    for i in range(0,144):

        a=label[i]
        b=cl2_model[a].predict(patch_pad_test[j][i].reshape(1,-1))[0]
        label_list.append(a*50+b+1)
                          
    test_label.append(label_list)
    
    
rfc = RandomForestClassifier(n_estimators=500)
rf_model = rfc.fit(train_label, train_y)
pred=rfc.predict(test_label)
accuracy_score(test_y,pred)
