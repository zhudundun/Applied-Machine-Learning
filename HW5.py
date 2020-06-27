from PIL import Image
import itertools
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

im = Image.open('smallsunset.jpg') # Can be many different formats.
pix = im.load()
sunset=list(im.getdata())

#make the range of pixels between 0 and 1
normalized=[]
for item in range(len(sunset)):
    normalized.append(tuple(i/255 for i in sunset[item]))
    
    
def identity(n):
    m = [[0 for i in range(n)] for j in range(n)]
    for q in range(0,n):
        m[q][q]=1
    return m
    
def identity2(n):
    m = [[0 for i in range(n)] for j in range(n)]
    for q in range(0,n):
        m[q][q]=1/400
    return m
    
#create initial cluster center
def initCentroids(K, X):
    # Select K points from datapoints randomly as centroids
    N = len(X)
    index = random.sample(range(N), K)
    centroids = np.zeros((K, 3))
    for i, j in enumerate(index):
        centroids[i] = X[j]

    return centroids
    
    
def Normal(Xi, Uk, Sk,d):
        # Calculate the value for Xi in normal distribution k
        # Xi stands for data[i]
        # Uk stands for mu[k]
        # Sk stands for sigma[k]
        # d stands for the dimension of datapoint
        probability = pow((2*math.pi), -d/2) * pow(abs(np.linalg.det(Sk)), -1/2) * \
                    np.exp(-1/2 * np.dot(np.dot((Xi-Uk).T, np.linalg.inv(Sk)), (Xi-Uk)))
        return probability
        
        
# parameter initialization
number_point=len(normalized)
number_cluster=10
cluster_weight=[1/number_cluster]*number_cluster
data=np.array(normalized)
cluster_mean=initCentroids(number_cluster, normalized)
cluster_cov=np.array([identity(3)]*number_cluster)


# w is the weight of each cluster
def maximizeLLH(N,K,w,X,U,S,d):
        # Calculate the maximum likelihood
        new_likelihood = 0
        for i in range(N):
            temp = 0
            for k in range(K):
                temp += w[k] * Normal(X[i], U[k], S[k], d)
            new_likelihood += np.log(temp)
        print("New_likelihood:",new_likelihood)

        return new_likelihood
        
        
        
def Estep(N,K,w,X,U,S,d):
    # E step
    print("Enter E step.")
    
    # Calculate r[k][i], which stands for Rik
    r=np.zeros([K,N]) 
    s = np.zeros(N)
    for i in range(N):
        temp = np.zeros(K)  # Temporary array
        # Calculate alpha[k]*N(Xi, Uk, Sk) for each data[i] and the summation of that in all distributions
        for k in range(K):
            temp[k] = float(w[k]) * Normal(X[i], U[k], S[k], d)
            s[i] += temp[k]
        for k in range(K):
            r[k][i] = temp[k]/s[i]
            
    return r
    
    
    
def Mstep(r,K,N,d,X):
    #M step
    print("Enter M step.")
    
    w = [None]*K
    U = np.zeros([K,3])

    for k in range(K):
        # Calculate alpha[k]
        w[k] = np.sum(r[k]) / N

        # Calculate mu[k]
        total = np.zeros(d)
        for i in range(N):
            total += r[k][i]* X[i]
        U[k] = total / np.sum(r[k])
        
    return w,U
    
    
new_lld = maximizeLLH(number_point,number_cluster,cluster_weight,data,cluster_mean,cluster_cov,3)
recursion = 0
likelihood=None
while((recursion == 0) or (new_lld - likelihood > 1e-2)):
    likelihood = new_lld
    point_weight=Estep(number_point,number_cluster,cluster_weight,data,cluster_mean,cluster_cov,3)
    cluster_weight,cluster_mean=Mstep(point_weight,number_cluster,number_point,3,data)
    new_lld =maximizeLLH(number_point,number_cluster,cluster_weight,data,cluster_mean,cluster_cov,3)
    recursion += 1
  
  
data10=np.zeros([198000,3])

for i in range(len(normalized)):
    idx=np.argmax(point_weight[:,i])
    data10[i]=cluster_mean[idx]
    
plt.imshow(data10.reshape(330,600,3))

fig=plt.figure(figsize=(32, 32))
columns =1 
rows = 10
for i in range(10):
    img = np.array(point_weight[i]).reshape(330,600)
    fig.add_subplot(10,1, i+1)
    plt.imshow(img)
plt.show()
