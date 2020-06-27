import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

iris = pd.read_csv('iris.csv')
df=iris[['setal_length', 'setal_width', 'petal_length', 'petal-width']]

#transform and inverse trandform
#final version 
sd_list=[0.1,0.2,0.5,1]
dfTemp=[None]*4
MSE_list=[None]*4
for j in range(4):
    dfTemp[j]=df+np.random.normal(0.0,sd_list[j],[150,4])
    MSE_list[j]=[None]*4
    for i in range(4):
        noisy_model = PCA(n_components=i+1)
        noisy_model.fit(dfTemp[j])
        pca_data = noisy_model.transform(dfTemp[j])
        pca_data = noisy_model.inverse_transform(pca_data)
        MSE_list[j][i]=mean_squared_error(df, pca_data)
        
        
#transform and inverse trandform
w_list=[1,2,3,4]
dfTemp=[None]*4
MSE_list2=[None]*4
for j in range(4):
    dfTemp[j]=df*np.random.binomial(1, 1-w_list[j]/60, [150,4])
    MSE_list2[j]=[None]*4
    for i in range(4):
        
        # use eigen decomposition to get the eigen values directly
        #scaler = StandardScaler()

        #dfScale=pd.DataFrame(scaler.fit_transform(dfTemp))

        noisy_model = PCA(n_components=i+1)
        noisy_model.fit(dfTemp[j])
        pca_data = noisy_model.transform(dfTemp[j])
        pca_data = noisy_model.inverse_transform(pca_data)
        MSE_list2[j][i]=mean_squared_error(df, pca_data)
        
        
        
x = [1,2,3,4]
for i in range(4):
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(x, MSE_list[i])
    plt.title("SD {}".format(sd_list[i]))
    plt.xlabel("PCA")
    plt.ylabel("MSE")
    for i,j in zip(x,MSE_list[i]):
        ax.annotate(str(j),xy=(i,j))
        
        
        
x = [1,2,3,4]
for i in range(4):
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(x, MSE_list2[i])
    plt.title("Weight {}".format(w_list[i]))
    plt.xlabel("PCA")
    plt.ylabel("MSE")
    for i,j in zip(x,MSE_list2[i]):
        ax.annotate(str(j),xy=(i,j))
