import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

math_df=pd.read_csv("student-mat.csv",sep=';')

del math_df['G1']
del math_df['G2']
del math_df['absences']

# create a categorical variable based on G3
math_df.loc[math_df['G3'] > 12, 'G4']='1' 
math_df.loc[math_df['G3'] <= 12, 'G4']='0' 

del math_df['G3']

def prior_probability(df):
    #prior_probs has the probability for each level in all multinomial and binomial variables
    prior_probs = dict()
    for column in df.columns:
        vec=list(df[column].value_counts().index)
        for i in vec:
            prior_probs[column,i] = (df[column].value_counts()[i] + 0.1) / (df.shape[0] + 0.1 * len(vec))
    return prior_probs
    
    
#count gives the count of (X=x,Y=y)
def count(data,colname,label,target):
    condition = (data[colname] == label) & (data['G4'] == target)
    return len(data[condition])
    
   
#columns[:-1] excludes y
#condition_prob has the probability of each level for binomial and multinomial columns given y is 0/1
def condition_probability(df,response):
    G4=df.loc[df['G4']==response,].shape[0]
    condition_prob = dict()
    #columns[:-1] excludes y
    #condition_prob0 has the probability of each level for binomial and multinomial columns given y is 0
    for column in df.columns[:-1]:
        vec=list(df[column].value_counts().index)
        for i in vec:
            condition_prob[column,i] = (count(df,column,i,response) + 0.1) / (G4 + 0.1 * len(vec))
    return condition_prob
    
    
#pGaussian gives the probability of a data given mean and variance
def pGaussian(x, mean, variance):

    # Input the arguments into a probability density function
    p = 1/(np.sqrt(2*np.pi*variance)) * np.exp((-(x-mean)**2)/(2*variance))
    
    # return p
    return p
    
    
    
#columns[:-1] exclues y
#prob_num has the probability of y is 0/1 for numeric part of each observation in test set
#prob has the probability of y is 0/1 for binary and multinomial part of each observation in test set
# response is the value of y
def probability(response):
    prob_list=list()

    for i in range(test_num.shape[0]):
        prob=1
        prob_num=1
        for column in test_cate.columns[:-1]:
            val=test_cate.loc[i,column]
            prob=prob*condition_prob[response][column,val]
        for column in test_num.columns[:-1]:

            val_num=test_num.loc[i,column]
            prob_num=prob_num*pGaussian(val_num,data_means.loc[response,column],data_variance.loc[response,column])
        prob_all=prob*prob_num*prior_probs['G4',response]
        prob_list.append(prob_all)
    return prob_list
    
    
    
#columns[:-1] exclues y
#prob_num has the probability of y is 0/1 for numeric part of each observation in test set
#prob has the probability of y is 0/1 for binary and multinomial part of each observation in test set
# response is the value of y
def probability2(response):
    prob_list=list()

    for i in range(test.shape[0]):
        prob=1
        for column in test.columns[:-1]:
            val=test.loc[i,column]
            # if the observation only exist in test set, means the probability of this observation in train is very small, use a small number to represent
            if (column,val) not in condition_prob['0'].keys():
                prob=prob*0.001
            else:
                prob=prob*condition_prob[response][column,val]

        prob=prob*prior_probs['G4',response]
        prob_list.append(prob)
    return prob_list
    
    
    
accuracy_list=[]
for i in range(10):
    
    # get train and test set
    train, test = train_test_split(math_df, test_size=0.15)

    #train/test_num contains only numeric variables
    train_num=train.select_dtypes(include='int64')
    test_num=test.select_dtypes(include='int64')

    #math_cate contains only category variables
    train_cate=train.select_dtypes(include='object')
    test_cate=test.select_dtypes(include='object')

    #Add the response back to numeric data
    train_num['G4']=train_cate['G4']
    test_num['G4']=test_cate['G4']

    # reset indexes
    train_cate=train_cate.reset_index(drop=True)
    test_cate=test_cate.reset_index(drop=True)
    train_num=train_num.reset_index(drop=True)
    test_num=test_num.reset_index(drop=True)

    prior_probs=prior_probability(train_cate)
    condition_prob0=condition_probability(train_cate,'0')
    condition_prob1=condition_probability(train_cate,'1')

    #combine conditional probability when y is 0 and 1
    condition_prob = dict()

    condition_prob['0']=condition_prob0
    condition_prob['1']=condition_prob1


    #data_mean has the mean for y is 0 and y is 1 group
    #data_variance has the variance for y is 0 and y is 1 group
    data_means=test_num.groupby('G4').mean()
    data_variance = test_num.groupby('G4').var()


    prob0=probability('0')
    prob1=probability('1')

    pred=[]
    for (i,j) in zip(prob0,prob1):
        if i>j:
            pred.append('0')
        else:
            pred.append('1')

    correct=0
    accuracy=0
    for (i,j) in zip(pred,test['G4']):
        if i==j:
            correct+=1
    accuracy=correct/len(pred)
    accuracy_list.append(accuracy)
    
#from statistics import mean,stdev

mean(accuracy_list)
stdev(accuracy_list)


accuracy_list2=[]
for i in range(10):
    
    # get train and test set
    train, test = train_test_split(math_df, test_size=0.15)

    # reset indexes
    train=train.reset_index(drop=True)
    test=test.reset_index(drop=True)
   
    prior_probs=prior_probability(train)
    condition_prob0=condition_probability(train,'0')
    condition_prob1=condition_probability(train,'1')

    #combine conditional probability when y is 0 and 1
    condition_prob = dict()

    condition_prob['0']=condition_prob0
    condition_prob['1']=condition_prob1  

    prob0=probability2('0')
    prob1=probability2('1')

    pred=[]
    for (i,j) in zip(prob0,prob1):
        if i>j:
            pred.append('0')
        else:
            pred.append('1')

    correct=0
    accuracy=0
    for (i,j) in zip(pred,test['G4']):
        if i==j:
            correct+=1
    accuracy=correct/len(pred)
    accuracy_list2.append(accuracy)
