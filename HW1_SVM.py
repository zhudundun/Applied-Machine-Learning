import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

train=pd.read_csv("trainAdult.csv")
test=pd.read_csv("testAdult.csv")
trainX=train.select_dtypes(include='int64')

# data preprocess
def normalize(data):
    data = data - np.mean(data, axis=0)
    data = data / np.std(data, axis=0)
    return data
    
    
s = train['NA']
labels = []
#create dummy for response variable
for x in s:
    if x == s[0]:
        labels.append(-1)
    else:
        labels.append(1)
        
        
labels = np.array(labels)
continuous = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
train = train[continuous].values
test = test[continuous].values
train = normalize(train)
test = normalize(test)


# calculate accuracy
def cal_accuracy(predictions, yi):
    correct = 0
    for pred, actual in zip(predictions, yi):
        if pred == actual:
            correct += 1
    return correct / len(yi)
    
    
# update and gradient,eta is learning rate,lamb is lambda
# this function calculates the gradient for SVM
def gradient(a, b, eta, lamb, x, yi):
    res = yi * (np.dot(a, x.T) + b)
    if res >= 1:
        temp = (eta * lamb) * a
        a -= temp
    else:
        temp = -yi * x+lamb * a
        a -= eta * temp
        b -= eta * (-yi)
    return a, b
    
    
# predict class
#if ax+b positive,predict to one class, else to another class
# think about 2D case
def sign(test_x, a, b):
    result = np.dot(test_x, a.T) + b
    result[result <= 0] = -1
    result[result > 0] = 1
    return result
    
    
# train test split
# size determines the ratio of train vs test set
def split(data, size):
    rows = data.shape[0]
    index = np.arange(0, rows)
    np.random.shuffle(index)
    test_id = index[:size]
    train_id = index[size:]
    return test_id, train_id
    
    
# choose learning rate by grid search
# trying out learning rate on a small set of data
def small_experiment():
    learning_rate = [0.01, 0.02, 0.05, 0.07, 0.1, 0.2, 0.5]

    # use 1000 obs to train, 100 to test
    test_index, train_index = split(train, 100)
    train_index = train_index[:1000]
    small_train = train[train_index]
    small_y = labels[train_index]
    small_test = train[test_index]
    small_target = labels[test_index]

    accuracy = []
    time = []
    # train and record accuracy and time
    for lr in learning_rate:
        start = datetime.datetime.now()
        # since there're 6 continuous variables included, a is a vector with 6 numbers
        a = np.random.rand(6)
        b = np.random.rand(1)[0]
        for i in range(5000):
            #pick one number from 1-1000 randomly, this is the single sample to run SGD
            rand_ind = np.random.choice(range(1000), 1)[0]
            xi = small_train[rand_ind]
            yi = small_y[rand_ind]
            a, b = gradient(a, b, lr, 0.01, xi, yi)
        end = datetime.datetime.now()
        # add the average error calculated from this particular a,b pair
        accuracy.append(cal_accuracy(sign(small_test, a, b), small_target))
        time.append((end-start).total_seconds())

    # plot accuracy and time
    plt.scatter(accuracy, time)
    for i, txt in enumerate(learning_rate):
        plt.annotate(txt, (accuracy[i], time[i]))
    plt.show()
    
    
# grid search lambda
lambda_vals = [1e-3, 1e-2, 1e-1, 5e-1, 1]
def search(eta):
    best_lambda = 0
    best_accuracy = 0
    for l in lambda_vals:
        avg_acc = 0
        for round in range(50):
            # train validation split,0.1 is the train test split ratio
            val_index, season_index = split(train, int(0.1 * len(train)))
            search_train = train[season_index]
            search_yi = labels[season_index]
            validation = train[val_index]
            val_label = labels[val_index]

            # train model
            a = np.random.rand(6)
            b = np.random.rand(1)[0]
            for i in range(5000):
                rand_ind = np.random.choice(range(len(search_train)), 1)[0]
                xi = search_train[rand_ind]
                yi = search_yi[rand_ind]
                a, b = gradient(a, b, eta, l, xi, yi)
            avg_acc += cal_accuracy(sign(validation, a, b), val_label)

        # find the max accuracy
        if avg_acc > best_accuracy:
            best_accuracy = avg_acc
            best_lambda = l
    return best_lambda
    
    
# parameters
epochs = 50
held_out = 50
season_steps = 300
acc_steps = 30
m = 1
n = 10


# SVM training
acc_30_steps = []
magnitude = []
def svm(lamb):
    a = np.random.rand(6)
    b = np.random.rand(1)[0]
    for i in range(epochs):
        print("Epochs ", i)

        # update eta, learning rate for each season
        eta = m / (i + n)

        # held out 50 evaluation examples
        eval_index, train_index = split(train, held_out)
        new_train = train[train_index]
        new_yi = labels[train_index]
        new_test = train[eval_index]
        new_label = labels[eval_index]

        # season and step are to ensure each data has been seen in the random picking method
        for j in range(season_steps):
            # gradient update parameter a and b
            rand_ind = np.random.choice(range(len(new_train)), 1)[0]
            xi = new_train[rand_ind]
            yi = new_yi[rand_ind]
            a, b = gradient(a, b, eta, lamb, xi, yi)

            # calculate accuracy every 30 steps
            if (j % acc_steps == 0) & (j > 0):
                predictions = sign(new_test, a, b)
                season_acc = cal_accuracy(predictions, new_label)
                acc_30_steps.append(season_acc)
                magnitude.append(np.linalg.norm(a))
                print("in step ", j, " season accuracy is ", season_acc)

        predictions = sign(new_test, a, b)
        epoch_acc = cal_accuracy(predictions, new_label)
        acc_30_steps.append(epoch_acc)
        magnitude.append(np.linalg.norm(a))
        print("Epochs accuracy is ", epoch_acc)
    return a, b
    
    
# train svm
plot_acc = {}
plot_mag = {}
for l in lambda_vals:
    acc_30_steps = []
    magnitude = []
    a, b = svm(l)
    predictions = sign(test, a, b)
    plot_acc["reg="+str(l)] = acc_30_steps
    plot_mag["reg="+str(l)] = magnitude
plot_mag['x'] = range(len(acc_30_steps))
plot_acc['x'] = range(len(magnitude))


#compare response in test set with predictions
test_50k=[]
for label in test.NA:
    if label == ' <=50K.':
        test_50k.append(-1)
    else:
        test_50k.append(1)
        
        
cal_accuracy(predictions, test_50k)
