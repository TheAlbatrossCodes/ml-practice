# -*- coding: utf-8 -*-
"""

@author: The Albatross
"""
#We have 3 Classes, each with 500 data points
#   1st class -> centered around (-2, 2)
#   2nd class -> centered around (2, -2)
#   3rd class -> centered around (2, 4)



import numpy as np
import matplotlib.pyplot as plt

#calculate classification rate
def classification_rate(Y, P):
    #Y is the real labels, P is our predicted label
    n_total = 0
    n_correct = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    #could also just do np.mean(Y==P)
    return float(n_correct) / n_total

#define cost function:
def cost_func(T, Y):
    tot  = T * np.log(Y)
    #sum it up, since its a vector containing cost for EACH data point
    return tot.sum()


#define feedforward function, this does our predictions
def forward(X, W0, b0, W1, b1):
    #function at layer M, before applying activation function
    Am = X.dot(W0) + b0
    Z = np.tanh(Am) #applying activation function
    
    Ak = Z.dot(W1) + b1 #function at output layer, before applying activ function
    
    #calculating softmax
    exp_Ak = np.exp(Ak) 
    P = exp_Ak / np.sum(exp_Ak, axis=1, keepdims=True)
    return Z, P


def deriv_w1(Z, T, Y): #deriv wrt W1 = hidden to output weights
    return Z.T.dot(T - Y)


def deriv_b1(T, Y): #derive wrt b1 = output bias
    return (T - Y).sum(axis=0)

def deriv_w0(X, T, Y, W1, Z): #deriv wrt to W0 = input to hidden weights
    dZ = (T - Y).dot(W1.T) * (1 - Z**2)
    return X.T.dot(dZ)


def deriv_b0(T, Y, W1, Z): #derv wrt b0 = bias at hidden units
    return (T - Y).dot(W1.T) * (1 - Z**2).sum(axis=0)


#We'll have 3 classes
#each with 500 datapoints
Nclass = 500 #No. data points in each class
    
#create fake data
X1 = np.random.randn(Nclass, 2) + np.array([-2, 2])
X2 = np.random.randn(Nclass, 2) + np.array([2, -2]) 
X3 = np.random.randn(Nclass, 2) + np.array([2, 4])

#final X
X = np.concatenate((X1, X2, X3))

#create targets
Y = np.array([0] * Nclass + [1] * Nclass + [2] * Nclass)

#visualize the data
# plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
# plt.show()
    
N = len(X) #number of data points
D = 2 #input units
M = 10 #hidden units
K = 3 #output units, no. of classes

#one hot encode the targets Y to indicator matrix T  
T = np.zeros((N,K))
    
for i in range(N):
    T[i, Y[i]] = 1
    
    
#initialize the weights
W0 = np.random.randn(D, M) #input to hidden weight
b0 = np.zeros(M) #hidden bias
W1 = np.random.randn(M,K) #output to hidden weight
b1 = np.zeros(K) #output bias
    
learn_rate = 0.000001
costs = [] #keep track of costs

#training the model 
for epoch in range(5000):
    Z, P_Y_given_X = forward(X, W0, b0, W1, b1) #P_Y_given_X is a NxK matrix, telling us of the prob of a data point belonging to each class
        
    if epoch % 100 == 0:
        our_cost = cost_func(T, P_Y_given_X)
        P = np.argmax(P_Y_given_X, axis=1) #our predicted label (this aint prob, its a Nx1 vector)
        rate = classification_rate(Y, P)
        print("epoch: {}, cost: {}, class. rate: {}".format(epoch, our_cost, rate))
        costs.append(our_cost)
    
    #learning_rate times the gradient of weight/bias
    dW1 = learn_rate * deriv_w1(Z, T, P_Y_given_X)
    db1 = learn_rate * deriv_b1(T, P_Y_given_X)
    dW0 = learn_rate * deriv_w0(X, T, P_Y_given_X, W1, Z)
    db0 = learn_rate * deriv_b0(T, P_Y_given_X, W1, Z)
    
    #doing gradient ascent
    W1 = W1 + dW1
    b1 = b1 + db1
        
    W0 = W0 + dW0
    b0 = b0 + db0

        
plt.plot(costs)
plt.show()
    

print('class. rate is: {}'.format(classification_rate(Y, P)))

