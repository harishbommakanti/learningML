from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
import numpy as np

#batch gradient descent
X_train = np.array([[1,1,1,1,1],[2104,1600,2400,1416,3000], [3,3,3,2,4]]) #using living area as X_1, # bedrooms for X_2. X_0 is defined as the 1 vector
Y_train = np.array([400,330,369,232,540]) #using cost as Y for example
X_predict = np.array([[1500,2500,4000,1200,6000],[1,2,3,4,5]]) #rando variables


theta=np.zeroes(3) #initialize it as the zero vector [0,0,0], x_0 (for the constant theta_0)

#h(x) = theta_0 * x_0 ^ (i) + ... + theta_n * x_n ^ (i), n is the number of features
def h(i):
    returnVal=0
    for j in range(0,len(theta)): #for every subarray, pick out x_train[j][i] and multiply by theta[i], add that to sum
        returnVal += theta[j] * X_train[j][i]
    return returnVal

#cost function: .5 * the sum of squares between the prediction and the actual value
def J():
    returnVal = 0
    for i in range(len(X_train[0])):
        returnVal += ( h(i) - Y_train[i]) ** 2
    return returnVal * .5

#goal is to minimize J()
# theta := theta + alpha * sum from 1 to n(y ^(i) - h(x^(i))) * x(i)
# remember, the 1/2 and derivative canceled out to produce x(i)
convergence=False
learningRate = 0
while(not convergence):
    newThetaJVal = 0
    for j in range(len(theta)):
        newThetaJVal + theta[j] #add the current value
        
        tempSum=0 #do the sum for the difference between true and predicted vals
        for i in range(0,5):
            tempSum += Y_train[i]-h[i]
        
        newThetaJVal + learningRate * tempSum * X_train[j][i] #apply learning rate * sum * simplified derivative

#using sklearn libraries

#n_samples, n_features = 10, 5
#rng = np.random.RandomState(0)
#y = rng.randn(n_samples)
#X = rng.randn(n_samples, n_features)
#clf = SGDRegressor(max_iter=1000, tol=1e-3)
#clf.fit(X, y)
#print("coefficient: ",clf.coef_)
#print("intercept: ",clf.fit_intercept)