from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
import numpy as np

#batch gradient descent
X_train = [[2104,1600,2400,1416,3000]] #using living area as X for example
Y_train = [400,330,369,232,540] #using cost as Y for example
X_predict = [[1500,2500,4000,1200,6000]] #rando variables


theta=[[0,0]] #initialize it as the zero vector

#def h(i):
#    return theta*X_train[0][i]

#def J():
#    sum = 0
#    for i in range(len(X_train[0])):
#        sum += (h[i]-Y_train[i])**2

n_samples, n_features = 10, 5
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)
clf = SGDRegressor(max_iter=1000, tol=1e-3)
clf.fit(X, y)
print("coefficient: ",clf.coef_)
print("intercept: ",clf.fit_intercept)