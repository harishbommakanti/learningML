from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

n_samples, n_features = 10, 5
rng = np.random.RandomState(0) #something to give random numbers
y = rng.randn(n_samples) #an array with 10 random samples between 0 and 1
x = rng.randn(n_samples, n_features) #a 2D array with 10 subarrays (features) with 5 rand floats each
x_predict = rng.randn(n_samples,n_features)

#regular linear regression
clf = LinearRegression(fit_intercept=True)
clf.fit(x,y)
clf.predict(x_predict)
#print(clf.coef_)


# using stochastic gradient descent to fit the data
clf = SGDRegressor(max_iter=1000, tol=1e-3)
clf.fit(x, y)
clf.predict(x_predict)
#print("coefficient: ",clf.coef_)
#print("intercept: ",clf.fit_intercept)


# use logistic regression
y = np.random.randint(5, size=10)
x = np.random.randint(50, size=(10,5))
clf = LogisticRegression(penalty='l2',max_iter=4000)
clf.fit(x,y)

x_predict = np.random.randint(50, size=(10,5))
clf.predict(x_predict)
#print("coefficient: ",clf.coef_)
#print("score: ",clf.score)

# logistic regression is also capable of handling multinomial (multi class) classification problems
clf = LogisticRegression(penalty='l2',multi_class='multinomial',max_iter=4000)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.33,random_state=42)
clf.fit(x_train,y_train)
clf.predict(x_test)
print("x",x)
print("y",y)
print("test results",clf.coef_)
print("y",y_test)