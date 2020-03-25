from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import numpy as np

n_samples, n_features = 20, 5
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
#print("coefficient: ",clf.predict(x_predict))
#print("intercept: ",clf.fit_intercept)


# use logistic regression
y = np.random.randint(5, size=10)
x = np.random.randint(50, size=(10,5))
clf = LogisticRegression(penalty='l2',max_iter=4000)
clf.fit(x,y)

x_predict = np.random.randint(50, size=(10,5))
#print("coefficient: ",clf.predict(x_predict))
#print("score: ",clf.score)

# logistic regression is also capable of handling multinomial (multi class) classification problems
clf = LogisticRegression(penalty='l2',multi_class='multinomial',max_iter=4000)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.33,random_state=42)
clf.fit(x_train,y_train)
#print("test results",clf.predict(x_test))
#print("y",y_test)


#gaussian discriminant analysis: a Generative learning algorithm
#which assumes p(x|y) is gaussian alongside bernoulli random variables
#(implying p(y|x) is logistic). advantage is that it uses less data, but logistic is more robust

GDA = LinearDiscriminantAnalysis()
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42)
GDA.fit(x,y)
#print('test results: ',GDA.predict(x_test))
#print("actual: ",y_test)


#Naive Bayes implementations, which use the assumption that (by the chain rule of probability,
#P(x_2 | y,x_1)) = P(x_2 | y): that is, knowing other variables should not determine P(x_n | y)
#it is a 'naive' assumption, but its mostly accurate for things like text classification
from sklearn.naive_bayes import GaussianNB
gaussNBClassifier = GaussianNB()
gaussNBClassifier.fit(x_train,y_train)
#print("test results: ",gaussNBClassifier.predict(x_test))
#print("actual: ",y_test)  #gaussNB isn't *too* accurate, random nums may not follow the distribution needed aka be gaussian haw

from sklearn.naive_bayes import BernoulliNB
bernNB = BernoulliNB(binarize=True)
bernNB.fit(x_train,y_train)
#print("test results: ",bernNB.predict(x_test))
#print("actual: ",y_test)  #not accurate at all, random data probably isn't meant for bernoulli distributions

from sklearn.naive_bayes import MultinomialNB
multiNB = MultinomialNB()
multiNB.fit(x_train,y_train)
#print("test results: ",multiNB.predict(x_test))
#print("actual: ",y_test)   # results are again, trash, probably need actual labeled data