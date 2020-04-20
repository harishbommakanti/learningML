from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
import numpy as np
import math

def TryToDoItMyself():
    #batch gradient descent
    X_train = np.array([[1,1,1,1,1],[2104,1600,2400,1416,3000], [3,3,3,2,4]]) #using living area as X_1, # bedrooms for X_2. X_0 is defined as the 1 vector
    Y_train = np.array([400,330,369]) #using cost as Y for example
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


    #Newton's Method: find where f'(theta) = 0
    # initialize theta_0 to 0: theta_t+1 = theta_t + l'(theta_x) / l''(theta_x)

    #l(theta) is the log of L(theta), the likelihood that the model works. so you need to maximize l(theta) to maximize h(theta)
    def l():
        returnVal=0
        for i in range(len(Y_train)):
            prediction = h(i)
            returnVal += Y_train[i]*math.log10(prediction) + (1-Y_train[i])*math.log10(prediction)

    newton_theta = np.zeroes(len(X_train[0])) #initialize all entries in theta vector to 0
    def lFirstDerivative(j): #first derivative of l is the sum from i=1 to m of (y^(i) - h(x^(i))) * x_j ^ (i)
        returnVal = 0
        for i in range(len(X_train)):
            intermediateSum = Y_train[i] - h(i)
            returnVal += X_train[j][i] * intermediateSum
        return returnVal

    def lSecondDerivative(): #too tired to figure this out haw
        return 8

    #now, need to maximize the log of the likelihood function using newton's method
    def newtonMethod():
        iterations = 12 #12 is usually good, every run through doubles the num of sig figs
        for i in range(1,iterations): #theta_0 is initialized as 0
            for j in range(len(newton_theta)-1):
                newton_theta[j+1] = newton_theta[j] - (lFirstDerivative(j) / lSecondDerivative())
        
        return newton_theta


#tryToDoItMyself()

#doing gradient descent in actual python/numpy, using stuff like np.dot
x = 2*np.random.rand(100,1) #100 points between 0 and 2, one single vector
y = 4 + 3*x + np.random.randn(100,1) #100 pts between 1 and 7, one single vector
#theta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y) #closed form for solving is (xTx)^-1 * xTy
def cost(theta,x,y):
    m=len(y)
    predictions = x.dot(theta)
    cost= (1/2*m) * np.sum(np.square(predictions-y))
    return cost

def gradientDescent(x,y,theta,learning_rate=0.01,iterations=100):
    m=len(y)
    cost_history=np.zeros(iterations)
    theta_history = np.zeros((iterations,2))

    for it in range(iterations): #doing it, iteration times
        prediction = np.dot(x,theta) #m1x1 + m2x2 ...

        theta = theta - (1/m)*learning_rate*(x.T.dot(prediction-y)) #x transpose dot (predictions - y), do for all theta
        theta_history[it,:] = theta.T
        cost_history[it] = cost(theta,x,y)
    
    return theta,cost_history,theta_history

def runGradientDescent():
    alpha = 0.01
    iterations = 1000
    theta = np.random.randn(2,1) #the example is just y = mx+b relationship, we have m_0 and m_1
    print("theta,",theta)

    x_b = np.c_[np.ones((100,1)),x] #make it a 2d vector
    theta,cost_history,theta_history = gradientDescent(x_b,y,theta,alpha,iterations)

    print('Theta0:          {:0.3f},\nTheta1:           {:0.3f}'.format(theta[0][0],theta[1][0]))
    print('final cost:      {:0.3f}'.format(cost_history[-1]))

    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    ax.set_ylabel('J(Theta)')
    ax.set_xlabel('Iterations')
    _=ax.plot(range(iterations),cost_history,'b.')
    plt.show()

def closedFormGradDescent():
    x_b = np.c_[np.ones((100,1)),x]
    theta =  np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
    print(theta[0][0],theta[1][0])

def plot_GD(n_iter,lr,ax,ax1=None):
     """
     n_iter = no of iterations
     lr = Learning Rate
     ax = Axis to plot the Gradient Descent
     ax1 = Axis to plot cost_history vs Iterations plot

     """
     _ = ax.plot(x,y,'b.')
     theta = np.random.randn(2,1)
     x_b = np.c_[np.ones((len(X),1)),X]

     tr =0.1
     cost_history = np.zeros(n_iter)
     for i in range(n_iter):
        pred_prev = x_b.dot(theta)
        theta,h,_ = gradientDescent(x_b,y,theta,lr,1)
        pred = x_b.dot(theta)

        cost_history[i] = h[0]

        if ((i % 25 == 0) ):
            _ = ax.plot(x,pred,'r-',alpha=tr)
            if tr < 0.8:
                tr = tr+0.2
     if not ax1== None:
        _ = ax1.plot(range(n_iter),cost_history,'b.')

runGradientDescent()
closedFormGradDescent()
#seem to work out for linear equations