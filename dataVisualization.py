import matplotlib.pyplot as plt
import numpy as np

randomNumGen = np.random

#simple distribution of like 500 rand vars between 1 and 1000
x = randomNumGen.randint(1000,size=500)
#plt.plot(x,'ro')
#plt.show()

#showing f(x) v x for simple functions
def f(x):
    #f(x) = x^2
    #return x**2

    #f(x) = bell curve
    sigma=1
    mu = 0
    return (1 / 2*np.pi*np.sqrt(sigma)) * np.exp(- (x - mu)**2 / 2*(sigma**2))

x = randomNumGen.uniform(-5,5,[1,5000])
plt.plot(x,f(x),'ro')
plt.show()