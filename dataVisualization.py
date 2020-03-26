import matplotlib.pyplot as plt
import numpy as np

randomNumGen = np.random

#simple distribution of like 500 rand vars between 1 and 1000
x = randomNumGen.randint(1000,size=500)
#plt.plot(x,'ro')
#plt.show()

#showing f(x) v x for simple functions
def xSquared(x):
    return x**2

def bellCurve(x):
    sigma=1
    mu = 0
    return (1 / 2*np.pi*np.sqrt(sigma)) * np.exp(- (x - mu)**2 / 2*(sigma**2))

x = randomNumGen.uniform(-5,5,[1,5000])
#plt.plot(x,xSquared(x),'ro')
#plt.show()


#plotting 2 functions on the same figure
fig, axes = plt.subplots(2)
x = np.linspace(0,2*np.pi,400)
axes[0].plot(x,np.sin(x**2))
axes[1].plot(x,bellCurve(x))
fig.suptitle("vertically stacked plots")
#plt.show()

#plotting 4 functions in the same image
fig, axes = plt.subplots(2,2)
axes[0,0].plot(x,x)
axes[1,1].plot(x,x**2)
axes[0,1].plot(x,bellCurve(x))
axes[1,0].plot(x**3,x)
#plt.show() 