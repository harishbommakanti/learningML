import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

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
#fig, axes = plt.subplots(2)
#x = np.linspace(0,2*np.pi,400)
#axes[0].plot(x,np.sin(x**2))
#axes[1].plot(x,bellCurve(x))
#fig.suptitle("vertically stacked plots")
#plt.show()

#plotting 4 functions in the same image
#fig, axes = plt.subplots(2,2)
#axes[0,0].plot(x,x)
#axes[1,1].plot(x,x**2)
#axes[0,1].plot(x,bellCurve(x))
#axes[1,0].plot(x**3,x)
#plt.show() 


# show newton's method to find where x_squared=0
x = np.linspace(-10,10,400)
#plt.plot(x,x**2)

def line(m,x,b):
    return m*x + b

def show_newtons_method():
    x_guess = random.uniform(-10,10)
    for i in range(4):
        #show dotted line between x_guess and the function
        plt.axvline(x=x_guess,ymin=-500,ymax=x_guess**2,c='g')
        
        f_x_guess = x_guess**2
        f_prime_x_guess = 2*x_guess #derivative of x^2 is 2x

        #draw line at that slope with that point
        #y = mx + b => b = y - mx = f_x_guess - f_prime_x_guess*x_guess
        plt.plot(x,line(f_prime_x_guess,x,f_x_guess - f_prime_x_guess*x_guess),'-r')

        #update x: new_x = x_guess - f(x_guess)/f'(x_guess)
        x_guess = x_guess - f_x_guess/f_prime_x_guess
    plt.show()

#show_newtons_method()


def company_sales_visualizations_simple_line():
    #read from csv and show some graphs based on that
    profit_list = company_data["total_profit"].tolist()

    #showing 1 simple line on the graph
    plt.plot(month_list,profit_list,label="Profit data of last year",color='red',marker='o',markerfacecolor='k',linestyle='--',linewidth='3')
    plt.xlabel('Month Number')
    plt.ylabel('Profit in dollars')
    plt.legend(loc='lower right')
    plt.xticks(month_list)
    plt.yticks([100000, 200000, 300000, 400000, 500000])
    #plt.show()

def company_sales_visualizations_all_data_lines():
    #show all data from the dataframe on the graph
    for i in range(1,len(company_data.columns)-2):
        col_label = company_data.columns[i]
        print(col_label)
        data = company_data[col_label]

        plt.plot(month_list,data.tolist(),label=col_label + " Sales Data",marker='o',linewidth='3')
    plt.xlabel("Month numbers")
    plt.ylabel("Sales units in numbers")
    plt.legend(loc='upper left')
    plt.title("Sales data")
    plt.xticks(np.linspace(1,12,12))
    plt.show()

def company_sales_visualizations_specifc_scatterplot(label):
    col_data = company_data[label]
    plt.scatter(month_list,col_data,color='blue',marker='o')
    plt.grid(True,linewidth=1,linestyle='--')
    plt.show()

def company_sales_visualizations_compare_bar_charts(prod1,prod2):
    prod1_col = company_data[prod1]
    prod2_col = company_data[prod2]

    plt.bar([a-.25 for a in month_list],prod1_col,width=.25,label=prod1 + " sales data", align='edge')
    plt.bar([a+.25 for a in month_list],prod2_col,width=-.25,label=prod2 + " sales data", align='edge')
    plt.grid(True,linewidth=1,linestyle='--')
    plt.show()

def company_sales_visualizations_histogram():
    profit_data = company_data['total_profit']
    plt.hist(profit_data)
    plt.show()
      
company_data = pd.read_csv("company_sales_data.csv")
month_list = company_data["month_number"].tolist()
#company_sales_visualizations_simple_line()
#company_sales_visualizations_all_data_lines()
#company_sales_visualizations_specifc_scatterplot("toothpaste")
#company_sales_visualizations_compare_bar_charts("facecream","facewash")
company_sales_visualizations_histogram()