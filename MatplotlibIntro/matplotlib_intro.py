# matplotlib_intro.py
"""Python Essentials: Intro to Matplotlib.
<Josh Moak>
<Math 345>
<September 14>
"""

import numpy as np
from matplotlib import pyplot as plt
data = np.load("FARS.npy")


# Problem 1
def var_of_means(n):
    """Construct a random matrix A with values drawn from the standard normal
    distribution. Calculate the mean value of each row, then calculate the
    variance of these means. Return the variance.

    Parameters:
        n (int): The number of rows and columns in the matrix A.

    Returns:
        (float) The variance of the means of each row.
    """
    
    A = np.random.normal(size=(n,n))
    mean = np.mean(A, axis=1)
    variance = np.var(mean)
    return variance

    
    #raise NotImplementedError("Problem 1 Incomplete")

def prob1():
    """Create an array of the results of var_of_means() with inputs
    n = 100, 200, ..., 1000. Plot and show the resulting array.
    """

    y = np.zeros(10)
    for i in range(10):
        y[i] = var_of_means((i+1)*100)
    plt.plot(np.arange(1,11)*100, y)
    plt.show()

    #aise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def prob2():
    """Plot the functions sin(x), cos(x), and arctan(x) on the domain
    [-2pi, 2pi]. Make sure the domain is refined enough to produce a figure
    with good resolution.
    """
    
    x = np.linspace(-2*np.pi, 2*np.pi, 100)
    plt.plot(x, np.sin(x), label = "sin")
    plt.plot(x, np.cos(x), label = "cos")
    plt.plot(x, np.arctan(x), label = "arctan")
    plt.legend(loc = "upper left")
    plt.show()

    #raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def prob3():
    """Plot the curve f(x) = 1/(x-1) on the domain [-2,6].
        1. Split the domain so that the curve looks discontinuous.
        2. Plot both curves with a thick, dashed magenta line.
        3. Set the range of the x-axis to [-2,6] and the range of the
           y-axis to [-6,6].
    """

    x_1 = np.linspace(-2, 0.999999999, 50)
    x_2 = np.linspace(1.000000001, 6, 50)
    plt.plot(x_1, 1/(x_1-1), "m--", lw = 4)
    plt.plot(x_2, 1/(x_2-1), "m--", lw = 4)
    plt.xlim(-2,6)
    plt.ylim(-6,6)
    plt.show()

    #raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def prob4():
    """Plot the functions sin(x), sin(2x), 2sin(x), and 2sin(2x) on the
    domain [0, 2pi].
        1. Arrange the plots in a square grid of four subplots.
        2. Set the limits of each subplot to [0, 2pi]x[-2, 2].
        3. Give each subplot an appropriate title.
        4. Give the overall figure a title.
        5. Use the following line colors and styles.
              sin(x): green solid line.
             sin(2x): red dashed line.
             2sin(x): blue dashed line.
            2sin(2x): magenta dotted line.
    """
    x = np.linspace(0, 2*np.pi, 100)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    plt.rc('axes', titlesize=8)
    plt.axis([0, 2*np.pi, -2, 2])
    ax1.set_title("sin(x)")
    ax2.set_title("sin(2x)")
    ax3.set_title("2sin(x)")
    ax4.set_title("2sin(2x)")
    fig.suptitle("Various sin curves")
    ax1.plot(x, np.sin(x), "g-")
    ax2.plot(x, np.sin(2*x), "r--")
    ax3.plot(x, 2*np.sin(x), "b--")
    ax4.plot(x, 2*np.sin(2*x), "m:")
    plt.show()

    #raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def prob5():
    """Visualize the data in FARS.npy. Use np.load() to load the data, then
    create a single figure with two subplots:
        1. A scatter plot of longitudes against latitudes. Because of the
            large number of data points, use black pixel markers (use "k,"
            as the third argument to plt.plot()). Label both axes.
        2. A histogram of the hours of the day, with one bin per hour.
            Label and set the limits of the x-axis.
    """

    x = data[:,0] # hours of day
    longitudes = data[:,1] #second column, all rows
    latitudes = data[:,2] # third column, all rows
    ax1 = plt.subplot(121)
    ax1.set_xlabel("Longitudes")
    ax1.set_ylabel("Latitudes")
    ax1.set_aspect("equal")
    ax1.plot(longitudes, latitudes, ".k", markersize = .5)
    ax2 = plt.subplot(122)
    ax2.set_xlabel("Hours of day")
    ax2.hist(x, bins = 24, range = [0, 23] )
    plt.show()

    #raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def prob6():
    """Plot the function f(x,y) = sin(x)sin(y)/xy on the domain
    [-2pi, 2pi]x[-2pi, 2pi].
        1. Create 2 subplots: one with a heat map of f, and one with a contour
            map of f. Choose an appropriate number of level curves, or specify
            the curves yourself.
        2. Set the limits of each subplot to [-2pi, 2pi]x[-2pi, 2pi].
        3. Choose a non-default color scheme.
        4. Add a colorbar to each subplot.
    """

    x = np.linspace(-2*np.pi, 2*np.pi, 100)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    Z = (np.sin(X)*np.sin(Y))/(X*Y)

    plt.subplot(121)
    plt.pcolormesh(X, Y, Z, cmap = "coolwarm")
    plt.colorbar()
    plt.xlim(-2*np.pi, 2*np.pi)
    plt.ylim(-2*np.pi, 2*np.pi)
    
    plt.subplot(122)
    plt.contour(X, Y, Z, 10, cmap = "coolwarm")
    plt.colorbar()

    plt.show()

    #raise NotImplementedError("Problem 6 Incomplete")
