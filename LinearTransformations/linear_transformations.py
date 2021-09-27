# linear_transformations.py
"""Volume 1: Linear Transformations.
<Josh Moak>
<Math 345>
<Sep 21, 2021>
"""

from random import random
import numpy as np
from matplotlib import pyplot as plt
data = np.load("horse.npy")
import time


# Problem 1
def stretch(A, a, b):
    """Scale the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    stretchy = np.array([[a,0],[0,b]])
    result_stretch = stretchy@A
    return result_stretch

    #raise NotImplementedError("Problem 1 Incomplete")

def shear(A, a, b):
    """Slant the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    sheary = np.array([[1,a],[b,1]])
    result_shear = sheary@A
    return result_shear

    #raise NotImplementedError("Problem 1 Incomplete")

def reflect(A, a, b):
    """Reflect the points in A about the line that passes through the origin
    and the point (a,b).

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): x-coordinate of a point on the reflecting line.
        b (float): y-coordinate of the same point on the reflecting line.
    """
    reflecty = (1/((a**2)+(b**2)))*np.array([[(a**2)-(b**2),2*a*b],[2*a*b,(b**2)-(a**2)]])
    result_reflect = reflecty@A
    return result_reflect

    #raise NotImplementedError("Problem 1 Incomplete")

def rotate(A, theta):
    """Rotate the points in A about the origin by theta radians.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        theta (float): The rotation angle in radians.
    """
    rotatey = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    result_rotate = rotatey@A
    return result_rotate

    #raise NotImplementedError("Problem 1 Incomplete")
#test function for problem 1
"""
def test_transformations():
    stretched = stretch(data, 1/2, 6/5)
    sheared = shear(data, 1/2, 0)
    reflected = reflect(data, 0, 1)
    rotated = rotate(data, 0.5*np.pi)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.plot(stretched[0], stretched[1], 'k,')
    ax2.plot(sheared[0], sheared[1], 'k,')
    ax3.plot(reflected[0], reflected[1], 'k,')
    ax4.plot(rotated[0], rotated[1], 'k,')
    plt.axis([-1,1,-1,1])
    plt.gca().set_aspect("equal")
    plt.show()"""





# Problem 2
def solar_system(T, x_e, x_m, omega_e, omega_m):
    """Plot the trajectories of the earth and moon over the time interval [0,T]
    assuming the initial position of the earth is (x_e,0) and the initial
    position of the moon is (x_m,0).

    Parameters:
        T (float): The final time.
        x_e (float): The earth's initial x coordinate.
        x_m (float): The moon's initial x coordinate.
        omega_e (float): The earth's angular velocity.
        omega_m (float): The moon's angular velocity.
    """
    
    time = np.linspace(0,T,300) #Time vector
    p_e_0 = np.array([x_e, 0]) #initial conditions
    p_m_0 = np.array([x_m, 0])
    p_e_x = [] #Empty lists that will contain x and y values for 
    p_e_y = [] #moon and earth
    p_m_x = []
    p_m_y = []
    for i in time: #Loop through every time value in time vector
        p_e = rotate(p_e_0, i*omega_e) #perform rotation for earth
        relative_p = p_m_0 - p_e_0 #Moon relative to earth
        moon_orient = rotate(relative_p, i*omega_m) #Rotation
        p_m = moon_orient + p_e #translation
        p_e_x.append(p_e[0]) #Populate empty lists with x coordinates
        p_e_y.append(p_e[1]) #Populate empty lists with y coordinates
        p_m_x.append(p_m[0])
        p_m_y.append(p_m[1])
    ax = plt.subplot(111)
    plt.plot(p_e_x, p_e_y, label = "Earth")
    plt.plot(p_m_x, p_m_y, label = "Moon")
    plt.legend(loc = "lower right")
    ax.set_aspect("equal")
    plt.show()

    #raise NotImplementedError("Problem 2 Incomplete")


def random_vector(n):
    """Generate a random vector of length n as a list."""
    return [random() for i in range(n)]

def random_matrix(n):
    """Generate a random nxn matrix as a list of lists."""
    return [[random() for j in range(n)] for i in range(n)]

def matrix_vector_product(A, x):
    """Compute the matrix-vector product Ax as a list."""
    m, n = len(A), len(x)
    return [sum([A[i][k] * x[k] for k in range(n)]) for i in range(m)]

def matrix_matrix_product(A, B):
    """Compute the matrix-matrix product AB as a list of lists."""
    m, n, p = len(A), len(B), len(B[0])
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
                                    for j in range(p) ]
                                    for i in range(m) ]

# Problem 3
def prob3():
    """Use time.time(), timeit.timeit(), or %timeit to time
    matrix_vector_product() and matrix-matrix-mult() with increasingly large
    inputs. Generate the inputs A, x, and B with random_matrix() and
    random_vector() (so each input will be nxn or nx1).
    Only time the multiplication functions, not the generating functions.

    Report your findings in a single figure with two subplots: one with matrix-
    vector times, and one with matrix-matrix times. Choose a domain for n so
    that your figure accurately describes the growth, but avoid values of n
    that lead to execution times of more than 1 minute.
    """

    n_vals = [2**n for n in range(9)]
    mat_vec_time = []
    mat_mat_time = []
    for n in n_vals:
        x = random_vector(n)
        A = random_matrix(n)
        B = random_matrix(n)
    
        start_1 = time.time() #time starts
        matrix_vector_product(A,x) #mat-vec operation
        mat_vec_time.append(time.time() - start_1) #time ends

        start_2 = time.time() #time start
        matrix_matrix_product(A,B) #mat-mat operation
        mat_mat_time.append(time.time() - start_2) #time end

    ax1 = plt.subplot(121) #plot times against n-values
    ax1.plot(n_vals, mat_vec_time)
    ax1.scatter(n_vals, mat_vec_time)
    ax1.set_xlabel("n") 
    ax1.set_ylabel("Seconds")
    ax1.set_title("Matrix-Vector Multiplication")

    ax2 = plt.subplot(122) #more plotting for other operation
    ax2.plot(n_vals, mat_mat_time)
    ax2.scatter(n_vals, mat_mat_time)
    ax2.set_xlabel("n")
    ax2.set_title("Matrix-Matrix Multiplication")

    plt.show()

    #raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def prob4():
    """Time matrix_vector_product(), matrix_matrix_product(), and np.dot().

    Report your findings in a single figure with two subplots: one with all
    four sets of execution times on a regular linear scale, and one with all
    four sets of exections times on a log-log scale.
    """

    n_vals = [2**n for n in range(9)]
    mat_vec_time = []
    mat_mat_time = []
    mat_vec_dot_time = []
    mat_mat_dot_time = []
    for n in n_vals:
        x = random_vector(n)
        A = random_matrix(n)
        B = random_matrix(n)
    
        start_1 = time.time() #starting time
        matrix_vector_product(A,x) #same as prob 3
        mat_vec_time.append(time.time() - start_1) #ending time

        start_2 = time.time()
        matrix_matrix_product(A,B)
        mat_mat_time.append(time.time() - start_2)

        start_3 = time.time()
        np.dot(A,x) #similar but using numpy's multiplication
        mat_vec_dot_time.append(time.time() - start_3)

        start_4 = time.time()
        np.dot(A,B) #numpy's operation again
        mat_mat_dot_time.append(time.time() - start_4)

    #plot using linear scale
    ax2 = plt.subplot(121)
    ax2.plot(n_vals, mat_vec_time)
    ax2.plot(n_vals, mat_mat_time)
    ax2.plot(n_vals, mat_vec_dot_time)
    ax2.plot(n_vals, mat_mat_dot_time)
    ax2.scatter(n_vals, mat_vec_time)
    ax2.scatter(n_vals, mat_mat_time)
    ax2.scatter(n_vals, mat_vec_dot_time)
    ax2.scatter(n_vals, mat_mat_dot_time)
    ax2.set_xlabel("n") 
    ax2.set_ylabel("Seconds")
    ax2.set_title("Linear Scale")
    
    #plot using log scale
    ax1 = plt.subplot(122)
    ax1.plot(n_vals, mat_vec_time)
    ax1.plot(n_vals, mat_mat_time)
    ax1.plot(n_vals, mat_vec_dot_time)
    ax1.plot(n_vals, mat_mat_dot_time)
    ax1.scatter(n_vals, mat_vec_time)
    ax1.scatter(n_vals, mat_mat_time)
    ax1.scatter(n_vals, mat_vec_dot_time)
    ax1.scatter(n_vals, mat_mat_dot_time)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel("n") 
    ax1.set_title("Logarithmic Scale")
    


    plt.show()

    #raise NotImplementedError("Problem 4 Incomplete")
