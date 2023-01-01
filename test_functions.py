import numpy as np


def circle_function(x: np.ndarray):
    return -x[0]**2 - x[1]**2


def quadratic_function(x: np.ndarray):
    return -x[0]**2


# range [-15, 15]
# [+-1.34, +-1.34] -> 2.06261
def cross_in_tray_function(x: np.ndarray):
    return -(-0.0001 * (abs(np.sin(x[0]) * np.sin(x[1]) * np.exp(abs(100 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi))) + 1) ** 0.1)


# range x1: [-15, -5] x2: [-3, 3]
# [-10, 1] -> doesnt work
def bukin_function(x: np.ndarray):
    return -(100*np.sqrt(abs(x[1]-0.01*x[0]**2))+0.01*abs(x[0]+10))


# range [-5.12, 5.12]
# [0, 0] -> 1
def drop_wave_function(x: np.ndarray):
    return -(-(1+np.cos(12*np.sqrt(x[0]**2+x[1]**2)))/(0.5*(x[0]**2+x[1]**2)+2))


# range [-512, 512]
# [512, 404.2319] -> doesnt work
def egg_holder_function(x: np.ndarray):
    return -(-(x[1]+47)*np.sin(np.sqrt(abs(x[1]+0.5*x[0]+47)))-x[0]*np.sin(np.sqrt(abs(x[0]-(x[1]+47)))))


# range [-600, 600]
# [0, 0] ->
def griewank_function(x: np.ndarray):
    return -((x[0]**2+x[1]**2)/4000 - np.cos(x[0]/np.sqrt(1))*np.cos(x[1]/np.sqrt(2)) + 1)


# range [-10, 10]
# [+-8.055, +-9.665] ->
def holder_table_function(x: np.ndarray):
    return -(-abs(np.sin(x[0])*np.cos(x[1])*np.exp(abs(1-np.sqrt(x[0]**2+x[1]**2)/np.pi))))


# range [-10, 10]
# [1, 1] ->
def levy_function_n13(x: np.ndarray):
    return -((np.sin(3*np.pi*x[0]))**2+(x[0]-1)**2*(1+(np.sin(3*np.pi*x[1]))**2)+(x[1]-1)**2*(1+(np.sin(2*np.pi*x[1]))**2))


# range [-5.12, 5.12]
# [0, 0] ->
def rastrigin_function(x: np.ndarray):
    return -(20+(x[0]**2-10*np.cos(2*np.pi*x[0]))+(x[1]**2-10*np.cos(2*np.pi*x[1])))


def dummy(d):
    pass


def salesman_function(vector: np.ndarray, data: np.ndarray):
    base = np.take(data, vector, axis=0)
    neighbours = np.roll(base, 1, axis=0)
    return -np.sum(np.abs(base[:, 1] - neighbours[:, 1]) + np.abs(base[:, 2] - neighbours[:, 2]))


def knapsack_function(vector: np.ndarray, data: np.ndarray, cap, *args):
    temp = np.array([*vector[0][2:]]).astype(int)
    idx = np.argwhere(temp == 1)
    if np.sum(data[:, 1][idx]) <= cap:
        return np.sum(data[:, 2][idx])
    else:
        return -99999999

# def knapsack_function(vector: np.ndarray, data: np.ndarray, *args):
#     return np.sum(np.multiply(vector, data[:, 2]))
