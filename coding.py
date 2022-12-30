import numpy as np
from itertools import repeat


def decimal_converter(n):
    while n > 1:
        n /= 10
    return n


def float_to_bin(n: float, decimal_places: int):

    whole, decimal = str(n).split(".")
    whole, decimal = int(whole), int(decimal)
    res = bin(whole)[2:] + "."

    for i in range(decimal_places):
        whole, decimal = str((decimal_converter(decimal))*2).split(".")
        decimal = int(decimal)
        res += whole

    return '0b'+res


def float_to_gray(n: float, decimal_places: int):

    binary = float_to_bin(n, decimal_places)
    joined = ''.join(binary.split('.'))
    gray = int_to_gray(int(joined, 2))[2:]
    gray = '.'.join([gray[:-4], gray[-4:]])

    return '0b'+gray


def int_to_gray(n: int):
    return bin(n ^ (n >> 1))


def binary_coding(population: np.ndarray, decimal_places: int = 0):

    if decimal_places == 0:
        for agent in population:
            agent.vector = np.array(list(map(bin, agent.vector)))
    else:
        for agent in population:
            agent.vector = np.array(list(map(float_to_bin, agent.vector, repeat(decimal_places))))


def gray_coding(population: np.ndarray, decimal_places: int = 0):

    if decimal_places == 0:
        for agent in population:
            agent.vector = np.array(list(map(int_to_gray, agent.vector)))
    else:
        for agent in population:
            agent.vector = np.array(list(map(float_to_gray, agent.vector, repeat(decimal_places))))


def triallelic_coding(poppulation: np.ndarray):
    pass
