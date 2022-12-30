import numpy as np
from itertools import repeat


def to_bin(num, bits: int, limits: np.ndarray):
    max_ = 2**bits
    int_space = np.linspace(0, max_ - 1, max_, dtype=int)
    org_space = np.linspace(limits[0], limits[1], max_)
    idx = (np.abs(org_space - num)).argmin()

    return '0b' + format(int_space[idx], f'0{bits}b')


def float_to_gray(n: float, bits: int, limits: np.ndarray):

    binary = to_bin(n, bits, limits)
    n = int(binary, 2)
    gray = format(n ^ (n >> 1), f'0{bits}b')

    return '0b'+gray


def binary_coding(population: np.ndarray, bits: int = 8):

    for agent in population:
        agent.vector = np.array(list(map(to_bin, agent.vector, repeat(bits), agent.limits)))


def gray_coding(population: np.ndarray, bits: int = 0):

    for agent in population:
        agent.vector = np.array(list(map(float_to_gray, agent.vector, repeat(bits), agent.limits)))


def triallelic_coding(poppulation: np.ndarray):
    pass
