import numpy as np


class OptimizationTask:
    def __init__(self, function: callable, limits: np.ndarray, target_x: list = None, target_y: float = None,
                 args: tuple = None):
        self._f = function
        self.limits = limits
        self.size = len(limits)
        self.args = () if args is None else args
        self.target_x = target_x
        self.target_y = target_y

    def __call__(self, vector: np.ndarray):
        return self._f(vector)


class Coding:
    def __init__(self, coding: callable, decoding: callable, args: tuple = None):
        self._coding = coding
        self._decoding = decoding
        self.args = () if args is None else args

    def encode(self, population: np.ndarray):
        self._coding(population, *self.args)

    def decode(self, population: np.ndarray):
        self._decoding(population, *self.args)
