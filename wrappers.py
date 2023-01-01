import numpy as np


class WrappedCallback:
    def __init__(self, function: callable, parameters: tuple = None):
        self._f = function
        self.parameters = () if parameters is None else parameters

    def __call__(self, vector: np.ndarray, args: tuple = None):
        if args is None:
            return self._f(vector, *())
        return self._f(vector, *args)


class OptimizationTask(WrappedCallback):
    def __init__(self, function: callable, limits: np.ndarray, target_x: list = None, target_y: float = None,
                 args: tuple = None, salesman: bool = False):
        super().__init__(function, args)
        self.limits = limits
        self.size = len(limits)
        self.target_x = target_x
        self.target_y = target_y
        self.salesman = salesman


class Coding(WrappedCallback):
    def __init__(self, coding: callable, decoding: callable, parameters: tuple = None):
        super().__init__(coding, parameters)
        self._decoding = decoding

    def encode(self, population: np.ndarray, args: tuple = None):
        if args is None:
            return self._f(population, *self.parameters)
        return self._f(population, *args, *self.parameters)

    def decode(self, population: np.ndarray, args: tuple = None):
        if args is None:
            return self._f(population, *self.parameters)
        return self._f(population, *args, *self.parameters)
