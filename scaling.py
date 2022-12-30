import numpy as np


def linear(parents: np.ndarray, children: np.ndarray, c: int = 2):
    fitness_extractor = np.vectorize(lambda agent: agent.fitness_value)
    parents_and_children = np.concatenate((parents, children))
    fitness_array = fitness_extractor(parents_and_children)
    favg = np.average(fitness_array)
    fmin = np.min(fitness_array)
    fmax = np.max(fitness_array)
    if fmin == fmax:
        return

    if fmin > (c * favg - fmax) / (c - 1):
        a = favg * (c - 1) / (fmax - favg)
        b = (1 - a) * favg
    else:
        a = favg / (favg - fmin)
        b = -a * fmin

    for agent in parents_and_children:
        agent.scaled_fitness_value = a * agent.fitness_value + b


def sigma_clipping(parents: np.ndarray, children: np.ndarray, c: int = 2):
    fitness_extractor = np.vectorize(lambda agent: agent.fitness_value)
    parents_and_children = np.concatenate((parents, children))
    fitness_array = fitness_extractor(parents_and_children)
    favg = np.average(fitness_array)
    sigma = np.sqrt(1/len(parents_and_children) * np.sum((fitness_array - favg)**2))

    for agent in parents_and_children:
        agent.scaled_fitness_value = agent.fitness_value + (favg - c * sigma)


def exponential(parents: np.ndarray, children: np.ndarray, k: float = 1.005):
    for agent in parents:
        agent.scaled_fitness_value = agent.fitness_value ** k

    for agent in children:
        agent.scaled_fitness_value = agent.fitness_value ** k
