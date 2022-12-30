import numpy as np
import random


def full_sub_strategy(parents: np.ndarray, children: np.ndarray):
    return children


def part_reproduction_random_sub_strategy(parents: np.ndarray, children: np.ndarray):
    return np.array(random.sample(list(np.concatenate((parents, children))), len(parents)))


def part_reproduction_elite_sub_strategy(parents: np.ndarray, children: np.ndarray):
    new_population = sorted(np.concatenate((parents, children)), key=lambda agent: agent.fitness_value, reverse=True)
    return new_population[:len(parents)]


def part_reproduction_similar_agents_fen_sub_strategy(parents: np.ndarray, children: np.ndarray, k: int = None):
    if len(parents) == len(children):
        return children
    if k is None:
        k = 2 if len(parents) < 7 else 5

    for child in children:
        agent_to_replace = min(random.sample(list(parents), k), key=lambda agent: agent.fen_similarity(child))
        idx = np.nonzero(parents == agent_to_replace)[0][0]
        parents[idx] = child

    return parents


def part_reproduction_similar_agents_gen_sub_strategy(parents: np.ndarray, children: np.ndarray, k: int = None):
    if len(parents) == len(children):
        return children
    if k is None:
        k = 2 if len(parents) < 7 else 5

    for child in children:
        agent_to_replace = min(random.sample(list(parents), k), key=lambda agent: agent.gen_similarity(child))
        idx = np.nonzero(parents == agent_to_replace)[0][0]
        parents[idx] = child

    return parents

