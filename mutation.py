import numpy as np
import random

from utils import decrease_to_limit


def _gen_bits(n_bits):
    bits = "0b"
    for i in range(n_bits + 1):
        if random.uniform(0, 1) < 0.5:
            bits += "0"
        else:
            bits += "1"
    return bits


# input example '0b1010101011'
def _bitwise_inverse_mutation(value: bin):
    borders = np.random.randint(2, len(value), size=(2,))
    start = min(borders)
    stop = max(borders)
    mutation_bits = _gen_bits(stop - start)
    return value[:start] + mutation_bits[2:] + value[stop + 1:]


# input example '0b1010101011'
def _bitwise_put_mutation(value: bin):
    return _gen_bits(len(value) - 3)


# input example '0b1010101011'
def _bitwise_transfer_mutation(value: bin):
    borders = np.random.randint(2, len(value), size=(2,))
    start = min(borders)
    stop = max(borders)
    mutation_bits = _gen_bits(stop - start)
    putting_place = np.random.randint(2, len(value) - len(mutation_bits) + 3)
    return value[:putting_place] + mutation_bits[2:] + value[putting_place + len(mutation_bits) - 2:]


def _bitwise_exchange_mutation(value: bin):
    bit_list = list(value[2:])
    random.shuffle(bit_list)
    return "0b" + ''.join(bit_list)


# Hard to test without coding/decoding algorithm
def mutation_bin_gen(population: np.ndarray, limits: np.ndarray, mutation_probability: float = 0.03, data: np.ndarray = None, capacity: int = None):
    for agent in population:
        if np.random.uniform(0, 1) <= mutation_probability:
            parameter_idx = np.random.randint(0, len(agent.vector))
            mutation_type = np.random.randint(0, 4)
            if mutation_type == 0:
                # print('inverse')
                value = _bitwise_inverse_mutation(agent.vector[parameter_idx])
            elif mutation_type == 1:
                # print('put')
                value = _bitwise_put_mutation(agent.vector[parameter_idx])
            elif mutation_type == 2:
                # print('transfer')
                value = _bitwise_transfer_mutation(agent.vector[parameter_idx])
            else:
                # print('exchange')
                value = _bitwise_exchange_mutation(agent.vector[parameter_idx])

            if isinstance(limits, str):
                if int(limits[parameter_idx][0], 2) < int(value, 2) < int(limits[parameter_idx][1], 2):
                    # print(f'Mutation from {agent.vector[parameter_idx]} to {value}')
                    if capacity is not None:
                        value = decrease_to_limit([value], (((data), capacity)))[0]
                    agent.vector[parameter_idx] = value
            else:
                if limits[parameter_idx][0] < int(value, 2) < limits[parameter_idx][1]:
                    if capacity is not None:
                        value = decrease_to_limit([value], ((((data), capacity)),), mutation=True)[0]
                    agent.vector[parameter_idx] = value


def trit_greater_than(first, second):
    if first == "#":
        return False
    if first == "0":
        if second == "#":
            return True
        return False
    if first == "1":
        if second in ["#", "0"]:
            return True
        return False
    return False


def trit_value_greater_than(first: str, second: str):
    for i in range(2, len(first)):
        print(first[i], second[i])
        if first[i] != second[i]:
            if trit_greater_than(first[i], second[i]):
                return True  # Greater
            return False  # Less
    return False  # Equal


def _gen_trits(n_bits):
    bits = "0t"
    for i in range(n_bits + 1):
        x = random.uniform(0, 1)
        if x < 0.333:
            bits += "0"
        elif x < 0.666:
            bits += "1"
        else:
            bits += "#"
    return bits


# input example '0b10#1001#0'
# [# -> -1]
def _triwise_inverse_mutation(value: str):
    borders = np.random.randint(2, len(value), size=(2,))
    start = min(borders)
    stop = max(borders)
    mutation_trits = _gen_trits(stop - start)
    return value[:start] + mutation_trits[2:] + value[stop + 1:]


# input example '0b10#1001#0'
# [# -> -1]
def _triwise_put_mutation(value: str):
    return _gen_trits(len(value) - 2)


# input example '0b10#1001#0'
# [# -> -1]
def _triwise_transfer_mutation(value: str):
    borders = np.random.randint(2, len(value), size=(2,))
    start = min(borders)
    stop = max(borders)
    mutation_trits = _gen_trits(stop - start)
    putting_place = np.random.randint(2, len(value) - len(mutation_trits) + 3)
    return value[:putting_place] + mutation_trits[2:] + value[putting_place + len(mutation_trits) - 2:]


# input example '0b10#1001#0'
# [# -> -1]
def _triwise_exchange_mutation(value: str):
    trits_list = list(value[2:])
    random.shuffle(trits_list)
    return "0b" + ''.join(trits_list)


# Hard to test without coding/decoding algorithm
def mutation_tri_gen(population: np.ndarray, limits: np.ndarray, mutation_probability: float = 0.03):
    for agent in population:
        if np.random.uniform(0, 1) <= mutation_probability:
            parameter_idx = np.random.randint(0, len(agent.vector))
            mutation_type = np.random.randint(0, 4)
            if mutation_type == 0:
                value = _triwise_inverse_mutation(agent.vector[parameter_idx])
            elif mutation_type == 1:
                value = _triwise_put_mutation(agent.vector[parameter_idx])
            elif mutation_type == 2:
                value = _triwise_transfer_mutation(agent.vector[parameter_idx])
            else:
                value = _triwise_exchange_mutation(agent.vector[parameter_idx])

            if trit_value_greater_than(value, limits[parameter_idx][0]) and \
                    trit_value_greater_than(limits[parameter_idx][1], value):
                agent.vector[parameter_idx] = value


def mutation_bin_fen(population: np.ndarray, limits: np.ndarray, mutation_probability: float = 0.03):
    for agent in population:
        if np.random.uniform(0, 1) <= mutation_probability:
            parameter_idx = np.random.randint(0, len(agent.vector))
            value = _bitwise_put_mutation(agent.vector[parameter_idx])

            if isinstance(limits, str):
                if int(limits[parameter_idx][0], 2) > int(value, 2) > int(limits[parameter_idx][1], 2):
                    agent.vector[parameter_idx] = value
            else:
                if limits[parameter_idx][0] > int(value, 2) > limits[parameter_idx][1]:
                    agent.vector[parameter_idx] = value


def mutation_tri_fen(population: np.ndarray, limits: np.ndarray, mutation_probability: float = 0.03):
    for agent in population:
        if np.random.uniform(0, 1) <= mutation_probability:
            parameter_idx = np.random.randint(0, len(agent.vector))
            value = _triwise_put_mutation(agent.vector[parameter_idx])

            if trit_value_greater_than(value, limits[parameter_idx][0]) and \
                    trit_value_greater_than(limits[parameter_idx][1], value):
                agent.vector[parameter_idx] = value


def mutation_real_fen(population: np.ndarray, limits: np.ndarray, mutation_probability: float = 0.03,
                      iteration: int = 0, max_iter: int = 10000):
    iter_factor = (1 - iteration / max_iter) ** 2
    for agent in population:
        if np.random.uniform(0, 1) <= mutation_probability:
            parameter_idx = np.random.randint(0, len(agent.vector))
            r = np.random.uniform(0, 1)
            upper_border = agent.vector[parameter_idx] + r * (
                        limits[parameter_idx][1] - agent.vector[parameter_idx]) * iter_factor
            down_border = agent.vector[parameter_idx] - r * (
                        agent.vector[parameter_idx] - limits[parameter_idx][0]) * iter_factor
            agent.vector[parameter_idx] = np.random.uniform(low=down_border, high=upper_border)


def mutation_salesman_problem(population: np.ndarray, limits: np.ndarray, mutation_probability: float = 0.03):
    for agent in population:
        if np.random.uniform(0, 1) <= mutation_probability:
            first = np.random.randint(0, len(agent.vector))
            second = np.random.randint(0, len(agent.vector))

            temp = agent.vector[first]
            agent.vector[first] = agent.vector[second]
            agent.vector[second] = temp
