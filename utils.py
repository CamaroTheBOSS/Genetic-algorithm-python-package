import numpy as np


def read_tsp_data(filepath: str):
    with open(filepath, 'r') as file:
        dimension = 0
        while True:
            line = file.readline().split()
            if line[0] == "DIMENSION":
                dimension = int(line[2])
            elif line[0] == "NODE_COORD_SECTION":
                break

        data = np.zeros((dimension, 3), dtype=int)
        for i in range(dimension):
            line = file.readline().split()
            data[i] = np.asarray(line, dtype=int)

    data[:, 0] = data[:, 0] - 1
    return data


def read_knapsack_data(filepath: str):
    with open(filepath, 'r') as file:
        while True:
            line = file.readline().split()
            if line[0] == "N_OBJECTS:":
                n_objects = int(line[1])
            elif line[0] == "CAPACITY:":
                capacity = int(line[1])
            elif line[0] == "ITEMS:":
                break

        data = np.zeros((n_objects, 3), dtype=int)
        for i in range(n_objects):
            line = file.readline().split()
            data[i] = np.asarray(line, dtype=int)

    return data, capacity


def decrease_to_limit(init_vector, args, mutation=False):
    temp = np.array([*init_vector[0][2:]]).astype(int)
    indexes = np.argwhere(temp == 1)

    while np.sum(args[0][0][:, 1][indexes]) > args[0][1]:
        idx = np.random.choice([i for i, ltr in enumerate(init_vector[0]) if ltr == '1'])
        init_vector[0] = init_vector[0][:idx] + '0' + init_vector[0][idx + 1:]
        temp = np.array([*init_vector[0][2:]]).astype(int)
        indexes = np.argwhere(temp == 1)

    return init_vector
