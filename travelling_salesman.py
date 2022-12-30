import numpy as np


def read_tsp_data(filepath: str):
    with open(filepath, 'r') as file:
        dimension = 0
        while True:
            line = file.readline().split()
            print(line)
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
