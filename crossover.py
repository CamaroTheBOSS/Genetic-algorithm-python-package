import numpy as np
from copy import deepcopy


def replace_pmx(this_temp, other_temp, value, j):
    replaced = False
    while not replaced:
        idx = np.argwhere(this_temp == value)[0][0]
        if other_temp[idx] not in this_temp:
            this_temp[j] = other_temp[idx]
            replaced = True
        else:
            value = other_temp[idx]


def pmx(population: np.ndarray):
    np.random.shuffle(population)
    parents1, parents2 = np.array_split(population, 2)
    children1, children2 = deepcopy(parents1), deepcopy(parents2)
    for i in range(min(len(parents1), len(parents2))):
        x_points = np.random.choice(range(len(population[0].vector)), 2, replace=False)
        p1, p2 = np.min(x_points), np.max(x_points)
        temp1, temp2 = -1 * np.ones_like(parents1[i].vector), -1 * np.ones_like(parents2[i].vector)
        temp1[p1:p2] = parents2[i].vector[p1:p2]
        temp2[p1:p2] = parents1[i].vector[p1:p2]

        for j in range(p1):
            if parents1[i].vector[j] not in temp1:
                temp1[j] = parents1[i].vector[j]
            else:
                replace_pmx(temp1, temp2, parents1[i].vector[j], j)

            if parents2[i].vector[j] not in temp2:
                temp2[j] = parents2[i].vector[j]
            else:
                replace_pmx(temp2, temp1, parents2[i].vector[j], j)

        for j in range(p2, len(parents1[i].vector)):
            if parents1[i].vector[j] not in temp1:
                temp1[j] = parents1[i].vector[j]
            else:
                replace_pmx(temp1, temp2, parents1[i].vector[j], j)

            if parents2[i].vector[j] not in temp2:
                temp2[j] = parents2[i].vector[j]
            else:
                replace_pmx(temp2, temp1, parents2[i].vector[j], j)

        children1[i].vector = temp1
        children2[i].vector = temp2
    return np.concatenate((children1, children2))


def arithmetic_crossover(population: np.ndarray):
    np.random.shuffle(population)
    parents1, parents2 = np.array_split(population, 2)
    children1, children2 = deepcopy(parents1), deepcopy(parents2)
    for i in range(min(len(parents1), len(parents2))):
        a = np.random.rand()
        temp1 = a * parents1[i].vector + (1 - a) * parents2[i].vector
        temp2 = a * parents2[i].vector + (1 - a) * parents1[i].vector
        children1[i].vector = temp1
        children2[i].vector = temp2
    return np.concatenate((children1, children2))


def mixed_crossover(population: np.ndarray):
    np.random.shuffle(population)
    parents1, parents2 = np.array_split(population, 2)
    children1, children2 = deepcopy(parents1), deepcopy(parents2)
    for i in range(min(len(parents1), len(parents2))):
        temp1, temp2 = parents1[i].vector, parents2[i].vector
        x_point = np.random.randint(len(parents1[i].vector))
        a = np.random.rand()
        temp1[x_point:] = a * parents1[i].vector[x_point:] + (1 - a) * parents2[i].vector[x_point:]
        temp2[x_point:] = a * parents2[i].vector[x_point:] + (1 - a) * parents1[i].vector[x_point:]
        children1[i].vector = temp1
        children2[i].vector = temp2
    return np.concatenate((children1, children2))


def binary_crossover(population: np.ndarray, n_of_points: int = 1):
    np.random.shuffle(population)
    parents1, parents2 = np.array_split(population, 2)
    children1, children2 = deepcopy(parents1), deepcopy(parents2)
    for i in range(min(len(parents1), len(parents2))):
        x_points = sorted(np.random.choice(range(1, len(population[0].vector[0][2:])), n_of_points, replace=False))
        for k in range(len(children1[i].vector)):
            temp1 = [children1[i].vector[k][:2+x_points[0]]]
            temp2 = [children2[i].vector[k][:2+x_points[0]]]
            for j in range(len(x_points)):
                if j+1 < len(x_points):
                    if j % 2 != 0:
                        temp1.append(children2[i].vector[k][2+x_points[j]:2+x_points[j + 1]])
                        temp2.append(children1[i].vector[k][2+x_points[j]:2+x_points[j + 1]])
                    else:
                        temp1.append(children1[i].vector[k][2+x_points[j]:2+x_points[j + 1]])
                        temp2.append(children2[i].vector[k][2+x_points[j]:2+x_points[j + 1]])
                else:
                    if j % 2 != 0:
                        temp1.append(children2[i].vector[k][2+x_points[j]:])
                        temp2.append(children1[i].vector[k][2+x_points[j]:])
                    else:
                        temp1.append(children1[i].vector[k][2+x_points[j]:])
                        temp2.append(children2[i].vector[k][2+x_points[j]:])

            children1[i].vector[k] = ''.join(temp1)
            children2[i].vector[k] = ''.join(temp2)

    return np.concatenate((children1, children2))


def ox_fill_empty_spots(v_: np.ndarray, parent_vector: np.ndarray, points: tuple):

    v_seq = np.concatenate((parent_vector[points[1]:], parent_vector[:points[1]]))
    v_seq_ = np.array([gene for gene in v_seq if gene not in v_])
    v_[points[1]:] = v_seq_[:len(v_[points[1]:])]
    v_[:points[0]] = v_seq_[len(v_[points[1]:]):]


def ox(population: np.ndarray):
    np.random.shuffle(population)
    parents1, parents2 = np.array_split(population, 2)
    children1, children2 = deepcopy(parents1), deepcopy(parents2)
    for i in range(min(len(parents1), len(parents2))):
        x_points = np.random.choice(range(len(population[0].vector)), 2, replace=False)
        p1, p2 = np.min(x_points), np.max(x_points)
        vr_, vs_ = -1 * np.ones_like(parents1[i].vector), -1 * np.ones_like(parents2[i].vector)
        vr_[p1:p2] = parents1[i].vector[p1:p2]
        vs_[p1:p2] = parents2[i].vector[p1:p2]
        ox_fill_empty_spots(vr_, parents2[i].vector, (p1, p2))
        ox_fill_empty_spots(vs_, parents1[i].vector, (p1, p2))
        children1[i].vector = vr_
        children2[i].vector = vs_

    return np.concatenate((children1, children2))
