# two_pass.py

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


def first_pass(g) -> list:
    graph = deepcopy(g)
    height = len(graph)
    width = len(graph[0])
    label = 1
    index_dict = {}
    for h in range(height):
        for w in range(width):
            if graph[h][w] == 0:
                continue
            if h == 0 and w == 0:
                graph[h][w] = label
                label += 1
                continue
            if h == 0 and graph[h][w - 1] > 0:
                graph[h][w] = graph[h][w - 1]
                continue
            if w == 0 and graph[h - 1][w] > 0:
                if graph[h - 1][w] <= graph[h - 1][min(w + 1, width - 1)]:
                    graph[h][w] = graph[h - 1][w]
                    index_dict[graph[h - 1][min(w + 1, width - 1)]] = graph[h - 1][w]
                elif graph[h - 1][min(w + 1, width - 1)] > 0:
                    graph[h][w] = graph[h - 1][min(w + 1, width - 1)]
                    index_dict[graph[h - 1][w]] = graph[h - 1][min(w + 1, width - 1)]
                continue
            if h == 0:
                graph[h][w] = label
                label += 1
                continue
            if w == 0:
                if graph[h - 1][min(w + 1, width - 1)] > 0:
                    graph[h][w] = graph[h - 1][min(w + 1, width - 1)]
                    continue
                graph[h][w] = label
                label += 1
                continue
            neighbors = [graph[h - 1][w], graph[h][w - 1], graph[h - 1][w - 1], graph[h - 1][min(w + 1, width - 1)]]
            neighbors = list(filter(lambda x: x > 0, neighbors))
            if len(neighbors) > 0:
                graph[h][w] = min(neighbors)
                for n in neighbors:
                    if n in index_dict:
                        index_dict[n] = min(index_dict[n], min(neighbors))
                    else:
                        index_dict[n] = min(neighbors)
                continue
            graph[h][w] = label
            label += 1
    return graph, index_dict


def remap(idx_dict) -> dict:
    index_dict = deepcopy(idx_dict)
    for id in idx_dict:
        idv = idx_dict[id]
        while idv in idx_dict:
            if idv == idx_dict[idv]:
                break
            idv = idx_dict[idv]
        index_dict[id] = idv
    return index_dict


def second_pass(g, index_dict) -> list:
    graph = deepcopy(g)
    height = len(graph)
    width = len(graph[0])
    for h in range(height):
        for w in range(width):
            if graph[h][w] == 0:
                continue
            if graph[h][w] in index_dict:
                graph[h][w] = index_dict[graph[h][w]]
    return graph


def flatten(g) -> list:
    graph = deepcopy(g)
    fgraph = sorted(set(list(graph.flatten())))
    flatten_dict = {}
    for i in range(len(fgraph)):
        flatten_dict[fgraph[i]] = i
    graph = second_pass(graph, flatten_dict)
    return graph


def two_pass(graph):
    graph_1, idx_dict = first_pass(graph)
    idx_dict = remap(idx_dict)
    graph_2 = second_pass(graph_1, idx_dict)
    graph_3 = flatten(graph_2)
    return graph_3


if __name__ == "__main__":
    np.random.seed(1)
    graph = np.random.choice([0, 1], size=(20, 20))
    graph_1, idx_dict = first_pass(graph)
    idx_dict = remap(idx_dict)
    graph_2 = second_pass(graph_1, idx_dict)
    graph_3 = flatten(graph_2)
    print(graph_3)

    plt.subplot(131)
    plt.imshow(graph)
    plt.subplot(132)
    plt.imshow(graph_3)
    plt.subplot(133)
    plt.imshow(graph_3 > 0)
    # plt.savefig('random_bin_graph.png')