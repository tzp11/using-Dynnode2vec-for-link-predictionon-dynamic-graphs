# 重写的 GraphToLabel 函数
from random import random

import networkx as nx
import numpy as np


def GraphToLabel(G):
    """
    This function takes a networkx graph G and returns a label array.
    """
    # 获取节点数量
    num_nodes = G.number_of_nodes()

    # 初始化标签数组
    label_array = np.zeros(int((num_nodes - 1) * num_nodes / 2))

    # 循环遍历所有可能的节点对
    index = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # 检查是否存在边
#            print(i,j)
            if G.has_edge(i, j):
                # 计算标签数组索引并设置对应值为 1
                label_array[index] = 1
            index += 1
    return label_array



if __name__ == '__main__':
    # Example usage
    G = nx.Graph()
    #随机生成图的边和节点
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (4,3), (3, 5), (5,4), (4, 6), (5, 6), (5, 7), (6, 7), (6, 8), (7, 8), (7, 9), (8, 9)]
    nodes = list(range(10))
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    print(edges)
    G.add_edges_from(edges)
    label_array = GraphToLabel(G)
    print(label_array)


