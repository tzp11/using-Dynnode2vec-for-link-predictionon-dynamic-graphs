import os
import re
import networkx as nx
import numpy as np


def read_file_and_process_data(file_path, k):
    # Read file and process data
    with open(file_path, 'r') as file:
        data = file.read()

    # Extract nodes and times using regex
    nodes_and_times = re.findall(r'(\d+) +(\d+) +(\d+)', data)
    nodes_and_times = [(int(node1), int(node2), int(time)) for node1, node2, time in nodes_and_times]


    # 统计图中所有节点中不同的个数
    all_nodes = set()
    for node1, node2, time in nodes_and_times:
        all_nodes.add(node1)
        all_nodes.add(node2)
    #将all_nodes进行排序
    all_nodes = sorted(list(all_nodes))
    print(len(all_nodes))
    #将nods_and_times中的节点id替换为排序后的id,并进行按照时间的排序
    nodes_and_times = sorted(nodes_and_times, key=lambda x: x[2])
    nodes_and_times = [(all_nodes.index(node1), all_nodes.index(node2), time) for node1, node2, time in nodes_and_times]
    #将nodes_and_times按照time等分为k份

    nodes_and_times = np.array_split(nodes_and_times, k)
    graphs = []
    for i in range(k):
        graph = nx.Graph()
        for node in all_nodes:
            graph.add_node(all_nodes.index(node))
        for node1, node2, time in nodes_and_times[i]:
            graph.add_edge(node1, node2)
#            print(node1, node2, time)
        graphs.append(graph)
    return graphs



if __name__ == '__main__':
    file_path = "D:\\qqqqqqqqqqqqqqqqqq\\pycharm\\work\\Thsis\\dataProcess\\data"
    num_partitions = 5
    graphs = read_file_and_process_data(file_path, num_partitions)
    #依次输出五个图
    for i in range(num_partitions):
        print(graphs[i].number_of_nodes())
        print(graphs[i].number_of_edges())
