#这是一个软件包，输入的一组节点的嵌入向量，输出每条边对应的嵌入向量，即节点嵌入向量的拼接。
#边向量是节点向量差的评分，即边向量 = （节点1的嵌入向量 - 节点2的嵌入向量）^2

import numpy as np

def WeightL2(node_embedding):
    """
    输入一组节点的嵌入向量，输出每条边对应的嵌入向量。
    """
    edge_embedding = []
    for i in range(len(node_embedding)):
        for j in range(i+1, len(node_embedding)):
            edge_embedding.append(np.square(node_embedding[i] - node_embedding[j]))
    return edge_embedding

def WeightL1(node_embedding):
    """
    输入一组节点的嵌入向量，输出每条边对应的嵌入向量。
    """
    edge_embedding = []
    for i in range(len(node_embedding)):
        for j in range(i+1, len(node_embedding)):
            edge_embedding.append(node_embedding[i] - node_embedding[j])
    return edge_embedding



if __name__ == '__main__':
    node_embedding = np.array([[2,7,3],[6,5,4],[2,9,8]])
    print(node_embedding)
    edge_embedding = Node2Edge(node_embedding)
    print(edge_embedding)

