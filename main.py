# 定义一个虚拟的图类
from GraphEmbedding import DyyNode2Vec


class VirtualGraph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def neighbors(self, node):
        return self.edges.get(node, [])

    def nodes(self):
        return self.nodes

# 实例化一个虚拟的图对象
graph = VirtualGraph(nodes=['A', 'B', 'C'], edges={'A': ['B', 'C'], 'B': ['A'], 'C': ['A']})

# 实例化DyyNode2Vec类
dyy_node2vec = DyyNode2Vec(graphs=[graph], dimension=128, p=1, q=1, walk_length=10, num_walks=5)

# 假设在类中的方法调用
prev_node = 'A'
cur_node = 'B'
neighbors = graph.neighbors(cur_node)
probabilities = [0.4, 0.1, 0.5]  # 假设的边缘概率
next_node = 'A'  # 假设通过随机选择得到的下一个节点

# 模拟随机漫步过程
walk = []
walk.append(prev_node)
walk.append(next_node)

print(walk)  # 打印随机漫步路径
