#实现DeepWalk算法的程序包
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.word2vec import LineSentence
import networkx as nx
import numpy as np

class DeepWalk:
    def __init__(self, graph,  emb_size,walk_length, num_walks):
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.emb_size = emb_size
        self.model = None

    def generate_random_walks(self):
        # 生成随机游走序列
        walks = []
        for _ in range(self.num_walks):
            for node in self.graph.nodes():
                walks.append(self.random_walk(node))
        return walks

    def random_walk(self, start_node):
        walk = [start_node]
        while len(walk) < self.walk_length:
            neighbors = list(self.graph.neighbors(walk[-1]))
            if len(neighbors) > 0:
                walk.append(np.random.choice(neighbors))
            else:
                break
        return list(map(str, walk))

    def train(self, walks):
        # 训练Word2Vec模型
        self.model = Word2Vec(walks, vector_size=self.emb_size, window=5, min_count=0, sg=1, workers=4,
                              epochs=1)
        nodes_embedding = self.model.wv.vectors
        return nodes_embedding

def deepwalk_embedding(graph, walk_length, num_walks, emb_size):
    dw=DeepWalk(graph=graph,emb_size=emb_size, walk_length=walk_length, num_walks=num_walks)
    walks = dw.generate_random_walks()
    nodes_embedding = dw.train(walks);
    return nodes_embedding

if __name__ == '__main__':
    # 示例代码
    G = nx.karate_club_graph()
    nodes_embedding = deepwalk_embedding(graph=G, walk_length=80, num_walks=10, emb_size=128)
    print(nodes_embedding)
