#实现dynnode2vec的软件包，
# 输入：一组时序图，dimension,p,q,walk_length,num_walks
# 输出：综合所有时序图节点嵌入
import random

from gensim.models import Word2Vec

from GraphEmbedding import Node2vec


class DyyNode2Vec:
    def __init__(self, graphs, dimension, p, q, walk_length, num_walks,sig,k):
        self.graphs = graphs
        self.dimension = dimension
        self.p = p
        self.q = q
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.sig = sig
        self.k=k

    def generate_walks(self, old_graph,new_graph):
        walks = []
        if self.sig==1:
            for _ in range(self.num_walks):
                change_nodes = self.ChangeNode(old_graph, new_graph)
                for node in change_nodes:
                    walks.append(self.biased_walk(new_graph, node))
        elif self.sig==2:
            for _ in range(int(self.num_walks*self.k)):
                change_nodes = self.ChangeNode(old_graph, new_graph)
                for node in change_nodes:
                    walks.append(self.biased_walk(new_graph, node))
        else:
            for _ in range(self.num_walks):
                for node in new_graph.nodes():
                    walks.append(self.biased_walk(new_graph, node))
        return walks

    def ChangeNode(self, old_graph, new_graph):
        change_nodes = []
        # 遍历所有节点,如果节点连接的边发生变化，则加入change_nodes
        for node in new_graph.nodes():
            #邻居有变化则加入
             if len(set(new_graph.neighbors(node)) - set(old_graph.neighbors(node))) > 0 or \
                     len(set(new_graph.neighbors(node)) - set(old_graph.neighbors(node))) < 0:
                change_nodes.append(node)
        return change_nodes

    def biased_walk(self, graph, start_node):
        walk = [start_node]
        while len(walk) < self.walk_length:
            cur = walk[-1]
            cur_neighbors = list(graph.neighbors(cur))
            if len(cur_neighbors) > 0:
                if len(walk) == 1:
                    walk.append(random.choice(cur_neighbors))
                else:
                    prev = walk[-2]
                    probabilities = self.get_edge_probabilities(graph,prev, cur, cur_neighbors)
                    next_node = random.choices(cur_neighbors, weights=probabilities, k=1)[0]
                    walk.append(next_node)
            else:
                break
        return walk

    def get_edge_probabilities(self, graph,prev, cur, neighbors):
        probs = []
        for nbr in neighbors:
            if nbr == prev:
                probs.append(1 / self.p)
            elif nbr in set(graph.neighbors(prev)):
                probs.append(1)
            else:
                probs.append(1 / self.q)
        prob_sum = sum(probs)
        return [p / prob_sum for p in probs]

    def DyWord2Vec(self, walks, nodes_embedding):

        # 初始化模型，如果Word2Vec模型还不存在
        if not hasattr(self, 'word2vec_model'):
            self.word2vec_model = Word2Vec(vector_size=self.dimension, window=5, sg=1, workers=4, min_count=1)
            self.word2vec_model.build_vocab(walks, progress_per=1000)

        # 使用已有的节点嵌入来初始化Word2Vec权重
        self.word2vec_model.wv.vectors = nodes_embedding
        # 进行模型的训练
        self.word2vec_model.train(walks, total_examples=self.word2vec_model.corpus_count, epochs=5)
        # 更新并返回节点嵌入
        nodes_embedding = self.word2vec_model.wv.vectors
        return nodes_embedding




def dynodes_embedding(graphs, dimension, p, q, walk_length, num_walks,sig,k):
    dw=DyyNode2Vec(graphs=graphs, dimension=dimension, p=p, q=q, walk_length=walk_length, num_walks=num_walks,sig=sig,k=k)
    nodes_embedding=Node2vec.nodetovec(graphs[0], dimension, p, q, walk_length, num_walks)
    for i in range(1,len(graphs)):
        walks=dw.generate_walks(graphs[i-1],graphs[i])
        nodes_embedding=dw.DyWord2Vec(walks,nodes_embedding)
    return nodes_embedding















