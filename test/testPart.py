import networkx as nx
import numpy as np
from node2vec import Node2Vec
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from dataProcess import GraphData, GraphToLabel,Node2Edge
from LinkPrediction import SVM,NN
graph=nx.karate_club_graph();
print(graph.number_of_nodes())
print(graph.number_of_edges())
node2vec = Node2Vec(graph,
                    dimensions=32,  # 嵌入维度
                    p=1,            # 回家参数
                    q=2,          # 外出参数
                    walk_length=100, # 随机游走最大长度
                    num_walks=200,  # 每个节点作为起始节点生成的随机游走个数
                    workers=4       # 并行线程数
                   )
model = node2vec.fit(window=3,    # Skip-Gram窗口大小
                     min_count=1,  # 忽略出现次数低于此阈值的节点（词）
                     batch_words=4 # 每个线程处理的数据量
                    )

nodes_embedding = model.wv.vectors
#print(type(nodes_embedding))
#print(nodes_embedding.shape)
labels =GraphToLabel.GraphToLabel(graph)
test_labels=GraphToLabel.GraphToLabel(graph)
#print(labels)
#通过节点向量获得每个边的嵌入向量
edges_embedding=Node2Edge.Node2Edge(nodes_embedding)
print(len(edges_embedding))
#accuracy=SVM.train_and_evaluate_svm(edges_embedding,labels,test_labels)
accuracy=NN.train_and_evaluate_nn(edges_embedding,labels,test_labels)
print("Accuracy:",accuracy)








