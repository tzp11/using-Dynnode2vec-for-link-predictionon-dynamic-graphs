

from LinkPrediction import NN, SVM
from dataProcess import GraphData, GraphToLabel, Node2Edge
from GraphEmbedding import DeepWalk

# 使用示例
file_path = "D:\\qqqqqqqqqqqqqqqqqq\\pycharm\\work\\Thsis\\dataProcess\\data"
num_partitions = 5
graphs = GraphData.read_file_and_process_data(file_path, num_partitions)
# 对子图graph[2]执行node2vec算法
graph = graphs[2]
print(graph.nodes())
print(graph.edges())



nodes_embedding =DeepWalk.deepwalk_embedding(graph, 32, 100, 200)

#print(nodes_embedding)
labels = GraphToLabel.GraphToLabel(graphs[2])
test_labels = GraphToLabel.GraphToLabel(graphs[3])
# 通过节点向量获得每个边的嵌入向量
edges_embedding = Node2Edge.WeightL2(nodes_embedding)


# 建立神经网络或SVM模型进行分类
accuracy=NN.train_and_evaluate_nn(edges_embedding,labels,test_labels)
#accuracy=SVM.train_and_evaluate_svm(edges_embedding,labels,test_labels)


print("Accuracy:",accuracy)


