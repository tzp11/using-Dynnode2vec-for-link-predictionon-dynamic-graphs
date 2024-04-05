

from LinkPrediction import NN, SVM
from dataProcess import GraphData, GraphToLabel, Node2Edge
from GraphEmbedding import DyyNode2Vec

# 使用示例
file_path = "D:\\qqqqqqqqqqqqqqqqqq\\pycharm\\work\\Thsis\\dataProcess\\data"
num_partitions = 5
graphs = GraphData.read_file_and_process_data(file_path, num_partitions)
# 对子图graph[2]执行node2vec算法
graph = graphs[2]
#print(graph.nodes())
#print(graph.edges())

#将graphs的0，1，2，部分提取出来，组成新的pgraphs
pgraphs = [graphs[0],graphs[1],graphs[2]]

#普通dynnode2vec时sig=1
#统计所有新图节点的dynnode2vec则sig=0
#普通dynnode2vec对游走次数加上k的限制，sig=2
nodes_embedding =DyyNode2Vec.dynodes_embedding(pgraphs,32,1,2,100,200,2,0.5)


#print(nodes_embedding)
labels = GraphToLabel.GraphToLabel(graphs[2])
test_labels = GraphToLabel.GraphToLabel(graphs[3])
# 通过节点向量获得每个边的嵌入向量
edges_embedding = Node2Edge.WeightL2(nodes_embedding)


# 建立神经网络或SVM模型进行分类
accuracy=NN.train_and_evaluate_nn(edges_embedding,labels,test_labels)
#accuracy=SVM.train_and_evaluate_svm(edges_embedding,labels,test_labels)


print("Accuracy:",accuracy)


