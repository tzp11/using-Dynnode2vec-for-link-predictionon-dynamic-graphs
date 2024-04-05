import networkx as nx
from node2vec import Node2Vec
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from dataProcess import GraphData, GraphToLabel,Node2Edge


def train_and_evaluate_svm(edges_embedding, labels, test_labels):
    edges_embedding = np.array(edges_embedding)
    labels = np.array(labels)
    labels_test = np.array(test_labels)

    # svm=SVC(kernel='rbf')
    svm = SVC(kernel='linear')
    svm.fit(edges_embedding,labels)
    labels_pred=svm.predict(edges_embedding)
    accuracy=accuracy_score(labels_test,labels_pred)
    return accuracy