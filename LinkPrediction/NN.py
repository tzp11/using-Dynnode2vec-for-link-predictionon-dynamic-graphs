import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Input

from sklearn.metrics import accuracy_score


def train_and_evaluate_nn(edges_embedding, labels, test_labels):

    """
    This function trains a neural network on the given edges embedding and labels and evaluates it on the test set.

    Parameters:
    edges_embedding (numpy array): The embedding of the edges.
    labels (numpy array): The labels of the nodes.
    test_labels (numpy array): The labels of the test set.

    Returns:
    accuracy (float): The accuracy of the model on the test set.
    """

    edges_embedding = np.array(edges_embedding)
    labels = np.array(labels)
    test_labels = np.array(test_labels)
    model=Sequential()
    model.add(Input(shape=(edges_embedding.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(edges_embedding, labels, epochs=10, batch_size=32, verbose=0)
    labels_pred = model.predict(edges_embedding)
    accuracy = accuracy_score(test_labels, np.round(labels_pred))
    return accuracy


