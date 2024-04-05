from node2vec import Node2Vec


def nodetovec(G,dimensions,p,q,walk_length,num_walks):

    """
    This function is used to generate node embeddings using Node2Vec algorithm.

    Parameters:
    G (networkx graph): The input graph.
    dimensions (int): The number of dimensions for the embedding.
    p (float): The probability of returning to the previous node in the random walk.
    qwalk_length (int): The length of the random walks.
    num_walks (int): The number of random walks to start at each node.
    workers (int): The number of workers to use for parallel processing.

    Returns:
    embedding (numpy array): The node embeddings.
    """
    node2vec=Node2Vec(G,
                      dimensions=dimensions,
                      p=p,
                      q=q,
                      walk_length=walk_length,
                      num_walks=num_walks,
                      )
    model=node2vec.fit(window=3, min_count=1, batch_words=4)
    nodes_embedding = model.wv.vectors
    return nodes_embedding

