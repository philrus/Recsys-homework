import numpy as np
from scipy.sparse import csr_matrix


class ItemKNN:
    def __init__(self, neighbors=150, shrinkage=25):
        self.neighbors = neighbors
        self.shrinkage = shrinkage

    def fit(self, data):
        self.x = data['positive']
        gram = (self.x.T @ self.x).tocsr()
        norms = np.sqrt(gram.diagonal())
        gram.setdiag(0)
        gram.eliminate_zeros()
        rows, cols, values = [], [], []
        for i in range(gram.shape[0]):
            start, end = gram.indptr[i:i + 2]
            items, counts = gram.indices[start:end], gram.data[start:end]
            scores = counts / (norms[i] * norms[items])
            scores *= counts / (counts + self.shrinkage)
            order = np.lexsort((items, -scores))[:self.neighbors]
            rows.extend([i] * len(order))
            cols.extend(items[order])
            values.extend(scores[order])
        self.weights = csr_matrix((values, (rows, cols)), shape=gram.shape)
        return self

    def score(self, users):
        return (self.x[users] @ self.weights).toarray()
