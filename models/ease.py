import numpy as np
from scipy.linalg import cho_factor, cho_solve


class EASE:
    def __init__(self, reg=100):
        self.reg = reg

    def fit(self, data):
        self.x = data['positive']
        gram = (self.x.T @ self.x).toarray()
        gram += self.reg * np.eye(gram.shape[0])
        inverse = cho_solve(cho_factor(gram), np.eye(gram.shape[0]))
        self.weights = -inverse / np.diag(inverse)
        np.fill_diagonal(self.weights, 0)
        return self

    def score(self, users):
        return self.x[users] @ self.weights
