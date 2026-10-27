import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix


class ALS:
    def __init__(self, factors=32, reg=1, alpha=20, iterations=12, seed=42):
        self.factors = factors
        self.reg = reg
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed

    def target(self, data):
        return (data['events'].rating.to_numpy() >= 4).astype(float)

    def update(self, target, fixed):
        base = fixed.T @ fixed + self.reg * np.eye(self.factors)
        factors = np.zeros((target.shape[0], self.factors))
        for row in range(target.shape[0]):
            start, end = target.indptr[row:row + 2]
            local = fixed[target.indices[start:end]]
            lhs = base + self.alpha * (local.T @ local)
            rhs = local.T @ ((1 + self.alpha) * target.data[start:end])
            factors[row] = cho_solve(cho_factor(lhs), rhs)
        return factors

    def fit(self, data):
        events = data['events']
        shape = data['seen'].shape
        # Нули сохраняем: у известных пар confidence тоже повышен.
        target = csr_matrix((self.target(data), (events.u, events.i)), shape=shape)
        transpose = target.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.items = rng.normal(0, 0.05, (shape[1], self.factors))
        for _ in range(self.iterations):
            self.users = self.update(target, self.items)
            self.items = self.update(transpose, self.users)
        return self

    def score(self, users):
        return self.users[users] @ self.items.T
