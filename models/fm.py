import numpy as np
from scipy.special import expit


class FM:
    def __init__(self, factors=32, lr=0.05, reg=1e-4, epochs=6,
                 batch_size=2048, metadata=True, seed=42):
        self.factors = factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.metadata = metadata
        self.seed = seed

    def gradients(self, ids, labels):
        vectors = self.v[ids]
        summed = vectors.sum(axis=1)
        logits = self.bias + self.w[ids].sum(axis=1)
        logits += 0.5 * ((summed ** 2).sum(axis=1) - (vectors ** 2).sum(axis=(1, 2)))
        error = (expit(logits) - labels) / len(labels)
        unique, inverse = np.unique(ids, return_inverse=True)
        dw = np.zeros(len(unique))
        dv = np.zeros((len(unique), self.factors))
        grad_w = error[:, None] + self.reg * self.w[ids] / len(labels)
        grad_v = error[:, None, None] * (summed[:, None, :] - vectors)
        grad_v += self.reg * vectors / len(labels)
        np.add.at(dw, inverse.ravel(), grad_w.ravel())
        np.add.at(dv, inverse.ravel(), grad_v.reshape(-1, self.factors))
        return error.sum(), unique, dw, dv

    def fit(self, data):
        rng = np.random.default_rng(self.seed)
        self.features = data['features'] if self.metadata else data['features'][:, :1]
        n_features = self.features.max() + 1
        self.w = np.zeros(n_features)
        self.v = rng.normal(0, 0.01, (n_features, self.factors))
        self.bias = 0.0
        accum_w, accum_v = np.zeros_like(self.w), np.zeros_like(self.v)
        accum_bias = 0.0
        events = data['events']
        users, items = events.u.to_numpy(), events.i.to_numpy()
        labels = (events.rating.to_numpy() >= 4).astype(float)
        n_items = data['seen'].shape[1]
        known = users * n_items + items
        negative_users = users[labels == 1]
        for _ in range(self.epochs):
            negatives = rng.integers(n_items, size=len(negative_users))
            invalid = np.isin(negative_users * n_items + negatives, known)
            while invalid.any():
                negatives[invalid] = rng.integers(n_items, size=invalid.sum())
                invalid = np.isin(negative_users * n_items + negatives, known)
            epoch_users = np.r_[users, negative_users]
            epoch_items = np.r_[items, negatives]
            epoch_labels = np.r_[labels, np.zeros(len(negatives))]
            order = rng.permutation(len(epoch_users))
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                ids = np.column_stack([epoch_users[batch], self.features[epoch_items[batch]]])
                db, unique, dw, dv = self.gradients(ids, epoch_labels[batch])
                accum_bias += db ** 2
                accum_w[unique] += dw ** 2
                accum_v[unique] += dv ** 2
                self.bias -= self.lr * db / (np.sqrt(accum_bias) + 1e-8)
                self.w[unique] -= self.lr * dw / (np.sqrt(accum_w[unique]) + 1e-8)
                self.v[unique] -= self.lr * dv / (np.sqrt(accum_v[unique]) + 1e-8)
        vectors = self.v[self.features]
        self.item_vectors = vectors.sum(axis=1)
        self.item_bias = self.w[self.features].sum(axis=1)
        self.item_bias += 0.5 * ((self.item_vectors ** 2).sum(axis=1) - (vectors ** 2).sum(axis=(1, 2)))
        return self

    def score(self, users):
        return (self.bias + self.w[users, None] + self.item_bias[None, :]
                + self.v[users] @ self.item_vectors.T)
