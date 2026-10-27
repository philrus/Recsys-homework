import numpy as np
from .ials import ALS


class CustomALS(ALS):
    def target(self, data):
        events = data['events']
        ratings = events.rating.to_numpy(float)
        users = events.u.to_numpy()
        rated = ratings > 0
        n_users = data['seen'].shape[0]
        mean = ratings[rated].mean()
        totals = np.bincount(users[rated], weights=ratings[rated], minlength=n_users)
        counts = np.bincount(users[rated], minlength=n_users)
        user_mean = (totals + 5 * mean) / (counts + 5)
        signal = np.clip((ratings - 3) / 4 + (ratings - user_mean[users]) / 4, -1, 1)
        signal = np.where(rated, signal, np.where(events.is_read, 0.25, 0.05))
        age = (data['cutoff'] - events.timestamp).dt.total_seconds().to_numpy() / 86400
        return signal * 2 ** (-age / 1095)
