import numpy as np
from scipy.spatial.distance import pdist, squareform

class TSNE:
    def __init__(self, n_components=2, perplexity=30, learning_rate=200, n_iter=1000):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit_transform(self, X):
        distances = squareform(pdist(X, metric='euclidean'))

        np.random.seed(42)

        n_samples = X.shape[0]
        Y = np.random.randn(n_samples, self.n_components)
        dY = np.zeros((n_samples, self.n_components))
        gains = np.ones((n_samples, self.n_components))

        P = self._compute_similarity(distances)

        for i in range(self.n_iter):
            sum_Y = np.sum(np.square(Y), axis=1)
            num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y))
            np.fill_diagonal(num, 0)
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)

            PQ = P - Q
            for j in range(n_samples):
                dY[j, :] = np.sum(np.tile(PQ[:, j] * num[:, j], (self.n_components, 1)).T * (Y[j, :] - Y), axis=0)

            gains = (gains + 0.2) * ((dY > 0) != (i < 250))
            gains[gains < 0.01] = 0.01
            Y = Y - self.learning_rate * dY / n_samples
            Y = Y - np.mean(Y, axis=0)

            if (i + 1) % 100 == 0:
                error = np.sum(P * np.log(P / Q))
                print(f"Iteration {i + 1}/{self.n_iter}, Error: {error}")

        return Y