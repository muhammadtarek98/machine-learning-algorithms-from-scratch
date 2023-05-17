import numpy as np

np.random.seed(42)


class Kmeans:
    @staticmethod
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    def __init__(self, epochs,k ):
        self.k=k
        self.X=None
        self.n_samples=None
        self.n_features=None
        self.epochs=epochs
        self.clusters=[[] for _ in range(self.k)]
        self.centroids=[]
    def _create_cluster(self,centroids):
        clusters=[[] for _ in range(self.k)]

    def predict(self,X):
        self.X=X
        self.n_samples,self.n_features=self.X.shape
        random_samples_idx=np.random.choice(self.n_samples,self.k,replace=False)
        self.centroids=[self.X[idx] for idx in random_samples_idx]

        for _ in range(self.epochs):
            pass
            #self.clusters=




