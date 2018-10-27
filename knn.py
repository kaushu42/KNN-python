import numpy as np

class KNN(object):
    '''
        KNN Classifier class.
        Performs a nearest neighbor based classificationself.

        Parameters:
            k: the number of nearest neighbors to use
    '''
    def __init__(self, k=5):
        self.k = k

    def train(self, X, y):
        # Training for KNN is just memorizing the data
        self.x = X
        self.y = y

    def predict(self, x):
        # Create lists to store the distances of each point
        distances = []

        for i in range(len(self.x)):
            distance = 0
            # Calculating the euclidean distance
            for j in range(len(self.x[i])):
                distance += (self.x[i, j] - x[j])**2
            distance = np.sqrt(distance)
            distances.append(distance)
        # We will now concatenate the distance and labels for each pointself
        # We can then get the labels for each point when we sort the data by distance
        distances = np.asarray(distances).reshape(-1, 1)
        pred = np.hstack([distances, self.y.reshape(-1, 1)])
        pred = pred[pred[:, 0].argsort()][:self.k, 1].astype(int)

        # Get the maximum number of neighbors
        (v, c) = np.unique(pred, return_counts=True)
        ind = np.argmax(c)
        return v[ind]
