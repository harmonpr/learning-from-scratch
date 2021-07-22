import numpy as np

class K_NN():
    ''' A class use to implement k-Nearest neighbors
    
    Attributes
    ----------
    n_neighbors : int
        the number of neighbors
    X : float
        the input array 
    y : float
        the output array
    T : float
        the input array that want to be predicted
    y_pred : float
        the prediction array
        
    Methods
    -------
    fit(X, y)
        Set the input and output array
    predict(T)
        Predict from trained model and return the prediction, y_pred.
        The process of this method are below,
            1. Compute the distance between one element of T and X
            2. Find the nearest distance based on the number of neighbors
            3. Get the y_pred by average of y of nearest neighbor
            4. Repeat process 1-3 for each element of T
    '''
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
    
    def fit(self, X, y):
        self.X = X
        self.y = y
    
    def predict(self, T):
        m = len(T)
        self.y_pred = np.zeros(m)
        
        for i in range(m):
            diff = self.X.reshape(-1, 1) - T[i]
            D = (diff**2).sum(1)
            idx = np.argsort(D.T)[:self.n_neighbors]
            self.y_pred[i] = self.y[idx].mean()
        
        return self.y_pred