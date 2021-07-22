import numpy as np

class PrincipalCA():
    ''' A class use to implement Principal Component Analysis
    
    Attributes
    ----------
    n_components : int
        the number of desired dimensions
    X : float
        the input array
    X_mean : float
        the standarized imput array
    components : float
        the sorted eigenvectors array with [:, n_components] elements
    explained_variance : float
        the sorted eigenvalues array with [:n_components] elements
    x_cov_matrix : float
        the covariance matrix of X_mean
    X_transform : float
        Reduced array dimension result
        
    Methods
    -------
    fit(X, n_components)
        Train the model and result the components and explained_variance
        The proccess inside this method as follow:
            1. Compute the standarized input array (X_mean)
            2. Form the covariance matrix from X_mean
            3. Find the eigenvalues and eigenvectors of the covariance matrix
            4. Sort descendingly the eigenvalues and associated eigenvectors 
               then get the index
            5. Get the components and explained_variance based on desired
               dimension (n_components)
    transform(X)
        transform input array (X) by X_mean
    '''
    def __init__(self, n_components):
        self.n_components = n_components
        self.X_mean = None
        self.components = None
        self.explained_variance = None
    
    def fit(self, X):
        ''' Train the model and result the components and explained_variance '''
        self.X_mean = X - X.mean(axis = 0)
        X_cov_matrix = np.cov(self.X_mean, rowvar = False)
        eigen_val, eigen_vec = np.linalg.eigh(X_cov_matrix)
        
        idx = np.argsort(eigen_val)[::-1]
        eigenval_sorted = eigen_val[idx]
        eigenvec_sorted = eigen_vec[:, idx]
        
        self.components = eigenvec_sorted[:, 0:self.n_components].T
        self.explained_variance = eigenval_sorted[:self.n_components]
    
    def transform(self):
        ''' transform input array (X) by X_mean '''
        X_transform = np.dot(self.components, self.X_mean.T).T
        return X_transform