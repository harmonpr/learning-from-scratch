import numpy as np

class LinearReg():
    ''' A class use to implement Linear Regression algorithm
    
    Attributes
    ----------
    intercept : float
        the intercept of the line equation
    coef : float
        the slope of the line equation
    X : float
        the input array
    y : float
        the output array
    y_pred : float
        the prediction array
        
    Methods
    -------
    dimX(X)
        Transform X if its shape (len(X),) become (len(X), 1)
    fit(X, y)
        Train and return the intercept and coefficient(slope).
        The calculation using normal function,
                    $$ \theta = (X^T X)^{-1} \cdot (X^T y) $$
        For theta[0] we get the intercept and theta[1:] we get 
        the coefficient(slope)
    predict(X)
        Predict from trained model and return y_pred. The prediction using
        straight line model,
                            $$ y = a + bX $$
        where y: y_pred, a: the intercept, b: the slope(coef), and the input
        array X
    meanSquaredError(y, y_pred)
        Return the mean squared error between y = y_test and y_prediction
    meanAbsoluteError(y, y_pred)
        Return the mean absolute error between y = y_test and y_prediction
    r2Score(y, y_pred)
        Return the r^2 score between y = y_test and y_prediction
    '''
    
    def __init__(self):
        self.intercept = None
        self.coef = None
        
    def dimX(self, X):
        ''' Transform X if its shape (len(X),) become (len(X), 1) '''
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)    
        return X
        
    def fit(self, X, y):
        ''' Train and return the intercept and coefficient(slope) '''
        X = self.dimX(X)     
        m = X.shape[0]
        X = np.c_[np.ones(m), X]
        theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
        self.intercept = theta[0]
        self.coef = theta[1:]
    
    def predict(self, X):
        ''' Predict from trained model and return y_pred '''
        X = self.dimX(X)   
        self.y_pred = np.dot(X, self.coef) + self.intercept
        return self.y_pred

    def meanSquaredError(self, y, y_pred):
        ''' Return the mean squared error between y = y_test and y_prediction '''
        return ((y_pred - y)**2).mean()
    
    def meanAbsoluteError(self, y, y_pred):
        ''' Return the mean absolute error between y = y_test and y_prediction '''
        return (np.abs(y_pred - y)).mean()
    
    def r2Score(self, y, y_pred):
        ''' Return the r^2 score between y = y_test and y_prediction '''
        y_min_yhat = ((y - y_pred)**2).mean()
        y_min_ybar = ((y - y.mean())**2).mean()
        r2 = 1 - (y_min_yhat / y_min_ybar)
        return r2