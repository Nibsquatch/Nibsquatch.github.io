import torch

class LinearModel:

    def __init__(self):
        self.w = None 

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))

        # your computation here: compute the vector of scores s
        return X@self.w

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        # your implementation here
        s = self.score(X)
        return (s >= 0).float()

class MyLinearRegression(LinearModel):

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction 
        for the ith data point is a real-valued score.

        ARGUMENTS: 
            X (torch.Tensor): The feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat (torch.Tensor): Predicted scores (real values), shape (n, 1)
        """
        # your implementation here
        return self.score(X)
    
    def loss(self, X, y):
        """
        Computes the Mean Squared Error (MSE) loss between predictions and true targets.

        Arguments:
            X (tensor): Feature matrix of shape (n_samples, n_features)
            y (tensor): Target vector of shape (n_samples, )

        Returns:
            float (tensor): Mean Squared Error
        """
        y_pred = self.predict(X)
        # Ensure both are 1D for fair comparison
        y_pred = y_pred.squeeze()
        y = y.squeeze()
        
        mse = torch.mean((y_pred - y) ** 2)
        return mse
    
class OverParameterizedLinearRegressionOptimizer():

    def __init__(self, model):
        self.model = model 
    """
    Fits the linear regression model to the training data using the 
    Moore-Penrose pseudoinverse.

    In overparameterized settings (when the number of features p > number of samples n),
    the closed-form solution for the optimal weights becomes:

        w* = X⁺y

    where X⁺ is the Moore-Penrose pseudoinverse of X. This ensures a valid solution 
    even when XᵀX is not invertible.

    Parameters:
    -----------
    X : torch.Tensor
        Feature matrix of shape (n_samples, n_features).

    y : torch.Tensor
        Target vector of shape (n_samples,) or (n_samples, 1).

    Sets:
    -----
    self.model.w : torch.Tensor
        Optimal weights computed as the pseudoinverse of X multiplied by y.
    """
    def fit(self, X, y):
        X_pinv = torch.linalg.pinv(X)
        self.model.w = X_pinv @ y
