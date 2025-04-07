import torch

torch.manual_seed(1234)

def perceptron_data(n_points = 300, noise = 0.2):
    
    y = torch.arange(n_points) >= int(n_points/2)
    X = y[:, None] + torch.normal(0.0, noise, size = (n_points,2))
    X = torch.cat((X, torch.ones((X.shape[0], 1))), 1)

    # convert y from {0, 1} to {-1, 1}
    #y = 2*y - 1

    return X, y

X, y = perceptron_data(n_points = 300, noise = 0.2)

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

class LogisticRegression(LinearModel):

    """
    Computes the sigmoid of the input score.

    Args:
        s (Tensor): A PyTorch tensor containing the input scores.

    Returns:
        Tensor: The element-wise sigmoid of the input tensor.
    """
    def sig(self, s):
        return 1 / (1 + torch.exp(-s))

    """
    Calculates the logistic loss using the current weight vector.

    Args:
        X (Tensor): A PyTorch tensor of input features with shape (n_samples, n_features).
        y (Tensor): A PyTorch tensor of binary labels with shape (n_samples,).

    Returns:
        Tensor: The mean logistic loss across all samples.
    """
    def loss(self, X, y):

        s = self.score(X)  # Compute scores for all samples
        sig_s = self.sig(s)  # Apply sigmoid function
        
        # Compute logistic loss
        loss = -y * torch.log(sig_s) - (1 - y) * torch.log(1 - sig_s)

        return loss.mean()  # Return the average loss
    
    """
    Calculates the gradient of the empirical risk (logistic loss) 
    for use in logistic regression optimization.

    Args:
        X (Tensor): A PyTorch tensor of input features with shape (n_samples, n_features).
        y (Tensor): A PyTorch tensor of binary labels with shape (n_samples,).

    Returns:
        Tensor: The average gradient of the logistic loss with respect to the weights, 
        with shape (n_features,).
    """
    def grad(self, X, y): 
        y = y.float()  # Ensure y is float
        s = self.score(X)  
        sig_s = self.sig(s)  

        gradient = (sig_s - y).unsqueeze(1) * X  # (300,1) * (300,3) → (300,3)
        return gradient.mean(dim=0)  # Return average gradient across samples
    
class GradientDescentOptimizer():

    def __init__(self, model):
        self.model = model 
        self.prev = None

    """
    Performs one step of logistic regression parameter update using 
    gradient descent with optional momentum.

    This method updates the model's weight vector based on the gradient 
    of the logistic loss and optionally includes a momentum term for smoother convergence.

    Args:
        X (Tensor): A PyTorch tensor of input features with shape (n_samples, n_features).
        y (Tensor): A PyTorch tensor of binary labels with shape (n_samples,).
        alpha (float): The learning rate.
        Beta (float): The momentum coefficient. When Beta > 0, a momentum term is included.

    Returns:
        None
    """
    def step(self, X, y, alpha, Beta):
        
        # handles the case where the weight vector w has not been specified
        if self.model.w is None:
            self.model.w = torch.randn(X.shape[1], requires_grad=False) * 0.01  # small random init
        
        # calculates the first step without the momentum term, as there was no previous weight vector
        if self.prev == None:
            # store the current weight vector before it is updated
            self.prev = self.model.w
            
            # update the weight vector based on a given alpha and the gradient
            self.model.w = self.model.w - (alpha * self.model.grad(X, y))
        # calculates the next step including the momentum term
        else:
            # pull out the stored previous weight vector
            prev = self.prev

            # store the current weight vector
            self.prev = self.model.w

            # update the weight vector based on the given alpha, Beta, gradient, and momentum term
            self.model.w = self.model.w - (alpha * self.model.grad(X, y)) + (Beta *(self.model.w - prev))
