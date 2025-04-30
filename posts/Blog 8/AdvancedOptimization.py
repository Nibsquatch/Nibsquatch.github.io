import torch

class LinearModel:

    def __init__(self):
        self.w = None 

    def score(self, X):
        """
        Compute the linear scores for input features.

        Args:
            X (Tensor): Feature matrix of shape (n_samples, n_features).

        Returns:
            Tensor: Score vector of shape (n_samples,).
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))

        return X @ self.w

    def predict(self, X):
        """
        Make binary predictions based on the sign of the score.

        Args:
            X (Tensor): Feature matrix of shape (n_samples, n_features).

        Returns:
            Tensor: Binary predictions in {0.0, 1.0}, shape (n_samples,).
        """
        s = self.score(X)
        return (s >= 0).float()


class LogisticRegression(LinearModel):
    def sig(self, s):
        """
        Apply the sigmoid function element-wise.

        Args:
            s (Tensor): Input tensor.

        Returns:
            Tensor: Sigmoid activation of input tensor.
        """
        return 1 / (1 + torch.exp(-s))

    def loss(self, X, y):
        """
        Compute the average logistic loss over the dataset.

        Args:
            X (Tensor): Feature matrix of shape (n_samples, n_features).
            y (Tensor): Binary labels of shape (n_samples,).

        Returns:
            Tensor: Scalar representing mean logistic loss.
        """
        s = self.score(X)
        sig_s = self.sig(s)
        loss = -y * torch.log(sig_s) - (1 - y) * torch.log(1 - sig_s)
        return loss.mean()

    def grad(self, X, y): 
        """
        Compute the gradient of the logistic loss.

        Args:
            X (Tensor): Feature matrix of shape (n_samples, n_features).
            y (Tensor): Binary labels of shape (n_samples,).

        Returns:
            Tensor: Gradient vector of shape (n_features,).
        """
        y = y.float()
        s = self.score(X)
        sig_s = self.sig(s)
        gradient = (sig_s - y).unsqueeze(1) * X
        return gradient.mean(dim=0)

    def hessian(self, X):
        """
        Compute the Hessian matrix for logistic regression.

        Args:
            X (Tensor): Feature matrix of shape (n_samples, n_features).

        Returns:
            Tensor: Hessian matrix of shape (n_features, n_features).
        """
        s = X @ self.w  
        sig = self.sig(s)  
        D = sig * (1 - sig)  
        weighted_X = X * D.unsqueeze(1)
        H = X.T @ weighted_X
        return H


class NewtonOptimizer():
    def __init__(self, model):
        self.model = model 
        self.prev = None

    def step(self, X, y, alpha):
        """
        Perform one Newton update step for logistic regression.

        Args:
            X (Tensor): Feature matrix of shape (n_samples, n_features).
            y (Tensor): Binary labels of shape (n_samples,).
            alpha (float): Learning rate.

        Returns:
            None
        """
        if self.model.w is None:
            self.model.w = torch.randn(X.shape[1]) * 0.01
        else:
            # Compute gradient and inverse Hessian
            grad = self.model.grad(X, y)
            H_inv = torch.linalg.inv(self.model.hessian(X))

            # Newton update step
            self.model.w = self.model.w - alpha * (H_inv @ grad)

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