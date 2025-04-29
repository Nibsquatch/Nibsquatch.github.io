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

class KernelLogisticRegression():

    def __init__(self, kernel, lam = .1, gamma = 1):
        self.a = None
        self.Xt = None 
        self.K = kernel
        self.lamb = lam
        self.gamma = gamma

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
    Computes the score (log-odds) for each input sample in sparse kernelized logistic regression.

    For each test input x in X, the score is given by:
        s(x) = sum_i a_i * k(x, x_t,i)

    or, in matrix form:
        s = K(X, X_t)^T @ a

    where:
        - K is a positive-definite kernel function,
        - X_t is the training data used to fit the model (shape: [n_train, n_features]),
        - a is the learned coefficient vector (shape: [n_train, 1]),
        - X is the test data (shape: [n_test, n_features]).

    Parameters:
        X (np.ndarray): Test data of shape (n_test, n_features).

    Returns:
        np.ndarray: Score vector of shape (n_test, 1), where each entry corresponds to the 
                    log-odds of the positive class for a test sample.
    """
    def score(self, X):
        K_xt = self.K(X, self.Xt, self.gamma)   # shape: [n_test, n_train]
        K_xt_T = K_xt.T                         # shape: [n_train, n_test]
        return K_xt @ self.a                    # shape: (n_test, 1)

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
        return (self.sig(s) >= .5 ).float()
    
        
    def loss(self, X, y):
        """
        Computes the sparse kernel logistic regression loss function:
            L(a) = -(1/m) * sum_j [ y_j * log(sigmoid(s_j)) + (1 - y_j) * log(1 - sigmoid(s_j)) ] 
                + λ * ||a||_1

        where:
            - s_j = score for sample j, computed via kernel,
            - sigmoid(s) = 1 / (1 + exp(-s)),
            - a is the learned coefficient vector,
            - λ is the regularization parameter.

        Parameters:
            X (np.ndarray): Training data of shape (m, n_features).
            y (np.ndarray): Binary labels of shape (m,) or (m, 1).
            lam (float): Regularization strength (λ).

        Returns:
            float: The total loss value.
        """
        # compute scores
        s = self.score(X)

        # Ensure y has the correct shape
        y = y.reshape(-1, 1)
    
        # Binary cross-entropy loss
        log_loss = -torch.mean(y * torch.log(self.sig(s)) + (1 - y) * torch.log(1 - self.sig(s)))

        # L1 regularization term
        return self.lamb * torch.linalg.norm(self.a, ord=1) + log_loss
    
    """
    Computes the gradient of the logistic loss with respect to the dual coefficients (a)
    in kernelized logistic regression.

    The gradient is computed as:
        ∇L = (1 / m) * K(X, X_t)^T @ (sigmoid(s) - y)

    where:
        - K(X, X_t) is the kernel matrix between input data X and training data X_t,
        - s = K(X, X_t)^T @ a is the score,
        - sigmoid(s) gives the predicted probabilities,
        - y is the binary label vector.

    Parameters:
    ----------
    X : torch.Tensor
        Input features of shape (m, n_features), where m is the number of examples.
    y : torch.Tensor
        Binary target labels of shape (m,) or (m, 1).

    Returns:
    -------
    torch.Tensor
        Gradient of the loss with respect to the dual coefficients, of shape (n_train, 1),
        where n_train is the number of training samples used to fit the model.
    """
    def grad(self, X, y): 
        # ensure {1,1}
        y = y.view(-1, 1).float()
        
        # compute scores and sigmoid scores
        s = self.score(X)
        sig_s = self.sig(s)
        
        # check mismatch
        error = sig_s - y  # shape: (n_samples, 1)
        
        # apply the Kernel
        K = self.K(X, self.Xt, self.gamma)  # shape: (n_samples, n_train)
        
        return K.T @ error / X.shape[0]  # shape: (n_train, 1)
    
    """
    Performs one step of logistic regression parameter update using 
    gradient descent

    This method updates the model's weight vector based on the gradient 
    of the logistic loss

    Args:
        X (Tensor): A PyTorch tensor of input features with shape (n_samples, n_features).
        y (Tensor): A PyTorch tensor of binary labels with shape (n_samples,).
        alpha (float): The learning rate.

    Returns:
        None
    """
    def step(self, X, y, alpha):

        # update the weight vector based on the given alpha and L1 regularization term
        l1_grad = self.lamb * torch.sign(self.a)

        self.a = self.a - alpha*(self.grad(X, y) + l1_grad)

    """
    Trains the model on the provided data using gradient-based updates.

    Parameters:
    ----------
    X : array-like or tensor
        Input features for training.
    y : array-like or tensor
        Target values corresponding to X.
    m_epochs : int
        Number of training epochs (full passes over the dataset).
    alpha : float
        Learning rate used to update model parameters.

    Returns:
    -------
    None
    """
    def fit(self, X, y, m_epochs, alph):

        # handles the case where the weight vector w has not been specified
        if self.a is None:
            self.Xt = X  # store training data for kernel computation
            n_train = X.shape[0]
            self.a = torch.zeros(n_train, 1)
        
        #iterables
        loss = 1.0
        epochs = 0

        # iterate until the max number of epochs has been reached or the loss reaches 0
        while epochs < m_epochs and loss > 0:

            loss = self.loss(X, y)

            # only this line actually changes the parameter value
            self.step(X, y, alpha = alph)
            
            # iterate
            epochs += 1