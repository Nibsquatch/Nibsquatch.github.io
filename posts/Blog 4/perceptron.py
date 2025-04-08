import torch

torch.manual_seed(1234)

def perceptron_data(n_points = 300, noise = 0.2):
    
    y = torch.arange(n_points) >= int(n_points/2)
    X = y[:, None] + torch.normal(0.0, noise, size = (n_points,2))
    X = torch.cat((X, torch.ones((X.shape[0], 1))), 1)

    # convert y from {0, 1} to {-1, 1}
    y = 2*y - 1

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

class Perceptron(LinearModel):

    def loss(self, X, y):
        """
        Compute the misclassification rate. A point i is classified correctly if it holds that s_i*y_i_ > 0, where y_i_ is the *modified label* that has values in {-1, 1} (rather than {0, 1}). 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}
        
        HINT: In order to use the math formulas in the lecture, you are going to need to construct a modified set of targets and predictions that have entries in {-1, 1} -- otherwise none of the formulas will work right! An easy to to make this conversion is: 
        
        y_ = 2*y - 1
        """

        # return the proportion
        return (1.0*((self.score(X) * (2 * y -1)) <= 0)).mean()

    """
    Compute the gradient update for a mini-batch using the Perceptron rule.

    Args:
        X_batch (Tensor): A batch of input features of shape (batch_size, d).
        y_batch (Tensor): Corresponding labels of shape (batch_size,).

    Returns:
        Tensor: The average update vector of shape (d,).
    """
    def grad(self, X_batch, y_batch):
        # Convert labels to {-1, 1}
        y_mod = 2 * y_batch - 1

        # Compute raw scores
        scores = self.score(X_batch)

        # Find where prediction is wrong: score * label <= 0
        mask = (scores * y_mod) <= 0
        
        # Compute the perceptron updates for wrong examples and average them
        if mask.sum() == 0: # no mismatches were found
            return torch.zeros_like(self.w)  # No update needed
        
        else:
            # Select misclassified examples
            misclassified_X = X_batch[mask]
            misclassified_y = y_mod[mask].unsqueeze(1)

            # Compute updates
            updates = misclassified_y * misclassified_X
            averaged_update = updates.mean(dim=0)

            return averaged_update

        return averaged_update

class PerceptronOptimizer:

    def __init__(self, model):
        self.model = model 
    
    def step(self, X, y, alpha):
        
        """
        Compute one step of the perceptron update using the feature matrix X 
        and target vector y. 
        """

        update = (alpha / len(X)) * self.model.grad(X, y)
        
        if update is not None:  # Ensure meaningful update
            self.model.w += update  # Correct perceptron weight update