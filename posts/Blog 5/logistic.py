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


    def grad(self, X, y):

        y_i = 2 * y - 1  # Convert {0,1} labels to {-1,1}

        s_i = self.score(X)  # Compute score

        if (s_i * y_i).item() <= 0:  # Misclassified point
            return y_i * X.squeeze(0)  # Ensure correct shape
    
        return torch.zeros_like(self.w)  # No update if correctly classified
        

class PerceptronOptimizer:

    def __init__(self, model):
        self.model = model 
    
    def step(self, X, y):
        """
        Compute one step of the perceptron update using the feature matrix X 
        and target vector y. 
        """

        loss = self.model.loss(X, y)

        update = self.model.grad(X, y)
        
        if update is not None:  # Ensure meaningful update
            self.model.w += update  # Correct perceptron weight update

class LogisticRegression(LinearModel):

    # returns the sigmoid of the score s
    def sig(self, s):
        return 1 / (1 + torch.exp(-s))

    # calculates the logistic loss with the current weight vector w
    def loss(self, X, y):

        s = self.score(X)  # Compute scores for all samples
        sig_s = self.sig(s)  # Apply sigmoid function
        
        # Compute logistic loss
        loss = -y * torch.log(sig_s) - (1 - y) * torch.log(1 - sig_s)

        return loss.mean()  # Return the average loss
    
    # calculates the gradient of the empirical risk for logistic regression optimization
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

    # computes one step of a Logistic Regression update using gradient descent
    def step(self, X, y, alpha, Beta):
        
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
