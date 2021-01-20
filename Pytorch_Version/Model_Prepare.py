# %%
import torch
# %%
#number of features
num_features = 2

# Initializing Model Parameters
## weight: x   bias: y
x, y = torch.normal(0, 0.01, size=(num_features, 1), requires_grad=True), torch.zeros(size=(1, ), requires_grad=True)


# Define the Model
#Remember in our problem, x, y are parameters and a, b and data
lin = lambda a: torch.mm(a, x) + y
def sigmoid(a):
    return 1/(1+torch.exp(-a))

# Define the Loss Function: logit_loss + \lambda * l2_penalty
## Define penalty
def l2_penalty(x):
    return torch.sum(x.pow(2)) / 2
def logit_loss(x, y):
    return torch.mean(torch.log(1 + torch.exp(-b*(torch.mm(a, x)+y))))

# Define the Optimization Algorithm (SGD here)
## basic SGD with fixed step size (so called learning rate in most cases)
def sgd(params, lr, batch_size):
    """Mini-batch Version"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size  #这里感觉有点问题，明天搞！

# Training
