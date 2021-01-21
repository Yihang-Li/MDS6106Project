# %%
# import sys
# sys.path.append("/Users/yihangli/Documents/GitHub/MDS6106Project")
# %%
from Pytorch_Version.Data_Prepare import *
import torch
# %%
#number of features
num_features = 2

# Initializing Model Parameters
## weight: x   bias: y
def initia_params():
    x, y = torch.normal(0, 0.01, size=(num_features, 1), requires_grad=True), torch.zeros(size=(1, ), requires_grad=True)
    return x, y

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
    """Mini-batch Version
        Non-Averaged Version From d2l
    """
    with torch.no_grad():

        for param in params:
            param -= lr * param.grad / batch_size  #这不是project要求的版本， project要求的是Averaged stochastic gradient descent
            param.grad.zero_()

# Training
#######草稿，待修改#########
def train(lambd):
    x, y = initia_params()
    num_epochs, lr = 100, 0.001
    for epoch in range(num_epochs):
        for a, b in data_iter(batch_size, features, labels):
            with torch.enable_grad():
                l = logit_loss(x, y) + lambd * l2_penalty(x)
            l.sum().backward()
            sgd((x, y), lr, batch_size)
    print('L2 norm of x:', torch.norm(x).item())
    print(x, '\n', y)
    print(l)

# %%
train(0.1)
# %%
