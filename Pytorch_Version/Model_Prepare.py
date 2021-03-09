# %%
import sys
sys.path.append("/Users/yihangli/Documents/GitHub/MDS6106Project")
# %%
from Pytorch_Version.Data_Prepare import *
import torch
# %%
#number of features
# num_features = 2

# Initializing Model Parameters
## weight: x   bias: y
def initia_params():
    x, y = torch.normal(0, 0.01, size=(num_features, 1), requires_grad=True), torch.zeros(size=(1, ), requires_grad=True)
    return x, y


# Define the Model
#Remember in our problem, x, y are parameters and a, b are data
lin = lambda a: torch.mm(a, x) + y
def sigmoid(a):
    return 1/(1+torch.exp(-a))
def net(a):
    return sigmoid(lin(a))


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

# Define Classification Accuracy
## Note: in our case, label is b,
## we need to predict b_hat and compare them
def accuracy(b_hat, b):
    return (b_hat.type(b.dtype) == b).type(b.dtype).mean().item()
# Define evaluate accuracy function for net and data_iter
def eval_accuracy(net, data_iter):
    """Learn from d2l, compute the accuracy for a model on a dataset."""
    acc_sum, n = 0.0, 0
    for a, b in data_iter:
        acc_sum += accuracy(2*(net(a) > 0.5).type(b.dtype)-1, b)
        n += 1
    return acc_sum/n

# %%

# Training
lambd = 0.1
x, y = initia_params()
num_epochs, lr = 50, 0.001
for epoch in range(num_epochs):
    for a, b in data_iter(batch_size, features, labels):
        with torch.enable_grad():
            l = logit_loss(x, y) + lambd * l2_penalty(x)
        l.backward() #l.sum().backward(), l here is already a scalar! no need to sum
        sgd((x, y), lr, batch_size)
        x.grad.zero_()
        y.grad.zero_()
    train_accuracy = eval_accuracy(net, data_iter(batch_size, features, labels))
    print('epoch {}'.format(epoch+1),
        ': with training accuracy {}'.format(train_accuracy))
print('L2 norm of x:', torch.norm(x).item())
print(x, '\n', y)
print('loss:', l.item(), ' training accuracy:', train_accuracy)


# %%
# Plot result by fited parameters with separate line
def plot_result(x, y, features, m1):
    plot_data(features, m1)
    A = x[0].item()
    B = x[1].item()
    C = - y.item()
    #C = (lin(a).mean() - y).item()
    ### Note: There is some problem with this intercept term.
    ####2021/03/09: Try to fix!
    # For different data, have different intercept?
    ## No! the truth is, the position at lin(a) should be zero!

    a1 = features[:,0]
    a2 = (-A * a1 + C) / B
    plt.plot(a1.detach(), a2.detach(), color='red')
    plt.xlabel('$a_1$')
    plt.ylabel('$a_2$')
# %%
plot_result(x, y, features, m1)

# %%
