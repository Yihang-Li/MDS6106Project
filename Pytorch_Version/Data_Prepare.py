# %%
import torch
import random
random.seed(1997)
import matplotlib.pyplot as plt
plt.style.use("ggplot")
# %%
#number of features
num_features = 2

#Generating Data
def generate_data(c1, c2, sigma1, sigma2, m1, m2):
    m = m1 + m2 # total number of examples
    a = torch.zeros((m, num_features))
    b = torch.zeros((m, 1))

    a1 = c1 + torch.normal(0, sigma1, (m1, num_features))
    b1 = torch.ones(m1).reshape(-1, 1)
    a2 = c2 + torch.normal(0, sigma2, (m2, num_features))
    b2 = torch.ones(m2).reshape(-1, 1)*-1

    a = torch.cat((a1, a2))
    b = torch.cat((b1, b2))

    return a, b
# %%
c1 = torch.tensor([0, 0])
c2 = torch.tensor([4, 5])
sigma1, sigma2 = 0.1, 1
m1, m2 = 999, 1024
features, labels = generate_data(c1, c2, sigma1, sigma2, m1, m2)  ### features = a;  labes = b
# %%

def plot_data(features, m1):
    x, y = features[:m1, 0], features[:m1, 1]
    plt.scatter(x.detach(), y.detach(), c='purple', s=5)
    x, y = features[m1:, 0], features[m1:, 1]
    plt.scatter(x.detach(), y.detach(), c='orange', s=5)
    plt.xlabel(r'$a_1$')
    plt.ylabel(r'$a_2$')
# %%
# plot_data(features, m1)

# %%
#Reading Data
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(num_examples, batch_size + i)])
        yield features[batch_indices], labels[batch_indices]

# # %%
batch_size = 10
# X, y = next(iter(data_iter(batch_size, features, labels)))
# print(X, '\n', y)

# %%
