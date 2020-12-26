# data preparation

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_data(data):
    s = 0.7 # control the point size
    x = a[0:m1, 0]
    y = a[0:m1, 1]
    plt.scatter(x, y, c='purple', s=s)
    
    x = a[m1:, 0]
    y = a[m1:, 1]
    plt.scatter(x, y, c='orange', s=s)
    
    
''' dataset1 '''
np.random.seed(100)
c1 = np.array([5, 5])
c2 = np.array([10,20])
sigma1 = sigma2 = 2.5
mean1 = mean2 = 0
m1 = m2 = 1000

m = m1 + m2
a = np.zeros((m, 2))
size = 2

for i in np.arange(m1):
    a[i] = c1 + np.random.normal(mean1, sigma1, size)

for i in np.arange(m1, m):
    a[i] = c2 + np.random.normal(mean2, sigma2, size)

plt.figure(1)
plot_data(a)

a = pd.DataFrame(a)
a.to_csv('dataset1.csv', header=False, index=False)

''' dataset2 '''
np.random.seed(100)
c1 = np.array([17, 14])
c2 = np.array([10,20])
sigma1 = 0.8
sigma2 = 2.5
mean1 = mean2 = 0
m1 = m2 = 1000

m = m1 + m2
a = np.zeros((m, 2))
size = 2

for i in np.arange(m1):
    a[i] = c1 + np.random.normal(mean1, sigma1, size)

for i in np.arange(m1, m):
    a[i] = c2 + np.random.normal(mean2, sigma2, size)

plt.figure(2)
plot_data(a)
a = pd.DataFrame(a)
a.to_csv('dataset2.csv', header=False, index=False)

''' dataset2 '''
np.random.seed(100)
c1 = np.array([10, 15])
c2 = np.array([15, 10])
sigma1 = 3
sigma2 = 2
mean1 = mean2 = 0
m1 = m2 = 2000

m = m1 + m2
a = np.zeros((m, 2))
size = 2

i = 0
while i < m1:
    error = np.random.normal(mean1, sigma1, size)
    if np.sum(error >= 0) < size:
        continue
    a[i] = c1 + error
    i = i + 1
    
for i in np.arange(m1, m):
    a[i] = c2 + np.random.normal(mean2, sigma2, size)

plt.figure(3)
plot_data(a)
a = pd.DataFrame(a)
a.to_csv('dataset3.csv', header=False, index=False)



