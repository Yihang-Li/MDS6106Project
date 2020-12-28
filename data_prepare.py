# data preparation

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use("ggplot")

def plot_data(data):
    s = 5 # control the point size
    x = data[0:m1, 0]
    y = data[0:m1, 1]
    plt.scatter(x, y, c='purple', s=s)
    
    x = data[m1:, 0]
    y = data[m1:, 1]
    plt.scatter(x, y, c='orange', s=s)
    plt.xlabel(r'$a_1$')
    plt.ylabel(r'$a_2$')
    
n = 2 #number of feature
np.random.seed(100)

def generate_data(c1, c2, sigma1, sigma2, m1, m2, dataset_num):
    m = m1 + m2
    a = np.zeros((m, n))
    b = np.zeros((m, 1))
    if (dataset_num == 4):
        i = 0
        while i < m1:
            error = np.random.normal(mean1, sigma1, n)
            if np.sum(error >= 0) < n:
                continue
            a[i] = c1 + error
            b[i] = 1
            i = i + 1
    else:
        for i in np.arange(m1):
            a[i] = c1 + np.random.normal(mean1, sigma1, n)
            b[i] = 1
    for i in np.arange(m1, m):
        a[i] = c2 + np.random.normal(mean2, sigma2, n)
        b[i] = -1
    plt.figure(dataset_num)
    plot_data(a)
    path = 'dataset'+str(dataset_num)+'.pdf'
    
    plt.savefig("./dataset_figures/"+path, dpi=1000)
    a = pd.DataFrame(a)
    b = pd.DataFrame(b)
    data = pd.concat([a, b], axis=1)
    data.to_csv('./dataset_csv_files/'+'dataset'+str(dataset_num)+'.csv', header=False, index=False)


''' dataset1 '''
c1 = np.array([5, 5])
c2 = np.array([8,10])
sigma1 = sigma2 = 2
mean1 = mean2 = 0
m1 = m2 = 500
generate_data(c1, c2, sigma1, sigma2, m1, m2, 1)


''' dataset2 '''
c1 = np.array([17, 14])
c2 = np.array([10,20])
sigma1 = 1.2
sigma2 = 2.5
mean1 = mean2 = 0
m1 = m2 = 1000
generate_data(c1, c2, sigma1, sigma2, m1, m2, 2)

''' dataset3 '''
c1 = np.array([0, 1])
c2 = np.array([1, 0])
sigma1 = 0.3
sigma2 = 0.3
mean1 = mean2 = 0
m1 = m2 = 600
generate_data(c1, c2, sigma1, sigma2, m1, m2, 3)

''' dataset4 '''
c1 = np.array([10, 15])
c2 = np.array([15, 10])
sigma1 = 3
sigma2 = 2
mean1 = mean2 = 0
m1 = m2 = 1200
generate_data(c1, c2, sigma1, sigma2, m1, m2, 4)





