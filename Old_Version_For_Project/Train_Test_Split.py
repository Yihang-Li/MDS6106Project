import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy import sparse
from scipy.io import loadmat

from sklearn.model_selection import train_test_split

a = loadmat('/home/ubuntu/CUHKSZ/MDS6106Optimization/Final Project/datasets/datasets/breast-cancer/breast-cancer_train.mat')
a = {k:v for k, v in a.items() if k[0] != '_'}
# print(a['A'])
a = a['A']

b = loadmat('/home/ubuntu/CUHKSZ/MDS6106Optimization/Final Project/datasets/datasets/breast-cancer/breast-cancer_train_label.mat')
b = {k:v for k, v in b.items() if k[0] != '_'}
b = b['b']

a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.3, random_state=97)
