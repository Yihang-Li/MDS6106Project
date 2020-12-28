import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import sparse

for i in np.arange(1,5):
    data = pd.read_csv('./dataset_csv_files/dataset'+str(i)+'.csv', header=None)
    features = np.array(data.iloc[:, 0:2])
    features = sparse.csc_matrix(features)
    labels = np.array(data.iloc[:, 2])
    
    # save
    sparse.save_npz('./dataset_sparse_files/dataset'+str(i)+'_train.npz',features)
    np.save('./dataset_sparse_files/dataset'+str(i)+'_train_labels.npy',labels)
    
a = sparse.load_npz('./dataset_sparse_files/dataset1_train.npz')
b = np.load('./dataset_sparse_files/dataset1_train_labels.npy')
