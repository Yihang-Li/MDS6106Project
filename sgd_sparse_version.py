import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.io import loadmat

#diminishing with pre-defined alpha_k 
diminishing = lambda k: 0.1/np.log10(k+2)

def plot_result(m, xk):
    m1 = m2 = 0
    for i in np.arange(m):
        if b[i] == 1:
            m1 = m1 + 1
        else:
            m2 = m2 + 1
            
    s = 2 # control the point size
    x = a[0:m1, 0].toarray()
    y = a[0:m1, 1].toarray()
    plt.scatter(x, y, c='purple', s=s)
    x = a[m1:, 0].toarray()
    y = a[m1:, 1].toarray()
    plt.scatter(x, y, c='orange', s=s)
    
    A = xk[0][0]
    B = xk[1][0]
    C = xk[2][0]
    x = a[:,0].toarray()
    y = (-A*x-C)/B
    plt.plot(x, y, color='red')

def f_logit_i(xk, i, Lambda):
    x, y = xk[:-1], xk[-1]
    Log_i = np.log(1+np.exp(-b[i]*(a[i].dot(x)+y)))
    return Lambda/2*np.linalg.norm(x)**2 + Log_i

def df_logit_i(xk, i, Lambda):
    x, y = xk[:-1], xk[-1]
    grad = np.zeros((x.size+1, 1)).reshape(-1)
    #for x
    df_Logx_i = np.exp(-b[i]*(a[i].dot(x)+y)) /(1+np.exp(-b[i]*(a[i].dot(x)+y)))*(-b[i]*a[i])
    grad[:-1] = Lambda*x + df_Logx_i
    #for y
    df_Logy_i = np.exp(-b[i]*(a[i].dot(x)+y)) /(1+np.exp(-b[i]*(a[i].dot(x)+y)))*-b[i]
    grad[x.size] = df_Logy_i
    return grad.reshape(-1,)

def Stochastic_gradient_method(f_i, f_grad_i, x_0: np.array, batch: int, tol: float, max_iter: int):
    """
    Input: funtion i: f_i, the gradient of f_i: f_grad_i, initial point: x_0, batch size: batch,
           tolerence: tol, maximium iteration times: max_iter 
    Output: 2-d list: result. Every element of result is a list for each iteration. 
            e.g. ['iteration k', 'x_k', 'stepsize alpha_k', 'mean norm of f_grad_i(x_k)']
    """
    x_k, k, alpha_k = np.array(x_0.tolist()), 0, 0
    
    np.random.seed(k) #for each k, the batch set is different
    batch_index = np.random.randint(low = 0, high = m, size = batch) 
      
    result = [['iteration k', 'x_k', 'alpha_k', 'f(x_k)', 'mean norm of f_grad_i(x_k)']]
    result.append([k, x_k.tolist(), alpha_k, np.mean([np.linalg.norm(f_grad_i(x_k, i)) for i in batch_index], axis=0)])
    
     
    while np.mean([np.linalg.norm(f_grad_i(x_k, i)) for i in batch_index], axis=0) > tol:
        np.random.seed(k) #for each k, the batch set is different
        batch_index = np.random.randint(low = 0, high = m, size = batch) 
        d_k = -np.mean([f_grad_i(x_k, i) for i in batch_index], axis=0)
    
        alpha_k = diminishing(k)
        
        x_k += alpha_k*d_k
        
        if k == max_iter:
            print('max iteration:',
                  k, 'mean norm of f_grad_i(x_k) is:', np.mean([np.linalg.norm(f_grad_i(x_k, i)) for i in batch_index], axis=0))
            break        
        
        k += 1
        print(np.mean([np.linalg.norm(f_grad_i(x_k, i)) for i in batch_index], axis=0))
        result.append([k, x_k.tolist(), alpha_k, np.mean([np.linalg.norm(f_grad_i(x_k, i)) for i in batch_index], axis=0)])
    
    xk_result = np.array(result[-1][1]).reshape(-1,1)
    #plot_result(m, xk_result)    
    return result

# Main begin
delta = 1e-3
Lambda = 0.1
max_iter = 10000

"""
dataset_num = 1
a = sparse.load_npz('./dataset_sparse_files/dataset'+str(dataset_num)+'_train.npz')
b = np.load('./dataset_sparse_files/dataset'+str(dataset_num)+'_train_labels.npy')
m, n = a.shape
"""
a = loadmat('../final_project/datasets/breast-cancer/breast-cancer_train.mat')['A']
b = loadmat('../final_project/datasets/breast-cancer/breast-cancer_train_label.mat')['b'].reshape(-1)
m, n = a.shape

initial = np.zeros((n+1, 1)).reshape(-1,) #the last element is y

sgd_result = Stochastic_gradient_method(lambda x_k, i: f_logit_i(x_k, i, Lambda), lambda x_k, i: df_logit_i(x_k, i, Lambda), initial, batch=30, tol=1e-4, max_iter=max_iter)
