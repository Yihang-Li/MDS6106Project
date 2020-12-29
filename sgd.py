import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#diminishing with pre-defined alpha_k 
diminishing = lambda k: 10/np.log10(k+2)

def armijo(f, f_grad, x_k: np.array, d_k: np.array, s: float, sigma: float, gamma: float):
    """
    Input: funtion f, the gradient of f, current point x_k, current_direction d_k, parameters s, sigma and gamma
    Output: current step size alpha_k by backtracking

    """
    alpha_k = s
    while f(x_k + alpha_k*d_k) - f(x_k) > gamma*alpha_k*f_grad(x_k).dot(d_k):
        alpha_k *= sigma
    return alpha_k

def plot_result(xk):
    
    m1 = m2 = 0
    for i in np.arange(m):
        if b[i] == 1:
            m1 = m1 + 1
        else:
            m2 = m2 + 1
            
    s = 2 # control the point size
    x = a[0:m1, 0]
    y = a[0:m1, 1]
    plt.scatter(x, y, c='purple', s=s)
    x = a[m1:, 0]
    y = a[m1:, 1]
    plt.scatter(x, y, c='orange', s=s)
    
    A = xk[0][0]
    B = xk[1][0]
    C = xk[2][0]
    x = a[:,0]
    y = (-A*x-C)/B
    plt.plot(x, y, color='red')

def f_logit_i(xk, i, Lambda):
    x, y = xk[:-1], xk[-1]
    Log_i = np.log(1+np.exp(-b[i]*(a[i].dot(x)+y)))
    return Lambda/2*np.linalg.norm(x)**2 + Log_i


def df_logit(xk, batch_index):
    x = xk[:-1]
    y = xk[-1]
    
    batch_index = batch_index.tolist()
    grad = np.zeros((x.size+1, 1)).reshape(-1)
    
    a_new = a[batch_index].reshape((-1, n))
    b_new = b[batch_index].reshape(-1)
    t = np.exp(-b_new.reshape(-1)*(a_new.dot(x)+y).reshape(-1))
    
    df_Log = np.dot((t /(1+t))*(-b_new), a_new)
    grad[:-1] = Lambda * x + df_Log/len(batch_index)
    df_Log = (t /(1+t))*(-b_new)
    grad[x.size] = df_Log.sum()/len(batch_index)
    return grad.reshape(-1,)

def test(xk, dataset_num):
    data = pd.read_csv('./test_dataset_csv_files/test_dataset'+str(dataset_num)+'.csv', header=None)
    a = np.array(data.iloc[:, 0:2])
    b = np.array(data.iloc[:, 2])
    m_test = b.size
    x = xk[:-1]
    y = xk[-1][0]
    
    accuracy = 0
    p = np.zeros(m_test)
    for i in np.arange(m_test):
        if np.dot(a[i], x)[0] + y > 0:
            p[i] = 1
        else:
            p[i] = -1
    accuracy = np.sum(np.abs(p + b)) / (2*m_test)
    return accuracy


def Stochastic_gradient_method(f_i, f_grad_i, x_0: np.array, batch: int, tol: float, max_iter: int):
    """
    Input: funtion i: f_i, the gradient of f_i: f_grad_i, initial point: x_0, batch size: batch,
           tolerence: tol, maximium iteration times: max_iter 
    Output: 2-d list: result. Every element of result is a list for each iteration. 
            e.g. ['iteration k', 'x_k', 'stepsize alpha_k', 'mean norm of f_grad_i(x_k)']
    """
    x_k, k, alpha_k = np.array(x_0.tolist()), 0, 0
    
    result = [['iteration k', 'x_k', 'alpha_k', 'f(x_k)', 'mean norm of f_grad_i(x_k)']]    
     
    # while np.mean([np.linalg.norm(f_grad_i(x_k, i)) for i in batch_index], axis=0) > tol:
    # np.random.seed(k) #for each k, the batch set is different
    # batch_index = np.random.choice(np.arange(0, m), batch, replace=False)
    while k < max_iter:

        np.random.seed(k) #for each k, the batch set is different
        batch_index = np.random.choice(np.arange(0, m), batch, replace=False)
        #batch_index = np.random.randint(low = 0, high = m, size = batch) 
        
        # d_k = -np.mean([f_grad_i(x_k, i) for i in batch_index], axis=0)
        d_k = -f_grad_i(x_k, batch_index)/len(batch_index)
        alpha_k = diminishing(k)
        #alpha_k = 0.001       
        x_k += alpha_k*d_k

        k += 1
        # print(np.mean([np.linalg.norm(f_grad_i(x_k, i)) for i in batch_index], axis=0))
        #print(k, min([np.linalg.norm(f_grad_i(x_k, i)) for i in batch_index]))
        result.append([k, x_k.tolist(), alpha_k])
        #print(np.linalg.norm(f_grad_i(x_k, batch_index)))
        if np.linalg.norm(f_grad_i(x_k, batch_index)) < tol:
            break
        """
        if k > max_iter:
            break
        min([np.linalg.norm(f_grad_i(x_k, i)) for i in batch_index]) > tol
        """

    xk_result = np.array(result[-1][1]).reshape(-1,1)
    plot_result(xk_result)
    print("iterations:", k)
    print("accuracy:", test(xk_result, dataset_num))
    #print(result[-1])
    return result

# Main begin
delta = 1e-3
Lambda = 0.1
max_iter = 100000
dataset_num = 0
m, n = 0, 0
for i in np.arange(1,5):
    dataset_num = i
    data = pd.read_csv('./dataset_csv_files/dataset'+str(dataset_num)+'.csv', header=None)
    a = np.array(data.iloc[:, 0:2])
    b = np.array(data.iloc[:, 2])
    m, n = a.shape
    plt.figure(i)
    initial = np.ones((n+1, 1)).reshape(-1,) #the last element is y
    start = time.time()
    sgd_result = Stochastic_gradient_method(lambda x_k, i: f_logit_i(x_k, i, Lambda), df_logit, initial, 66, tol=1e-4, max_iter=max_iter)
    stop = time.time()
    print('time:', stop-start)
    plt.savefig('./figures_with_seperate_line/'+'SGD_dataset'+str(dataset_num)+'.pdf', dpi=1000)
    print()


    
    
    
    