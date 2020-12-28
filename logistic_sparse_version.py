import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.io import loadmat
import scipy.sparse.linalg

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

def test(xk):

    a = loadmat('../final_project/datasets/rcv1/rcv1_test.mat')['A']
    b = loadmat('../final_project/datasets/rcv1/rcv1_test_label.mat')['b'].reshape(-1)
    m_test = b.size
    x = xk[:-1]
    y = xk[-1][0]
    accuracy = np.sum(np.abs((2 * ((a.dot(x) + y > 0) + 0) - 1).reshape(-1) + b))/(2*m_test)
    return accuracy 

def f_logit(xk, Lambda):
    x, y = xk[:-1], xk[-1]
    Log = np.log(1+np.exp(-b*(a @ x+y)))
    return Lambda/2 * np.linalg.norm(x)**2 + Log.sum()/m

def df_logit(xk, Lambda):
    x = xk[:-1]
    y = xk[-1]

    grad = np.zeros((x.size+1, 1)).reshape(-1)
    
    t = np.exp(-b.reshape(-1)*(a @ x+y).reshape(-1))
    
    df_Log = (t /(1+t))*(-b) @ a
    grad[:-1] = Lambda * x + df_Log/m
    df_Log = (t /(1+t))*(-b)
    grad[x.size] = df_Log.sum()/m
    return grad.reshape(-1,)

#Stepsize
#diminishing with pre-defined alpha_k 

#exact by using golden section
#Note: w.r.t alpha, f is a 1-D function, this is why we can use golden section to perform exact line search


def armijo(f, f_grad, x_k: np.array, d_k: np.array, s: float, sigma: float, gamma: float):
    """
    Input: funtion f, the gradient of f, current point x_k, current_direction d_k, parameters s, sigma and gamma
    Output: current step size alpha_k by backtracking

    """
    alpha_k = s
    while f(x_k + alpha_k*d_k) - f(x_k) > gamma*alpha_k*f_grad(x_k).dot(d_k):
        alpha_k *= sigma
    return alpha_k

##GM
#define our stepsize_strategies dictionary
stepsize_strategies = {"backtrack": armijo}

def gradient_method(f, f_grad, x_0: np.array, tol: float, stepsize: str, max_iter: int):
    """
    Input: funtion: f, the gradient of f: f_grad, initial point: x_0, tolerence: tol, stepsize method: stepsize,
           maximium iteration times: max_iter 
    Output: 2-d list: result. Every element of result is a list for each iteration. 
            e.g. ['iteration k', 'x_k', 'stepsize alpha_k', 'f(x_k)', 'norm of f_grad(x_k)']
    """
    x_k = np.array(x_0.tolist())
    k, alpha_k = 0, 0    
    
    result = [['iteration k', 'x_k', 'alpha_k', 'f(x_k)', 'norm of f_grad(x_k)']]
    result.append([k, x_k.tolist(), alpha_k, f(x_k), np.linalg.norm(f_grad(x_k))])
    
    stepsize_strategy = stepsize_strategies[stepsize]
    
    
    while np.linalg.norm(f_grad(x_k)) > tol:
        
        d_k = -f_grad(x_k)
        
        if stepsize == 'backtrack':
            alpha_k = stepsize_strategy(f, f_grad, x_k, d_k, s=1, sigma=0.5, gamma=0.1)
        elif stepsize == 'diminishing':
            alpha_k = stepsize_strategy(k)
        else:
            alpha_k = stepsize_strategy(f, x_k, d_k, 1e-6)
        
        x_k += alpha_k*d_k
        
        if k == max_iter:
            print('max iteration:',
                  k, 'the function value is', f(x_k), 'the norm of gradient is:', np.linalg.norm(f_grad(x_k)))
            break        
        print(np.linalg.norm(f_grad(x_k)))
        k += 1
        print(np.linalg.norm(f_grad(x_k)))
        result.append([k, x_k.tolist(), alpha_k, f(x_k), np.linalg.norm(f_grad(x_k))])
    xk_result = np.array(result[-1][1]).reshape(-1,1)
    #plot_result(m, xk_result)
    print(test(xk_result))
    return result

##AGM
def AGM(f, f_grad, x_0: np.array, tol: float, max_iter: int, L: float):
    """
    Input: funtion: f, the gradient of f: f_grad, initial point: x_0, tolerence: tol,
           maximium iteration times: max_iter, L: Lipschitz Constant  
           
    Output: 2-d list: result. Every element of result is a list for each iteration. 
            e.g. ['iteration k', 'x_k', 'stepsize alpha_k', 'f(x_k)', 'norm of f_grad(x_k)']
    """
    x_pre = np.array(x_0.tolist())
    x_k = np.array(x_0.tolist()) #x_next
    k = 0
    
    t_pre, t_next = 1, 1
    beta_k = 0
    y_k = x_pre
    alpha_k = 1/L 
    
    result = [['iteration k', 'x_k', 'alpha_k', 'f(x_k)', 'norm of f_grad(x_k)']]
    result.append([k, x_k.tolist(), alpha_k, f(x_k), np.linalg.norm(f_grad(x_k))])
    
    
    while np.linalg.norm(f_grad(x_k)) > tol:
        
        x_k = y_k - alpha_k*f_grad(y_k)
        
        t_pre, t_next = t_next, (1 + np.sqrt(1+4*t_pre**2))/2
        beta_k = (t_pre - 1)/t_next
        y_k = x_k + beta_k*(x_k - x_pre)
        
        #Remember to update x_pre!
        x_pre = x_k
        
        if k == max_iter:
            print('max iteration:',
                  k, 'the function value is', f(x_k), 'the norm of gradient is:', np.linalg.norm(f_grad(x_k)))
            break        
        
        k += 1
        print(np.linalg.norm(f_grad(x_k)))
        result.append([k, x_k.tolist(), alpha_k, f(x_k), np.linalg.norm(f_grad(x_k))])
    xk_result = np.array(result[-1][1]).reshape(-1,1)
    #plot_result(m, xk_result)
    print(test(xk_result))
    return result

# Main begin
delta = 1e-3
Lambda = 0.1
max_iter = 10000

"""
dataset_num = 4
a = sparse.load_npz('./dataset_sparse_files/dataset'+str(dataset_num)+'_train.npz')
b = np.load('./dataset_sparse_files/dataset'+str(dataset_num)+'_train_labels.npy')
m, n = a.shape

"""
a = loadmat('../final_project/datasets/rcv1/rcv1_train.mat')['A']
b = loadmat('../final_project/datasets/rcv1/rcv1_train_label.mat')['b'].reshape(-1)
m, n = a.shape

initial = np.zeros((n+1, 1)).reshape(-1,) #the last element is y


#gm_result = gradient_method(lambda x_k: f_logit(x_k, Lambda), lambda x_k: df_logit(x_k, Lambda), initial, tol=1e-4, stepsize='backtrack', max_iter=max_iter)

L = scipy.sparse.linalg.norm(a)**2/4/m
AGM_result = AGM(lambda x_k: f_logit(x_k, Lambda), lambda x_k: df_logit(x_k, Lambda), initial, tol=1e-4, max_iter=max_iter, L=L)


