import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse

def f_logit(xk, Lambda):
    x, y = xk[:-1], xk[-1]
    Log = np.log(1+np.exp(-b*(a.dot(x)+y)))
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

#backtrack(armijo)
def armijo(f, f_grad, x_k: np.array, d_k: np.array, s: float, sigma: float, gamma: float):
    """
    Input: funtion f, the gradient of f, current point x_k, current_direction d_k, parameters s, sigma and gamma
    Output: current step size alpha_k by backtracking

    """
    alpha_k = s
    while f(x_k + alpha_k*d_k) - f(x_k) > gamma*alpha_k*f_grad(x_k).dot(d_k):
        alpha_k *= sigma
    return alpha_k

#define our stepsize_strategies dictionary
stepsize_strategies = {"backtrack": armijo}

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
def L_BFGS(f, f_grad, x_0: np.array, tol: float, stepsize: str, max_iter: int, m_lbfgs: int):
    """
    Input: funtion: f, the gradient of f: f_grad, initial point: x_0, tolerence: tol,
           step size strategies: stepsize, maximium iteration times: max_iter, 
           memory parameter: m_lbfgs
           
    Output: 2-d list: result. Every element of result is a list for each iteration. 
            e.g. ['iteration k', 'x_k', 'stepsize alpha_k', 'f(x_k)', 'norm of f_grad(x_k)'] ：： may change
    """
    
    k, x_k = 0, np.array(x_0.tolist())
    alpha_k = 0
    
    result = [['iteration k', 'x_k', 'alpha_k', 'f(x_k)', 'norm of f_grad(x_k)']]
    result.append([k, x_k.tolist(), alpha_k, f(x_k), np.linalg.norm(f_grad(x_k))])
    
    #initiate yk_list and sk_list
    y_list, s_list = [], []
    for i in np.arange(m_lbfgs):
        y_list.append(np.zeros(n+1))
        s_list.append(np.zeros(n+1))
    ###Update first x_next by using backtracking and gradient method
    d_k = -f_grad(x_k)
        
    stepsize_strategy = stepsize_strategies[stepsize]
    
    if stepsize == 'backtrack':
        alpha_k = stepsize_strategy(f, f_grad, x_k, d_k, s=1, sigma=0.5, gamma=0.1)
    
    x_next = x_k + alpha_k*d_k
    k = k + 1
    result.append([k, x_next.tolist(), alpha_k, f(x_next), np.linalg.norm(f_grad(x_next))]) #first step
    while np.linalg.norm(f_grad(x_next)) >= tol:
        
        yk = f_grad(x_next) - f_grad(x_k)
        
        #update sk
        sk = x_next - x_k
        
        y_list.append(yk)
        s_list.append(sk) # add from back to forward
        y_list.pop(0)
        s_list.pop(0)

        #compute gamma_k
        if sk.dot(yk) < 1e-14:
            gamma_k = 1
        else:
            gamma_k = sk.dot(yk)/np.linalg.norm(yk)**2

        alpha_list = []
        q = f_grad(x_k)
        #Recursive for initializing {H_k}^0
        for i in range(m_lbfgs-1, -1, -1):  ##倒着来
            if s_list[i].dot(y_list[i]) < 1e-14:
                alpha_list.append(0)
                continue
            else:
                alpha_i = s_list[i].dot(q)/(s_list[i].dot(y_list[i]))
                q = q - alpha_i*y_list[i]
                alpha_list.append(alpha_i)           
        
        r = gamma_k*q
            
        #Iterative for our final r
        for i in range(m_lbfgs): ##顺着来
            if s_list[i].dot(y_list[i]) < 1e-14:
                continue
            else:
                beta = y_list[i].dot(r)/(s_list[i].dot(y_list[i]))
                r = r + (alpha_list[i] - beta)*s_list[i]
        
        d_k = -r
        ## For Stepsize
        if stepsize == 'backtrack':
            alpha_k = stepsize_strategy(f, f_grad, x_k, d_k, s=1, sigma=0.5, gamma=0.1)
        
        #update x_k, x_next
        x_k = np.array(x_next.tolist())
        x_next = x_next + alpha_k*d_k
    
        k += 1
        print(np.linalg.norm(f_grad(x_next)))
        result.append([k, x_next.tolist(), alpha_k, f(x_next), np.linalg.norm(f_grad(x_next))])
        
        if k == max_iter:
            print('max iteration:', 
                  k, 'the function value is', f(x_next), 'the norm of gradient is:', np.linalg.norm(f_grad(x_next)))
            break 
    
    xk_result = np.array(result[-1][1]).reshape(-1,1)
    plot_result(m, xk_result)
    return result

# Main begin
Lambda = 0.1
max_iter = 10000

dataset_num = 1
a = sparse.load_npz('./dataset_sparse_files/dataset'+str(dataset_num)+'_train.npz')
b = np.load('./dataset_sparse_files/dataset'+str(dataset_num)+'_train_labels.npy')
m, n = a.shape

initial = np.zeros((n+1, 1)).reshape(-1,) #the last element is y

lbfgs_result = L_BFGS(lambda x_k: f_logit(x_k, Lambda), lambda x_k: df_logit(x_k, Lambda), initial, tol=1e-7, stepsize='backtrack', max_iter=max_iter, m_lbfgs= 5)

lbfgs_result
