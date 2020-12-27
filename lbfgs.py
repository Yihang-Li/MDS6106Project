import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def f_logit(xk, Lambda):
    x, y = xk[:-1], xk[-1]
    Log = np.log(1+np.exp(-b*(a.dot(x)+y)))
    return Lambda/2 * np.linalg.norm(x)**2 + Log.sum()/m

def df_logit(xk, Lambda):
    x = xk[:-1]
    y = xk[-1]

    grad = np.zeros((x.size+1, 1)).reshape(-1)
    # derivative for x1-xn 
    df_Log = (np.exp(-b.reshape(-1)*(a.dot(x)+y).reshape(-1)) /(1+np.exp(-b.reshape(-1)*(a.dot(x)+y).reshape(-1)))).reshape(-1,1)*(-b.reshape(-1, 1)*a)
#     print(df_Log)
#     print(np.sum(df_Log,axis=0))
#     print(x)
#     print(grad[:-1])
    grad[:-1] = Lambda * x + np.sum(df_Log,axis=0)/m
    
    # derivative for y
    df_Log = (np.exp(-b.reshape(-1)*(a.dot(x)+y).reshape(-1)) /(1+np.exp(-b.reshape(-1)*(a.dot(x)+y).reshape(-1)))).reshape(-1,1)*(-b)
#     print(df_Log)
    grad[x.size] = df_Log.sum()/m
    return grad.reshape(-1,)

###stepsize
#diminishing with pre-defined alpha_k 
diminishing = lambda k: 0.01/np.log10(k+2)

#exact by using golden section
#Note: w.r.t alpha, f is a 1-D function, this is why we can use golden section to perform exact line search

def exact(f, x_k: np.array, d_k: np.array, tol: float):
    """
    Input: funtion f, current point x_k, current_direction d_k, tolerence tol
    Output: current exact step size alpha_k by using golden section
    """
    phi = lambda alpha: f(x_k + alpha*d_k)
    alpha_k, _ = Golden_Section(phi, 0, 2, tol)
    
    return alpha_k

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
    
    ###Update first x_next by using backtracking and gradient method
    d_k = -f_grad(x_k)
        
    stepsize_strategy = stepsize_strategies[stepsize]
    
    if stepsize == 'backtrack':
        alpha_k = stepsize_strategy(f, f_grad, x_k, d_k, s=1, sigma=0.5, gamma=0.1)
    elif stepsize == 'diminishing':
        alpha_k = stepsize_strategy(k)
    else:
        alpha_k = stepsize_strategy(f, x_k, d_k, 1e-6)
    
    x_next = x_k + alpha_k*d_k
    
    while np.linalg.norm(f_grad(x_k)) >= tol:
        
        ## For Direction
        
#         print(x_next, x_k)
        
        #update yk
        yk = f_grad(x_next) - f_grad(x_k)
        
        #update sk
        sk = x_next - x_k
        
        #check whether to add them to the curvature pairs
        if sk.dot(yk) > 1e-14:
            if y_list and len(y_list) > m_lbfgs: 
                y_list.pop(0)
                s_list.pop(0)
            y_list.append(yk)
            s_list.append(sk) # add from back to forward
        
#         print("yk:", yk)
        #compute gamma_k
        gamma_k = sk.dot(yk)/np.linalg.norm(yk)**2

        alpha_list = []
        q = f_grad(x_k)
        #Recursive for initializing {H_k}^0
        for i in range(min(k, m_lbfgs), -1, -1):  ##倒着来
            if y_list and i < len(y_list):
                alpha_i = s_list[i].dot(q)/(s_list[i].dot(y_list[i]))
                q = q - alpha_i*y_list[i]
                alpha_list.append(alpha_i)
        r = gamma_k*q
            
        #Iterative for our final r
        for i in range(len(alpha_list)): ##顺着来
            beta = y_list[i].dot(r)/(s_list[i].dot(y_list[i]))
            r = r + (alpha_list[i] - beta)*s_list[i]
        
        d_k = -r
        
        ## For Stepsize
        if stepsize == 'backtrack':
            alpha_k = stepsize_strategy(f, f_grad, x_k, d_k, s=1, sigma=0.5, gamma=0.1)
        elif stepsize == 'diminishing':
            alpha_k = stepsize_strategy(k)
        else:
            alpha_k = stepsize_strategy(f, x_k, d_k, 1e-6)
        
        #update x_k, x_next
        x_next, x_k = x_next + alpha_k*d_k, np.array(x_next.tolist())
            
        k += 1
        print(np.linalg.norm(f_grad(x_k)))
        result.append([k, x_k.tolist(), alpha_k, f(x_k), np.linalg.norm(f_grad(x_k))])
    
        if k == max_iter:
            print('max iteration:', 
                  k, 'the function value is', f(x_k), 'the norm of gradient is:', np.linalg.norm(f_grad(x_k)))
            break        
        
    return result

# Main begin
n = 2 #number of features
delta = 1e-3
Lambda = 0.1
max_iter = 1000

data = pd.read_csv('/home/ubuntu/MDS6106Project/MDS6106Project/dataset_csv_files/dataset2.csv', header=None)
a = np.array(data.iloc[:, 0:2])
b = np.array(data.iloc[:, 2])

initial = np.zeros((n+1, 1)).reshape(-1,) #the last element is y
m = b.size

lbfgs_result = L_BFGS(lambda x_k: f_logit(x_k, Lambda), lambda x_k: df_logit(x_k, Lambda), initial, tol=1e-4, stepsize='backtrack', max_iter=max_iter, m_lbfgs= 5)

lbfgs_result
