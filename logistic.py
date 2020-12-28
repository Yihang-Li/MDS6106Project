import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

plt.style.use("ggplot")
method = ['GM', 'AGM', 'L-BFGS']
color_list = ['blue', 'orange', 'purple', 'darkred', 'red', 'cyan', 'yellow', 'teal',
              'coral', 'brown', 'black']

def plot_result(m, xk, num_fig):
    plt.figure(num_fig)
    m1 = m2 = 0
    for i in np.arange(m):
        if b[i] == 1:
            m1 = m1 + 1
        else:
            m2 = m2 + 1
            
    s = 5 # control the point size
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
    plt.xlabel('$a_1$')
    plt.ylabel('$a_2$')
    plt.savefig('./figures_with_seperate_line/'+'Logistic_dataset'+str(dataset_num)+method[num_fig-1]+'.pdf', dpi=1000)
    
def plot_convergence(y, subfig_num):
    plt.figure(10, figsize=(12,8))
    plt.subplot(3,1,subfig_num)
    n = y.size
    x = np.arange(n)
    y = np.log(y)
    plt.plot(x, y, label = method[subfig_num-1], color = color_list[subfig_num], linewidth=2)
    plt.legend(loc='best',edgecolor='black', facecolor='white') #设置图例边框颜色
    plt.xlabel('number of iterations')
    plt.ylabel(r'$log\left(\nabla f(x,y)\right)$')
    plt.tight_layout()
    plt.savefig('./figures_convergence/Logistic_convergence_dataset'+str(dataset_num)+'.pdf', dpi=1000)

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

def f_logit(xk, Lambda):
    x, y = xk[:-1], xk[-1]
    Log = np.log(1+np.exp(-b*(a.dot(x)+y)))
    return Lambda/2 * np.linalg.norm(x)**2 + Log.sum()/m

def df_logit(xk, Lambda):
    x = xk[:-1]
    y = xk[-1]

    grad = np.zeros((x.size+1, 1)).reshape(-1)
    
    t = np.exp(-b.reshape(-1)*(a.dot(x)+y).reshape(-1))
    
    df_Log = np.dot((t /(1+t))*(-b), a)
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
    norm_gradient_list = np.array(gm_result)[1:,-1].shape
    plot_convergence(norm_gradient_list, 1)
    print("iterations:", k)
    plot_result(m, xk_result, 1)
    print("accuracy:", test(xk_result, dataset_num))
    
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
    norm_gradient_list = np.array(gm_result)[1:,-1].shape
    plot_convergence(norm_gradient_list, 2)
    print("iterations:", k)
    plot_result(m, xk_result, 2)
    print("accuracy:", test(xk_result, dataset_num))
   
    return result

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
    norm_gradient_list = np.array(gm_result)[1:,-1].shape
    plot_convergence(norm_gradient_list, 3)
    print("iterations:", k)
    plot_result(m, xk_result, 3)
    print("accuracy:", test(xk_result, dataset_num))
    return result

# Main begin
delta = 1e-3
Lambda = 0.1
max_iter = 100

dataset_num = 1
data = pd.read_csv('./dataset_csv_files/dataset'+str(dataset_num)+'.csv', header=None)
a = np.array(data.iloc[:, 0:2])
b = np.array(data.iloc[:, 2])
m, n = a.shape

initial = np.zeros((n+1, 1)).reshape(-1,) #the last element is y

print("----------GM----------")
start = time.time()
gm_result = gradient_method(lambda x_k: f_logit(x_k, Lambda), lambda x_k: df_logit(x_k, Lambda), initial, tol=1e-4, stepsize='backtrack', max_iter=max_iter)
stop = time.time()
print("time:", stop - start)
print()

print("----------AGM----------")
start = time.time()
L = np.linalg.norm(a)**2/4/m
AGM_result = AGM(lambda x_k: f_logit(x_k, Lambda), lambda x_k: df_logit(x_k, Lambda), initial, tol=1e-4, max_iter=max_iter, L=L)
stop = time.time()
print("time:", stop - start)
print()


print("----------L-BFGS----------")
start = time.time()
lbfgs_result = L_BFGS(lambda x_k: f_logit(x_k, Lambda), lambda x_k: df_logit(x_k, Lambda), initial, tol=1e-7, stepsize='backtrack', max_iter=max_iter, m_lbfgs= 5)
stop = time.time()
print("time:", stop - start)
print()

