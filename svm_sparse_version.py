import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy import sparse
from scipy.io import loadmat

plt.style.use("ggplot")
method = ['GM', 'AGM', 'BFGS']
color_list = ['blue', 'orange', 'purple', 'darkred', 'red', 'cyan', 'yellow', 'teal',
              'coral', 'brown', 'black']

def Normalize(features, method):
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MaxAbsScaler
    if (method == 1):
        features = features.todense()
        scaler = StandardScaler()
    else:
        scaler = MaxAbsScaler()
    features = scaler.fit_transform(features)
    return features

def Split(features,labels,test_size=0.3):
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size = test_size,random_state=0)
    return x_train, x_test, y_train,y_test

def phi_plus(t):
    if t <= delta:
        return 1/(2*delta) * (max(0, t)**2)
    else:
        return t - delta/2

def f_svm(xk, m, Lambda):
    x = xk[:-1]
    y = xk[-1][0]
    return Lambda/2 * np.linalg.norm(x)**2 + sum(map(phi_plus, 1 - b * (a @ x + y).reshape(-1)))
def df(xk, m, Lambda):
    x = xk[:-1]
    y = xk[-1][0]
    grad = np.zeros((x.size+1, 1))
    t = 1 - b * (a @ x + y).reshape(-1)
    dphi_list = np.zeros(m)
    dphi_list[t>delta] = 1
    dphi_list[(0<t) & (t<delta)] = t[(0<t) & (t<delta)]/delta
    grad[:-1] = Lambda*x + ((dphi_list * (-b)) @ a).reshape(-1,1)
    grad[x.size] = np.sum(-dphi_list * b)
    return grad
def plot_convergence(y, subfig_num):
    plt.figure(1, figsize=(12,5.2))
    plt.subplot(2,1,subfig_num-1)
    n = y.size
    x = np.arange(n)
    y = np.log(y)
    plt.plot(x, y, label = method[subfig_num-1], color = color_list[subfig_num], linewidth=2)
    plt.legend(loc='best',edgecolor='black', facecolor='white') #设置图例边框颜色
    plt.xlabel('number of iterations')
    plt.ylabel(r'$log\left(||\nabla f(x,y)||\right)$')
    plt.tight_layout()
    plt.savefig('./figures_convergence/SVM_convergence_dataset_'+dataset_name+'.pdf', dpi=1000)
def test(xk):
    
    a_test = loadmat('../final_project/datasets/'+dataset_name+'/'+dataset_name+'_test.mat')['A']
    b_test = loadmat('../final_project/datasets/'+dataset_name+'/'+dataset_name+'_test_label.mat')['b'].reshape(-1)
    a_test = Normalize(a_test, 2)
    
    m_test = b_test.size
    x = xk[:-1]
    y = xk[-1][0]
    accuracy = np.sum(np.abs((2 * ((a_test.dot(x) + y > 0) + 0) - 1).reshape(-1) + b_test))/(2*m_test)
    return accuracy   
def gradient_method(initial, m, Lambda):
    s = 1
    sigma = 0.5
    gamma = 0.1
    tol = 1e-4
    
    xk = initial
    gradient = df(initial, m, Lambda)
    norm_gradient_list = []
    norm_gradient_list.append(np.linalg.norm(gradient))
    num_iteration = 0
    
    while np.linalg.norm(gradient) > tol and num_iteration < max_iter:
        alphak = s
        dk = -df(xk, m, Lambda)
        while True:
            if f_svm(xk + alphak*dk, m, Lambda) - f_svm(xk, m, Lambda) <= gamma * alphak * np.dot(df(xk, m, Lambda).T, dk):
                break
            alphak = alphak * sigma
        
        xk = xk + alphak * dk
        gradient = df(xk, m, Lambda)
        norm_gradient_list.append(np.linalg.norm(gradient))
        num_iteration = num_iteration + 1
        print(num_iteration, np.linalg.norm(gradient))
    norm_gradient_list = np.array(norm_gradient_list)
    plot_convergence(norm_gradient_list, 1)
    print("iterations:", num_iteration)
    print("accuracy:", test(xk))

def AGM(initial, m, Lambda):
    x_minus = xk = initial
    t_minus = tk = 1
    alpha = 0.5
    yita = 0.5
    tol = 1e-4
    gradient = df(initial, m, Lambda)
    norm_gradient_list = []
    norm_gradient_list.append(np.linalg.norm(gradient))
    num_iteration = 0
    
    while np.linalg.norm(gradient) > tol and num_iteration < max_iter:
        betak = 1/tk * (t_minus - 1)
        yk = xk + betak * (xk - x_minus)
        x_bar = yk - alpha * df(yk, m, Lambda)
        while f_svm(x_bar, m, Lambda) - f_svm(yk, m, Lambda) > -alpha/2 * np.linalg.norm(df(yk, m, Lambda))**2:
            alpha = alpha * yita
            x_bar = yk - alpha * df(yk, m, Lambda)
        t_minus = tk
        tk = 0.5 * (1 + np.sqrt(1+4*tk**2))
        x_minus = xk
        
        xk = x_bar
        gradient = df(xk, m, Lambda)
        norm_gradient_list.append(np.linalg.norm(gradient))
        print(num_iteration, np.linalg.norm(gradient))
        num_iteration = num_iteration + 1

    norm_gradient_list = np.array(norm_gradient_list)
    plot_convergence(norm_gradient_list, 2)
    print("iterations:", num_iteration)
    print("accuracy:", test(xk))
def BFGS(initial, m, Lambda):
    Hk = np.identity(n+1)
    xk = initial
    s = 1
    sigma = 0.5
    gamma = 0.1
    tol = 1e-4
    gradient = df(initial, m, Lambda)
    norm_gradient_list = []
    norm_gradient_list.append(np.linalg.norm(gradient))
    num_iteration = 0
    
    while np.linalg.norm(gradient) > tol and num_iteration < max_iter:
        alphak = s
        dk = -np.dot(Hk, df(xk, m, Lambda))
        while True:
            if f_svm(xk + alphak*dk, m, Lambda) - f_svm(xk, m, Lambda) <= gamma * alphak * np.dot(df(xk, m, Lambda).T, dk):
                break
            alphak = alphak * sigma
        xk_new = xk + alphak * dk
        
        sk = xk_new - xk
        yk = df(xk_new, m, Lambda) - df(xk, m, Lambda)
        if np.dot(sk.T, yk) <= 1e-14:
            pass
        else:
            Hk = Hk + (np.dot((sk - np.dot(Hk, yk)), sk.T) + \
                       np.dot(sk, (sk-np.dot(Hk, yk)).T))/(np.dot(sk.T, yk)) \
                - np.dot((sk-np.dot(Hk, yk)).T, yk)/(np.dot(sk.T, yk)**2) * \
                    np.dot(sk, sk.T)
        num_iteration = num_iteration + 1
        xk = xk_new
        gradient = df(xk, m, Lambda)
        norm_gradient_list.append(np.linalg.norm(gradient))
        print(num_iteration, np.linalg.norm(gradient))
    
    norm_gradient_list = np.array(norm_gradient_list)
    plot_convergence(norm_gradient_list, 3)
    print("iterations:", num_iteration)
    print("accuracy:", test(xk))
    
# Main begin
delta = 1e-3
Lambda = 0.1
max_iter = 1000
dataset_name = 'a9a'

"""
dataset_num = 1
a = sparse.load_npz('./dataset_sparse_files/dataset'+str(dataset_num)+'_train.npz')
b = np.load('./dataset_sparse_files/dataset'+str(dataset_num)+'_train_labels.npy')
"""
a = loadmat('../final_project/datasets/'+dataset_name+'/'+dataset_name+'_train.mat')['A']
b = loadmat('../final_project/datasets/'+dataset_name+'/'+dataset_name+'_train_label.mat')['b'].reshape(-1)
a = a[:, :-1]
#normalize
a = Normalize(a, 2)
#a, a_test, b, b_test = Split(a,b,test_size=0.3)

m, n = a.shape

initial = np.zeros((n+1, 1)) #the last element is y

"""
print("----------GM----------")
start = time.time()
gradient_method(initial, m, Lambda)
stop = time.time()
print("time:", stop - start)
print()
"""

print("----------AGM----------")
start = time.time()
AGM(initial, m, Lambda)
stop = time.time()
print("time:", stop - start)
print()

print("----------BFGS----------")
start = time.time()
BFGS(initial, m, Lambda)
stop = time.time()
print("time:", stop - start)
print()


