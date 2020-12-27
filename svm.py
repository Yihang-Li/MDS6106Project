import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse

def phi_plus(t):
    if t <= delta:
        return 1/(2*delta) * (max(0, t)**2)
    else:
        return t - delta/2

def d_phi_plus(t):
    if t < 0:
        return 0
    elif t > delta:
        return 1
    else:
        return t/delta
def f_svm(xk, m, Lambda):
    x = xk[:-1]
    y = xk[-1][0]
    Sum = 0
    for i in np.arange(m):
        t = 1-b[i]*(np.dot(a[i],x)[0] + y)
        Sum = Sum + phi_plus(t)
    return Lambda/2 * np.linalg.norm(x)**2 + Sum
def df(xk, m):
    x = xk[:-1]
    y = xk[-1][0]
    grad = np.zeros((x.size+1, 1))
    # derivative for x1-xn 
    for k in np.arange(x.size): #df_xk
        Sum = 0
        for i in np.arange(m):
            t = 1-b[i]*(np.dot(a[i],x)[0] + y)
            Sum = Sum + d_phi_plus(t)*(-b[i]*a[i][k])
        grad[k] = Lambda * x[k] + Sum
    
    # derivative for y
    Sum = 0
    for i in np.arange(m):
        t = 1-b[i]*(np.dot(a[i],x)[0] + y)
        Sum = Sum + d_phi_plus(t)*(-b[i])
    grad[x.size] = Sum
    return grad

#gradient method
def gradient_method(initial, m, Lambda):
    s = 1
    sigma = 0.5
    gamma = 0.1
    tol = 1e-8
    
    xk = initial
    gradient = df(initial, m)
    num_iteration = 0
    
    while np.linalg.norm(gradient) > tol and num_iteration < max_iter:
        alphak = s
        dk = -df(xk, m)
        while True:
            if f_svm(xk + alphak*dk, m, Lambda) - f_svm(xk, m, Lambda) <= gamma * alphak * np.dot(df(xk, m).T, dk):
                break
            alphak = alphak * sigma
        
        xk = xk + alphak * dk
        gradient = df(xk, m)
        #print(np.linalg.norm(gradient))
        #print(xk)
        print(f_svm(xk + alphak*dk, m, Lambda))
        num_iteration = num_iteration + 1
    print(xk)

def AGM(initial, m, Lambda):
    x_minus = xk = initial
    t_minus = tk = 1
    alpha = 0.5
    yita = 0.5
    tol = 1e-8
    gradient = df(initial, m)
    num_iteration = 0
    
    while np.linalg.norm(gradient) > tol and num_iteration < max_iter:
        betak = 1/tk * (t_minus - 1)
        yk = xk + betak * (xk - x_minus)
        x_bar = yk - alpha * df(yk, m)
        while f_svm(x_bar, m, Lambda) - f_svm(yk, m, Lambda) > -alpha/2 * np.linalg.norm(df(yk, m))**2:
            alpha = alpha * yita
            x_bar = yk - alpha * df(yk, m)
        t_minus = tk
        tk = 0.5 * (1 + np.sqrt(1+4*tk**2))
        x_minus = xk
        xk = x_bar
        gradient = df(xk, m)
        num_iteration = num_iteration + 1
        print(np.linalg.norm(gradient))
    print(xk)

def BFGS(initial, m, Lambda):
    Hk = np.identity(n+1)
    xk = initial
    s = 1
    sigma = 0.5
    gamma = 0.1
    tol = 1e-4
    gradient = df(initial, m)
    num_iteration = 0
    
    while np.linalg.norm(gradient) > tol and num_iteration < max_iter:
        alphak = s
        dk = -np.dot(Hk, df(xk, m))
        while True:
            if f_svm(xk + alphak*dk, m, Lambda) - f_svm(xk, m, Lambda) <= gamma * alphak * np.dot(df(xk, m).T, dk):
                break
            alphak = alphak * sigma
        xk_new = xk + alphak * dk
        
        sk = xk_new - xk
        yk = df(xk_new, m) - df(xk, m)
        if np.dot(sk.T, yk) <= 1e-14:
            pass
        else:
            Hk = Hk + np.dot((sk - np.dot(Hk, yk)), sk.T)/(np.dot(sk.T, yk)) - np.dot((sk-np.dot(Hk, yk)).T, yk)/(np.dot(sk.T, yk)**2) * np.dot(sk, sk.T)
        num_iteration = num_iteration + 1
        xk = xk_new
        gradient = df(xk, m)
        print(np.linalg.norm(gradient))
        
    print(xk)
    
# Main begin
n = 2 #number of features
delta = 1e-3
Lambda = 0.1
max_iter = 1000

data = pd.read_csv('./dataset2.csv', header=None)
a = np.array(data.iloc[:, 0:2])
b = np.array(data.iloc[:, 2])

initial = np.zeros((n+1, 1)) #the last element is y
m = b.size

#gradient_method(initial, m, Lambda)
#AGM(initial, m, Lambda)
BFGS(initial, m, Lambda)




'''
m1 = 1000
m2 = 1000
def plot_data():
    s = 0.7 # control the point size
    x = a[0:m1, 0]
    y = a[0:m1, 1]
    plt.scatter(x, y, c='purple', s=s)
    
    x = a[m1:, 0]
    y = a[m1:, 1]
    plt.scatter(x, y, c='orange', s=s)
plot_data()

A = 1.23080664
B = -1.27891814
C = 1.39519902
x = a[:,0]
y = (-A*x-C)/B
plt.plot(x, y, color='red')
'''




