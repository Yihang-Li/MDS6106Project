import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def f_logit(xk, m, Lambda):
    x, y = xk[:-1], xk[-1][0]
    Sum = 0
    for i in np.arange(m):
        Log = np.log(1+np.exp(-b[i]*(a[i].dot(x)+y)))
        Sum += Log
    return Lambda/2 * np.linalg.norm(x)**2 + Sum/m

def df_logit(xk, m, Lambda):
    x = xk[:-1]
    y = xk[-1][0]
    grad = np.zeros((x.size+1, 1))
    # derivative for x1-xn 
    for k in np.arange(x.size): #df_xk
        Sum = 0
        for i in np.arange(m):
            df_Log = np.exp(-b[i]*(a[i].dot(x)+y))/(1+np.exp(-b[i]*(a[i].dot(x)+y)))*(-b[i]*a[i])[k]
            Sum = Sum + df_Log
        grad[k] = Lambda * x[k] + Sum/m
    
    # derivative for y
    Sum = 0
    for i in np.arange(m):
        df_Log = np.exp(-b[i]*(a[i].dot(x)+y))/(1+np.exp(-b[i]*(a[i].dot(x)+y)))*(-b[i])
        Sum = Sum + df_Log
    grad[x.size] = Sum/m
    return grad

#gradient method
def gradient_method(initial, m, Lambda):
    s = 1
    sigma = 0.5
    gamma = 0.1
    tol = 1e-8
    
    xk = initial
    gradient = df_logit(initial, m, Lambda)
    num_iteration = 0
    
    while np.linalg.norm(gradient) > tol and num_iteration < max_iter:
        alphak = s
        dk = -df_logit(xk, m, Lambda)
        while True:
            if f_logit(xk + alphak*dk, m, Lambda) - f_logit(xk, m, Lambda) <= gamma * alphak * np.dot(df_logit(xk, m, Lambda).T, dk):
                break
            alphak = alphak * sigma
        
        xk = xk + alphak * dk
        gradient = df_logit(xk, m, Lambda)
        #print(np.linalg.norm(gradient))
        #print(xk)
        print(f_logit(xk + alphak*dk, m, Lambda))
        num_iteration = num_iteration + 1
    print(xk)

def AGM(initial, m, Lambda):
    x_minus = xk = initial
    t_minus = tk = 1
    alpha = 0.5
    yita = 0.5
    tol = 1e-4
    gradient = df_logit(initial, m, Lambda)
    num_iteration = 0
    
    while np.linalg.norm(gradient) > tol and num_iteration < max_iter:
        betak = 1/tk * (t_minus - 1)
        yk = xk + betak * (xk - x_minus)
        x_bar = yk - alpha * df_logit(yk, m, Lambda)
        while f_logit(x_bar, m, Lambda) - f_logit(yk, m, Lambda) > -alpha/2 * np.linalg.norm(df_logit(yk, m, Lambda))**2:
            alpha = alpha * yita
            x_bar = yk - alpha * df_logit(yk, m, Lambda)
        t_minus = tk
        tk = 0.5 * (1 + np.sqrt(1+4*tk**2))
        x_minus = xk
        xk = x_bar
        gradient = df_logit(xk, m, Lambda)
        num_iteration = num_iteration + 1
        print(np.linalg.norm(gradient))
    print(xk)

    
# Main begin
n = 2 #number of features
delta = 1e-3
Lambda = 0.1
max_iter = 1000

data = pd.read_csv('./dataset_csv_files/dataset2.csv', header=None)
a = np.array(data.iloc[:, 0:2])
b = np.array(data.iloc[:, 2])

initial = np.zeros((n+1, 1)) #the last element is y
m = b.size

# gradient_method(initial, m, Lambda)
AGM(initial, m, Lambda)


