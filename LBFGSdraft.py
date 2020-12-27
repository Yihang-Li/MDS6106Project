import numpy as np
import pandas as pd


def f_logit(xk, m, Lambda):
    x, y = xk[:-1], xk[-1][0]
    Sum = 0
    for i in np.arange(m):
        Log = np.log(1 + np.exp(-b[i] * (a[i].dot(x) + y)))
        Sum += Log
    return Lambda / 2 * np.linalg.norm(x) ** 2 + Sum / m


def df_logit(xk, m, Lambda):
    x = xk[:-1]
    y = xk[-1][0]
    grad = np.zeros((x.size + 1, 1))
    # derivative for x1-xn
    for k in np.arange(x.size):  # df_xk
        Sum = 0
        for i in np.arange(m):
            df_Log = np.exp(-b[i] * (a[i].dot(x) + y)) / (1 + np.exp(-b[i] * (a[i].dot(x) + y))) * (-b[i] * a[i])[k]
            Sum = Sum + df_Log
        grad[k] = Lambda * x[k] + Sum / m

    # derivative for y
    Sum = 0
    for i in np.arange(m):
        df_Log = np.exp(-b[i] * (a[i].dot(x) + y)) / (1 + np.exp(-b[i] * (a[i].dot(x) + y))) * (-b[i])
        Sum = Sum + df_Log
    grad[x.size] = Sum / m
    return grad


mLBFGS = 5


def LBFGS(initial, m, Lambda, mLBFGS):
    s = 1
    sigma = 0.5
    gamma = 0.1

    tol = 1e-8
    num_iteration = 0

    xk = initial
    Hk0 = np.identity(n + 1)
    gradient = df_logit(initial, m, Lambda)
    dk_new = -np.dot(Hk0, gradient)

    # dict for two loop
    s_list = []
    y_list = []

    while np.linalg.norm(gradient) > tol:

        Hk = Hk0
        alphak = s
        dk = dk_new

        # backtracking to find alphak
        while True:
            if f_logit(xk + alphak * dk, m, Lambda) - f_logit(xk, m, Lambda) <= gamma * alphak * np.dot(
                    df_logit(xk, m, Lambda).T, dk):
                break
            alphak = alphak * sigma

        xk_new = xk + alphak * dk

        sk_1 = xk_new - xk
        yk_1 = df_logit(xk_new, m, Lambda) - df_logit(xk, m, Lambda)

        # 更新sk-1,yk-1
        if num_iteration < mLBFGS:
            s_list.append(sk_1)
            y_list.append(yk_1)
        else:
            for i in range(mLBFGS - 1):
                s_list[i] = s_list[i + 1]
                y_list[i] = y_list[i + 1]
            s_list[mLBFGS - 1] = sk_1
            y_list[mLBFGS - 1] = yk_1

        if np.dot(sk_1, yk_1) > 1e-14:
            Hk0 = np.dot((np.dot(sk_1, yk_1) / np.dot(yk_1, yk_1)), np.identity(n + 1))  # Hk0 for k+1
        else:
            Hk0 = np.identity(n + 1)
        num_iteration += 1
        xk = xk_new
        gradient = df_logit(xk, m, Lambda)

        # two loop recursion
        alpha_i = {}
        q = gradient
        for i_loop1 in reversed(range(len(s_list))):
            rho_i = 1 / np.dot(s_list[i_loop1], y_list[i_loop1])
            alpha_i[i_loop1] = rho_i * np.dot(s_list[i_loop1], q)
            q = q - np.dot(alpha_i[i_loop1], y_list[i_loop1])
        r = np.dot(Hk0, q)
        for i_loop2 in range(len(s_list)):
            rho_i = 1 / np.dot(s_list[i_loop1], y_list[i_loop1])
            beta = rho_i * np.dot(y_list[i_loop1], r)
            r = r + np.dot(alpha_i[i_loop2] - beta, s_list[i_loop2])
        dk_new = -r

        print(xk)


n = 2 #number of features
delta = 1e-3
Lambda = 0.1
max_iter = 1000
#
data = pd.read_csv('dataset2.csv', header=None)
a = np.array(data.iloc[:, 0:2])
b = np.array(data.iloc[:, 2])

initial = np.zeros((n+1, 1)) #the last element is y
m = b.size
LBFGS(initial,m,Lambda,mLBFGS)