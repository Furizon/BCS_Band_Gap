import numpy as np
import random
from matplotlib import pyplot as plt
from tqdm import tqdm


############Constants#############
# S波
PHI = 1
# 临界温度设为1
T_C = 1
# 曲线拟合的系数
alpha = 1.764

##################################
# 牛顿法求根
############Parameters############
# 公式 : x_1 = x_0 - f(x_n) / f'(x_n)
# Function_x : 需要求根的函数
# dFunction_x : 待求根函数的导数，如果留空会自动计算导数
# Initialx : 初始猜测点，缺省值为1.8
# MaxEpochs : 最大迭代次数，默认为1000
#############Return###############
# X[Epoch] : 最终求解的根
#############Errors###############
# RuntimeError : 到达最大迭代次数仍未收敛
##################################
def newtonFindRoot(Function_x, dFunction_x = None, Initialx = 1.8, MaxEpochs = 1000, Tolerance = 1e-6):
    
    # 暂存每一次迭代的根
    X = np.zeros(MaxEpochs)
    X[0] = Initialx
    # 进行迭代
    for epoch in range(1, MaxEpochs):
        # 若传入了导数，就用传入的函数导数
        if dFunction_x == None:
            dF = (Function_x(X[epoch - 1] + 0.005) - Function_x(X[epoch - 1] - 0.005)) / 0.01
        else:
            dF = dFunction_x(X[epoch - 1])
        # 判断导数是否为零，若为零就随机一个数(0, 1]，从头开始迭代
        if(np.abs(dF) < 1e-10):
            X[epoch] = random.random()
            print("The Derivative Of Function is 0, Restart with x = {:.3f}".format(X[epoch]))
            continue
        
        # 更新迭代出的根
        X[epoch] = X[epoch - 1] - Function_x(X[epoch - 1]) / dF
            
        # 判断是否满足收敛条件
        if np.abs(X[epoch] - X[epoch - 1]) < Tolerance:
            return X[epoch]
    raise RuntimeError("Reaching MAX_EPOCHS!")

##################################
# 计算积分
############Parameters############
# Function : 需要积分的函数，需要支持传入数组
# range : 积分范围
#############Return###############
# 积分值
##################################
def integral(Function, range):
    # 积分间隔
    dx = range[1] - range[0]
    return np.sum(Function * dx, axis = 0)

###########Introduction###########
# BCS理论下能隙所满足的函数
############Parameters############
# Delta_k : 能隙
# t : 温度 t = T / T_c
#############Return###############
# 能隙函数计算的结果
##################################
def bcsGapFunction(Delta_k, t):
    
    Sum = 0
    # 求和范围(0 - 1000)
    N = range(1000)
    # 计算求和
    for n in N:
        # 计算Matsubara频率
        Omega_n = (2 * n + 1) * np.pi * t * T_C
        # 计算求和项并累加
        Sum += np.abs(PHI) ** 2 / np.sqrt(((Omega_n / T_C) ** 2) + (np.abs(Delta_k) / T_C) ** 2)
        Sum -= np.abs(PHI) ** 2 * T_C / Omega_n
    
    # 返回能隙函数的结果
    return np.log(t) - 2 * np.pi * t * Sum / np.abs(PHI) ** 2


###########Introduction###########
# 计算熵
############Parameters############
# Delta_n : 能隙
# t : 温度 t = T / T_c
#############Return###############
# 能隙函数计算的结果
##################################
def calEntropyIntergrate(Delta_n, t):
    
    def calEntropy(kxi, Delta_n, t):
        
        # 限制指数小于700，不然计算指数时会溢出
        LimitExp = np.vectorize(lambda x:700 if x > 701 else x)
        # 计算能量
        E_k = np.sqrt(kxi ** 2 + Delta_n ** 2)
        # 把能量带入费米函数计算占据数
        f_E = 1 / (np.exp(LimitExp(alpha / t * E_k)) + 1)
        # 由占据概率计算熵
        Entropy = (1 - f_E) * np.log(1 - f_E) + f_E * np.log(f_E)
        return Entropy
    # 对不同带隙下的熵进行求和
    return -6 * alpha / (np.pi**2) * integral(calEntropy(np.linspace(1e-2, 20, 1000), Delta_n, t), np.linspace(0, 20, 1000))

###########Introduction###########
# 计算BCS理论下能隙、熵以及热容
############Parameters############
# range : 计算范围
#############Return###############
# Delta_n : 能隙
# S_n : 熵
# C_n : 热容
##################################
def calBCSGapAndEntropy(range):
    
    # 能隙、熵以及热容
    Delta_n = np.zeros(range.shape[0])
    S_n = np.zeros(range.shape[0])
    C_n = np.zeros(range.shape[0])
    
    # 计算不同温度下能隙和熵
    for i, t in enumerate(tqdm(range)):
        # 如果大于临界温度，直接让能隙为0
        if t > 1:
            Delta_n[i] = 0
        else:
            # 用牛顿法求能隙函数的根
            Delta_n[i] = newtonFindRoot(lambda delta:bcsGapFunction(delta, t))
        # 计算熵
        S_n[i] = calEntropyIntergrate(Delta_n[i], t)
    
    # 计算热容 dS / dt
    for i, t in enumerate(range):
        dt = range[1] - range[0]
        if i == 0:
            C_n[i] = (S_n[i + 1] - S_n[i]) / dt
        elif i == range.shape[0] - 1:
            C_n[i] = (S_n[i] - S_n[i - 1]) / dt
        else:
            C_n[i] = (S_n[i + 1] - S_n[i - 1]) / dt
    return Delta_n, S_n, C_n

def calAndPlot():
    # 计算能隙、熵和热容
    Delta_n, S_n, C_n= calBCSGapAndEntropy(np.linspace(1e-3, 1.25, 1000))
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(np.linspace(1e-2, 1.25, 1000)[1:-1], Delta_n[1:-1])
    plt.xlabel("$T/T_c$")
    plt.ylabel("$\\frac{\Delta (T)}{\Delta(0)}$")
    plt.subplot(1, 3, 2)
    plt.plot(np.linspace(1e-2, 1.25, 1000)[1:-1], S_n[1:-1])
    plt.xlabel("$T/T_c$")
    plt.ylabel("$S$")
    plt.subplot(1, 3, 3)
    plt.plot(np.linspace(1e-2, 1.25, 1000)[1:-1], C_n[1:-1])
    plt.xlabel("$T/T_c$")
    plt.ylabel("$C$")
    plt.show()

if __name__ == '__main__':

    calAndPlot()