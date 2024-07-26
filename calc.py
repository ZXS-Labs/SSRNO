import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt 
inp = torch.from_numpy(np.linspace(-5,5,50000))
gt = nn.GELU()(inp)
predf = np.polyfit(inp,gt,15)
p = np.poly1d(predf)
# print(p)
# plt.figure()
# plt.plot(inp,gt,color='red')
# plt.plot(inp,p(inp),color='green')
# plt.savefig("function.png")
a = [3.414e-22,1.068e-09,-3.913e-20,-1.503e-07,1.804e-18,8.671e-06,-4.325e-17,-0.0002655,5.736e-16,0.004689,-3.94e-15,-0.04958,1.014e-14,0.3766,0.5,0.005081]
def hf(a_n):
    n = len(a_n)-1
    def b_n(i):
        if i == n: return "((scalar_t) "+str(a_n[i])+")"
        return "((scalar_t) "+str(a_n[i])+")+("+b_n(i+1)+")*inp"
    return b_n(0)
a.reverse()
print(hf(a))
exit()
# ((scalar_t) 0.01224)+(((scalar_t) 0.5)+(((scalar_t) 0.3557)+(((scalar_t) -3.898e-15)+(((scalar_t) -0.03971)+(((scalar_t) 7.389e-16)+(((scalar_t) 0.002953)+(((scalar_t) -6.752e-17)+(((scalar_t) -0.0001209)+(((scalar_t) 3.016e-18)+(((scalar_t) 2.511e-06)+(((scalar_t) -6.32e-20)+(((scalar_t) -2.067e-08)+(((scalar_t) 4.972e-22))*inp)*inp)*inp)*inp)*inp)*inp)*inp)*inp)*inp)*inp)*inp)*inp)*inp
def sf(x):
    return 0.005081+(0.5+(0.3766+(1.014e-14+(-0.04958+(-3.94e-15+(0.004689+(5.736e-16+(-0.0002655+(-4.325e-17+(8.671e-06+(1.804e-18+(-1.503e-07+(-3.913e-20+(1.068e-09+(3.414e-22)*x)*x)*x)*x)*x)*x)*x)*x)*x)*x)*x)*x)*x)*x)*x
plt.figure()
plt.plot(inp,gt,color='red')
plt.plot(inp,sf(inp),color="blue")
plt.savefig("function.png")
