import torch
import math

from torch import tanh
import matplotlib.pyplot as plt

n = 512
x = torch.randn(n).normal_(0,1)#.uniform_(0, 1)
for i in range(100):
    #a = torch.Tensor(512,512).normal_(0,1) * math.sqrt(1./512)
    #a = torch.eye(n,n) #* (math.sqrt(n)) #torch.randn(n,n) * 1/(math.sqrt(n))
    a = torch.randn(n,n).normal_(0,1)# * 1/(math.sqrt(n))
    #b = torch.randn(n,n).normal_(0,1)# * 1/(math.sqrt(n))
    #b = torch.eye(n,n) #* (math.sqrt(n)) #torch.randn(n,n) * 1/(math.sqrt(n))
    #b = torch.Tensor(512,512).normal_(0,1) * math.sqrt(1./512)
    x = (a @ x)**2 #(a @ x) * (b @ x) #* (b @ x) #tanh(a @ x)
    #x = x.clamp(-100, 100)
    #x = 1-torch.exp(-x/2)
    #x = (a @ x)


    print("mean:" + str(float(x.mean())))
    print("std:" + str(float(x.std())))
    plt.hist(x.numpy(), 150)
    plt.show()