import imp
import torch

x =  torch.rand(5,4)
y =  torch.rand(3,7)
print(x)
print(x[:,2])
print(y)
print(y[1,:])
print(y[1,2].item())
print(y[1][2])

z = x.view(5*4)
print(z)
z = x.view(-1, 2)
print(z)

import numpy as np
a = torch.ones(5, dtype=torch.int)
print(a)
b = a.numpy()
print(b)