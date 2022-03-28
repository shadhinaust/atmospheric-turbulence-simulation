import torch
import torch.nn as nn
import numpy as np

def cross_entropy(actual, predicted):
    loss = -np.sum(actual*np.log(predicted))
    return loss

Y = np.abs([1,0,0])

Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
loss_1 = cross_entropy(Y, Y_pred_good)
loss_2 = cross_entropy(Y, Y_pred_bad)
print(f'Loss 1 numpy: {loss_1:.4}')
print(f'Loss 2 numpy: {loss_2:.2}')

loss = nn.CrossEntropyLoss()
torch
Y = torch.tensor([2,0,1])
Y_pred_good = torch.tensor([[0.7,0.2,0.1], [0.9,0.1,0.0], [0.5,0.1,0.4]])
Y_pred_bad =  torch.tensor([[0.1,0.3,0.6], [0.2,0.5,0.3], [0.1,0.4,0.5]])
loss_1 = loss(Y_pred_good, Y)
loss_2 = loss(Y_pred_bad, Y)
print(f'Loss 1 tensor: {loss_1.item():.2}')
print(f'Loss 2 tensor: {loss_2.item():.2}')

_, pred_1 = torch.max(Y_pred_good, 1)
_, pred_2 = torch.max(Y_pred_bad, 1)

print(pred_1)
print(pred_2)