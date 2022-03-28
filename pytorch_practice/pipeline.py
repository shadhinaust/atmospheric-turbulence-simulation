import torch
import torch.nn as nn

from gd_with_autograd import forward

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

n_sample, n_features = X.shape

input_size = n_features
output_size =  n_features

X_test = torch.tensor([[5]], dtype=torch.float32)

model = nn.Linear(input_size, output_size)

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.lin(x)

linear_model = LinearRegression(input_size, output_size)

print(f'Prediction before training: f(5) = {linear_model(X_test).item():.3f}')

learning_rate = 0.1
n_iters = 1000
loss = nn.MSELoss()
optimizer = torch.optim.SGD(linear_model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    y_pred = linear_model(X)
    l = loss(Y, y_pred)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 100 == 0:
        [w,b] = linear_model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {linear_model(X_test).item():.3f}')