import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.linear_1(x)
        out = self.relu(out)
        out = self.linear_2(out)
        y_pred = torch.sigmoid(out)
        return y_pred

model = NeuralNet(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss()