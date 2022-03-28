import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples

dataset = WineDataset()

# first_data = dataset[0]
# features, labels = first_data
# print(features, labels)

batch_size = 4
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
dataiter = iter(dataloader)
data = dataiter.next()
features, labels = data
print(features, labels)

# training loop
n_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/batch_size)
print(total_samples, n_iterations)

for epoch in range(n_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward, backward, and update
        # if (i+1) % 5 == 0:
        print(f'epoch {epoch+1}/{n_epochs}, step {i+1}/{n_iterations}, input {inputs.shape}')