import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
n_epochs = 4 # MAX = 19; 500*19 ~= total samples 
batch_size = 4
learning_rate = 0.001

# dataset has PILImage images of range [0,1].
# we transform them to tensors of normalized range [-1,1]
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.LFWPeople(root='./data', split='train', image_set='deepfunneled', download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.LFWPeople(root='./data', split='test', image_set='deepfunneled', transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# print(len(train_loader))
# examples = iter(train_loader)
# example = examples.next()
# images, labels = example
# print(len(images), len(labels))
# images, labels = example

# conv_1 = nn.Conv2d(3, 6, 25)
# pool = nn.MaxPool2d(4, 4)
# conv_2 = nn.Conv2d(6, 16, 25)

# print(images.shape, labels.shape)

# images = conv_1(images)
# print(images.shape)

# images = pool(images)
# print(images.shape)

# images = conv_2(images)
# print(images.shape)

# images = pool(images)
# print(images.shape)
# # 16*8*8


# for i, (images, labels) in enumerate(train_loader):
#     images = images.to(device)
#     labels = labels.to(device)
#     print(labels)

class ConvolutionNeuNet(nn.Module):
    def __init__(self):
        super(ConvolutionNeuNet, self).__init__()
        self.conv_1 = nn.Conv2d(3, 6, 25)
        self.pool = nn.GaussianNLLLoss(4,4)
        self.conv_2 = nn.Conv2d(6, 16, 25)
        self.fc_1 = nn.Linear(16*8*8, 1024)
        self.fc_2 = nn.Linear(1024, 128)
        self.fc_3 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv_1(x)))
        x = self.pool(F.relu(self.conv_2(x)))
        x = x.view(-1, 16*8*8)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x[0]

model = ConvolutionNeuNet()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(n_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(torch.float32).to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print(f'epoch {epoch+1}/{n_epochs}, step {i+1}/{n_total_steps}, loss {loss.item():.2f}')

with torch.no_grad():
    n_samples = 0
    n_correct = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs[0], 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
    
    acc = 100.0 * n_correct/n_samples
    print(f'accuracy = {acc:.2f}%')