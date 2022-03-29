import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import mahotas as mh
from zernik import zernike_reconstruct as zernik

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_epochs = 4 # MAX = 19; 500*19 ~= total samples 
batch_size = 1
learning_rate = 0.001
kernel_size = [31, 31]
sigma = [1, 5]

transform = transforms.Compose([transforms.Resize(size=30),
    transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma),
    transforms.ToTensor()])

train_dataset = torchvision.datasets.LFWPeople(root='./data', split='train', image_set='deepfunneled', download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.LFWPeople(root='./data', split='test', image_set='deepfunneled', transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# def imshow(images):
#     images = images.detach().numpy()
#     plt.imshow(np.transpose(images, (1, 2, 0)))
#     plt.show()

# # dataiter = iter(train_loader)
# # images, _ = dataiter.next()

# # imshow(torchvision.utils.make_grid(images))

# # print(len(train_loader))
# examples = iter(train_loader)
# example = examples.next()
# images, labels = example
# print(len(images), len(labels))
# images, _ = example

# conv_1 = nn.Conv2d(3, 6, 25)
# pool = nn.MaxPool2d(4, 4)
# conv_2 = nn.Conv2d(6, 1, 25)

# print(images.shape)

# images = conv_1(images)
# print(images.shape)

# images = pool(images)
# print(images.shape)

# images = conv_2(images)
# print(images.shape)

# images = pool(images)
# plt.imshow(images[0][0])
# plt.show()
# # images = np.transpose(images, (1, 2, 0))
# image = mh.features.zernike_moments(images[0][0],16)
#     # pl.figure(1)
#     # pl.imshow(img, cmap=cm.jet, origin = 'upper')
#     # pl.figure(2)    
#     # pl.imshow(reconst.real, cmap=cm.jet, origin = 'upper')
# print(image.shape)

# # 16*8*8


# for i, (images, labels) in enumerate(train_loader):
#     images = images.to(device)
#     labels = labels.to(device)
#     print(labels)

# def basic_function():
#     pass

# def coefficient():
#     pass

class ConvolutionNeuNet(nn.Module):
    def __init__(self):
        super(ConvolutionNeuNet, self).__init__()
        self.conv_1 = nn.Conv2d(3, 1, 1)
        # self.pool = nn.MaxPool2d(4,4)
        # self.conv_2 = nn.Conv2d(6, 1, 25)

    def forward(self, x):
        x = self.conv_1(x)
        # x = self.pool(F.relu(self.conv_1(x)))
        # x = self.pool(F.relu(self.conv_2(x)))
        return x[0][0]

model = ConvolutionNeuNet()
for i, (image, _) in enumerate(train_loader):
    image = image.to(device)
    output = model(image).to(device).detach().numpy()
    D = 20
    rows, cols = output.shape
    radius = cols//2 if rows > cols else rows//2
    z_output = zernik(img=output, radius=radius, D=D, cof=(rows/2., cols/2.))
    z_output = z_output.reshape(output.shape).astype(np.float32)
    recon_image = np.multiply(output, z_output)

    fig = plt.figure()  
    fig.add_subplot(1,2, 1)
    plt.imshow(output)
    fig.add_subplot(1,2, 2)
    plt.imshow(recon_image)
    plt.show(block=True)
    
    print(recon_image)