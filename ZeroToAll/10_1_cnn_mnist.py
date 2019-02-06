import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
# https://pytorch.org/docs/stable/autograd.html#tensor-autograd-functions
# Training settings
batch_size = 64

# MNIST data
train_dataset = datasets.MNIST(
    root = 'data/mnist_data/',
    train = True,
    transform = transforms.ToTensor(),
    download = True
)
test_dataset = datasets.MNIST(
    root = 'data/mnist_data/',
    train = False,
    transform = transforms.ToTensor()
)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        # https://pytorch.org/docs/stable/nn.html#conv2d
        self.conv1 = nn.Conv2d(1, 10, kernel_size = 5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 5)
        # https://pytorch.org/docs/stable/nn.html#maxpool2d
        self.mp = nn.MaxPool2d(2)
        # https://pytorch.org/docs/stable/nn.html#linear
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        return F.log_softmax(x)

model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        print(data.type(), target.type())
        data, target = data, target
        data.requires_grad_()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print(data.requires_grad, target.requires_grad, output.requires_grad)
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data, target
        data.requires_grad_()
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, reduction = 'sum').data.item()
        # get the index of the max log-probability
        # https://pytorch.org/docs/stable/torch.html?highlight=torch.max#torch.max
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 10):
    train(epoch)
    test() 
