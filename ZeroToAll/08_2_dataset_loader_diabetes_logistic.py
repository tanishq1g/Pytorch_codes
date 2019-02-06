import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os

# https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html
class DiabetesDataset(Dataset):
    '''
    All other datasets should subclass Dataset class. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    '''
    def __init__(self):
        # download and read data
        dir_path = os.path.dirname(os.path.realpath(__file__))
        xy = np.loadtxt(dir_path + '/data/diabetes.csv.gz', delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 0:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        # returns one item on the index
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # returns length of the dataset
        return self.len


# dataset object

dataset = DiabetesDataset()

# https://pytorch.org/docs/stable/data.html
# dataloader Combines a dataset and a sampler, and provides single- or multi-process iterators over the dataset.
train_loader = DataLoader(
    dataset = dataset,
    batch_size = 32,
    shuffle = True,
    num_workers = 2
)

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(8, 6)
        self.l2 = torch.nn.Linear(6, 4)
        self.l3 = torch.nn.Linear(4, 1)
    
    def forward(self, x):
        out1 = torch.sigmoid(self.l1(x))
        out2 = torch.sigmoid(self.l2(out1))
        y_pred = torch.sigmoid(self.l3(out2))
        return y_pred

# our model
model = Model()


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.BCELoss(reduction = 'mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


# Training loop
for epoch in range(2):
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(inputs)

        # Compute and print loss
        loss = criterion(y_pred, labels)
        print(epoch, i, loss.data.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()