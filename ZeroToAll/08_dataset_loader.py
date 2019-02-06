# one epoch : one forward pass or one backward pass for all training examples
# batch size : the number of training examples on one forward/backward pass. the higher the batch size, the more memory space you will need
# number of iterations : number of passes, each pass using[batch size] number of examples
# one pass = one backward pass and one forward pass
# example : if 1000 examples and batchsize is 500 it will take 2 iterations to complete one epoch

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

for epoch in range(2):
    for i, data in enumerate(train_loader):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # Run your training process
        print(epoch, i, "inputs", inputs.data.shape, "labels", labels.data.shape)