import torch
import torch.nn as nn
from torch.autograd import Variable

# One hot encoding for each char in 'hello'
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

# One cell RNN input_dim (4) -> output_dim (2). sequence: 5
# https://pytorch.org/docs/stable/nn.html#rnn
cell = nn.RNN(input_size = 4, hidden_size = 2, batch_first = True)

# one letter input
inputs = torch.Tensor([[h]], requires_grad = True) # rank = (1, 1, 4)

# intialize the hidden state
hidden = torch.randn((1, 1, 2), requires_grad = True)

# feed one element at a time
# after each step, hidden contains the hidden state
out, hidden = cell(inputs, hidden)
print('out', out.data)