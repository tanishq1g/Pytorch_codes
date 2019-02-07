import torch
import torch.nn as nn
from torch.autograd import Variable

# important q/a
# https://discuss.pytorch.org/t/autograd-require-grad-true/20847

# One hot encoding for each char in 'hello'
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

# One cell RNN input_dim (4) -> output_dim (2). sequence: 5
# https://pytorch.org/docs/stable/nn.html#rnn

# for seq length = 1, batch_size = 1
cell = nn.RNN(input_size = 4, hidden_size = 2, batch_first = True)

# one letter input
inputs = torch.FloatTensor([[h]]) # rank = (1, 1, 4)

# intialize the hidden state
hidden = torch.randn((1, 1, 2))

# feed one element at a time
# after each step, hidden contains the hidden state
out, hidden = cell(inputs, hidden)
print('out', out, out.type(), out.shape)


# for seq length = 5, batch_size = 1
cell = nn.RNN(input_size = 4, hidden_size = 2, batch_first = True)

inputs = torch.Tensor([[h, e, l, l, o]])  # shape = (1, 5, 4)
print('inputs', inputs.type(), inputs.shape)

hidden = torch.randn((1, 1, 2))

out, hidden = cell(inputs, hidden)
print('out', out, out.type(), out.shape)


# for seq length = 5, batch_size = 3
cell = nn.RNN(input_size = 4, hidden_size = 2, batch_first = True)

inputs = torch.Tensor([[h, e, l, l, o],
                       [e, o, l, l, l],
                       [l, l, e, e, l]])  # shape = (3, 5, 4)
print('inputs', inputs.type(), inputs.shape)

hidden = torch.randn((1, 3, 2))

out, hidden = cell(inputs, hidden)
print('out', out, out.type(), out.shape)

