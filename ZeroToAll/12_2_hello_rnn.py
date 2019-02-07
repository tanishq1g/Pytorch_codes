import sys
import torch
import torch.nn as nn

# https://pytorch.org/docs/stable/nn.html#rnn

# input of shape (seq_len, batch, input_size): if batch_first is false
# input of shape (batch, seq_length, input_size): if batch_first is false

# h_0 of shape (num_layers * num_directions, batch, hidden_size)
# output of shape (seq_len, batch, num_directions * hidden_size)
# h_n (num_layers * num_directions, batch, hidden_size)

# https://pytorch.org/docs/stable/torch.html#random-sampling
torch.manual_seed(777)  # reproducibility

#            0    1    2    3    4
idx2char = ['h', 'i', 'e', 'l', 'o']
one_hot_lookup = [[1, 0, 0, 0, 0],  # 0
                  [0, 1, 0, 0, 0],  # 1
                  [0, 0, 1, 0, 0],  # 2
                  [0, 0, 0, 1, 0],  # 3
                  [0, 0, 0, 0, 1]]  # 4   , shape = (5, 5)

# Teach hihell -> ihello

x_data = [0, 1, 0, 2, 3, 3]   # hihell
x_one_hot = [one_hot_lookup[x] for x in x_data]

y_data = [1, 0, 2, 3, 3, 4]   # ihello

inputs = torch.Tensor(x_one_hot)    # shape = (6, 5)
print('inputs', inputs, inputs.type(), inputs.shape)

labels = torch.LongTensor(y_data)

num_classes = 5
input_size = 5  # one-hot size
hidden_size = 5  # output from the RNN. 5 to directly predict one-hot
batch_size = 1   # one sentence
sequence_length = 1  # One by one
num_layers = 1  # one-layer rnn

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size = input_size, hidden_size = hidden_size, batch_first = True)

    def forward(self, x, hidden):
        # Reshape input (batch first)
        x = x.view(batch_size, sequence_length, input_size)
        # Propagate input through RNN
        # Input: (batch, seq_len, input_size)
        # hidden: (num_layers * num_directions, batch, hidden_size)
        out, hidden = self.rnn(x, hidden)
        return hidden, out.view(-1, num_classes)
    
    def init_hidden(self):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size)
        return torch.zeros((num_layers, batch_size, hidden_size), requires_grad = True)

# Instantiate RNN model
model = Model()
print(model)

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Train the model
for epoch in range(100):
    optimizer.zero_grad()

    loss = 0
    hidden = model.init_hidden()
    sys.stdout.write("predicted string: ")

    for input, label in zip(inputs, labels):
        hidden, output = model(input, hidden)
        val, idx = output.max(1)
        sys.stdout.write(idx2char[idx.data])
        label  = label.view(-1)
        # input(N, C), target(C) => output(N)
        loss += criterion(output, label)
        # print('output', output, output.type(), output.shape)
        # print('label', label, label.type(), label.shape)
    
    print(", epoch: %d, loss: %1.3f" % (epoch + 1, loss))

    loss.backward()
    optimizer.step()

print("Learning finished!")