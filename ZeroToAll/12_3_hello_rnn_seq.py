import torch
import torch.nn as nn

# link to understand dimensions
# https://discuss.pytorch.org/t/why-3d-input-tensors-in-lstm/4455

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
sequence_length = 6  # One by one
num_layers = 1  # one-layer rnn

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size = input_size, hidden_size = hidden_size, batch_first = True)

    def forward(self, x):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size)
        h_0 = torch.zeros((num_layers, batch_size, hidden_size), requires_grad = True)

        # Reshape input
        x = x.view(batch_size, sequence_length, input_size)

        # Propagate input through RNN
        # Input: (batch, seq_len, input_size)
        # h_0: (num_layers * num_directions, batch, hidden_size)
        out, _ = self.rnn(x, h_0)
        return out.view(-1, num_classes)

# Instantiate RNN model
model = Model()
print(model)

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(100):
    outputs = model.forward(inputs)
    optimizer.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    _, idx = outputs.max(1)
    idx = idx.data.numpy()
    result_str = [idx2char[c] for c in idx.squeeze()]
    print("epoch: %d, loss: %1.3f" % (epoch + 1, loss))
    print("Predicted string: ", ''.join(result_str))

print("Learning finished!")