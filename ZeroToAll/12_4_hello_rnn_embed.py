import torch
import torch.nn as nn

def print_tensor(T, name = 'tensor', val = True):
    if(val):
        print(name, T, T.type(), T.shape)
    else:
        print(name, T.type(), T.shape)

torch.manual_seed(777)  # reproducibility

idx2char = ['h', 'i', 'e', 'l', 'o']

# Teach hihell -> ihello
x_data = [0, 1, 0, 2, 3, 3]   # hihell
y_data = [1, 0, 2, 3, 3, 4]    # ihello

inputs = torch.LongTensor(x_data)    # shape = (6, 5)
print_tensor(inputs, 'inputs')

labels = torch.LongTensor(y_data)

num_classes = 5
input_size = 5
embedding_size = 10  # embedding size
hidden_size = 5  # output from the LSTM. 5 to directly predict one-hot
batch_size = 1   # one sentence
sequence_length = 6  # |ihello| == 6
num_layers = 1  # one-layer rnn

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        # num_embeddings, embedding_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.RNN(input_size = embedding_size, hidden_size = 5, batch_first = True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size)
        h_0 = torch.zeros((num_layers, batch_size, hidden_size), requires_grad = True)

        # input: LongTensor of arbitrary shape containing the indices to extract
        # output : (*, embedding_dim), where * is the input shape
        emb = self.embedding(x)
        emb = emb.view(batch_size, sequence_length, embedding_size)
        print_tensor(emb, 'emb', False)

        # Propagate embedding through RNN
        # Input: (batch, seq_len, embedding_size)
        # h_0: (num_layers * num_directions, batch, hidden_size)
        out, _ = self.rnn(emb, h_0)
        return self.fc(out.view(-1, num_classes))

# Instantiate RNN model
model = Model()
print(model)

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(100):
    outputs = model(inputs)
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