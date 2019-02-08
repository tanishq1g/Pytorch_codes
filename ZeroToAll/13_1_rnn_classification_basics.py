import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def print_tensor(T, name = 'tensor', val = True):
    if(val):
        print(name, T, T.type(), T.shape)
    else:
        print(name, T.type(), T.shape)

# Parameters and DataLoaders
HIDDEN_SIZE = 100
N_CHARS = 128  # ASCII
N_CLASSES = 18


class RNN_classifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # we run this all at once (over the whole input sequence)

        batch_size = 1
        emb = self.embedding(x)
        emb = emb.view(-1, batch_size, self.hidden_size)
        print_tensor(emb, 'emb', False)

        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        print_tensor(hidden, 'h_0', False)
        
        output, hidden = self.gru(emb, hidden)
        print_tensor(output, 'output', False)
        print_tensor(hidden, 'hidden', False)

        fc_output = self.fc(hidden)
        print_tensor(fc_output, 'fc_output', False)

        return fc_output
        
def str2ascii_arr(msg):
    # ord() : Given a string of length one, return an integer representing the Unicode code point of the character when the argument is a unicode object, or the value of the byte when the argument is an 8-bit string.
    arr = [ord(c) for c in msg]
    return arr, len(arr)

if (__name__ == '__main__'):
    names = ['adylov', 'solan', 'hard', 'san']
    
    classifier = RNN_classifier(N_CHARS, HIDDEN_SIZE, N_CLASSES)

    for name in names:
        print(name)
        arr, _ = str2ascii_arr(name)
        print(arr, _)
        inp = torch.LongTensor(arr)
        print_tensor(inp, 'inp')
        out = classifier(inp)
        print("in", inp.size(), "out", out.size())
 