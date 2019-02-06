import torch
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))
print(x_data, x_data.shape)
print(y_data, y_data.shape)

# https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html
class Model(torch.nn.Module):
    
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super().__init__()
        # https://pytorch.org/docs/stable/nn.html#linear-layers
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x)
        return y_pred

# our model

model = Model()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.

# https://pytorch.org/docs/stable/nn.html#mseloss
criterion = torch.nn.MSELoss(reduction = 'sum')
# optimizer object, will hold the current state and will update the parameters based on the computed gradients.
# https://pytorch.org/docs/stable/optim.html#optimizer-step
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
print(criterion)
print(optimizer)
print(model.parameters())

# Training loop
for epoch in range(700):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model.forward(x_data)

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data.item())
    # print(epoch, loss.data[0])

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# After training
hour_var = Variable(torch.Tensor([[4.0]]))
y_pred = model(hour_var)
print("predict (after training)",  4, model(hour_var).data[0][0])