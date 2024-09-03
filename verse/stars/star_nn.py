import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        
        return x

input_size = 1     # Number of input features -- this should change to reflect dimensions of starset 
hidden_size = 25     # Number of neurons in the hidden layers -- this may change, I know NeuReach has this at default 64
output_size = 1     # Number of output neurons -- this should stay 1 until nn outputs V instead of mu, whereupon it should reflect dimensions of starset

model = SimpleNN(input_size, hidden_size, output_size)

criterion = nn.MSELoss() # this eventually needs to change

# Use SGD as the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 100 # sample number of epoch -- can play with this/set this as a hyperparameter

# Toy Function to learn: x^2+20
to_learn = lambda x : x**2+20

inputs = torch.randn(50, 1)  # 50 inputes to lear
labels = inputs * inputs + 20 # apply_ is inplace which I don't want, just use torch multiplication

# Training loop
for epoch in range(num_epochs):
    # Zero the parameter gradients
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(inputs)
    
    # Compute the loss
    loss = criterion(outputs, labels)
    
    # Backward pass and optimize
    loss.backward()
    optimizer.step()
    
    # Print loss periodically
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    
# test the new model
test_inputs = torch.randn(25, 1)
test_lables = test_inputs*test_inputs + 20

model.eval()

with torch.no_grad():
    test_outputs = model(test_inputs)
    test_loss = criterion(test_outputs, test_lables)

print(f'Test Loss: {test_loss.item()}')
print(test_lables, test_outputs, test_inputs, inputs)