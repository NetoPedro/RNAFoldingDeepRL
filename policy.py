import torch.nn as nn
import torch.nn.functional as F

# Mock policy for testing

class Policy(nn.Module):
    # TODO Change to Policy + Value
    # TODO add sigmoid layer
    # TODO Input Sequence + Structure?
    # TODO One-hot encoded input or other representation ?
    def __init__(self,input_size):
        super().__init__()

        #Fully connected layer
        self.fc1 = nn.Linear(input_size,100)
        self.output = nn.Linear(100,2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x






