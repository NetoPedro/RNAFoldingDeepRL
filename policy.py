import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Mock policy for testing

class Policy(nn.Module):
    # TODO add sigmoid layer
    # TODO Input Sequence + Structure?
    # TODO One-hot encoded input or other representation ?
    def __init__(self,input_size):
        super().__init__()

        # Input input_size (10) * 8 (2 * 4 nucleoids)
        # Output  input_size  * 20
        self.conv1 = nn.Conv1d(8,20,4,1,padding=2)

        # Input  input_size  * 20
        # Output  input_size  * 20
        self.conv2 = nn.Conv1d(20,20,4,1,padding=2)
        # Input  input_size  * 20
        # Output  input_size  * 20
        self.conv3 = nn.Conv1d(20,20,2,1,padding=1)

        # Input  input_size  * 20
        # Output  input_size  * 20
        self.conv4 = nn.Conv1d(20,20,4,1,padding=2)
        # Input  input_size  * 20
        # Output  input_size  * 20
        self.conv5 = nn.Conv1d(20,20,2,1,padding=1)

        # Input  input_size  * 20
        # Output  input_size  * 8
        self.conv5 = nn.Conv1d(20,20,2,1,padding=1)

        self.value_layer = nn.Linear(20*input_size,1)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        y = F.relu(self.conv2(x))
        y = F.relu(self.conv3(y))
        x = x + y

        y = F.relu(self.conv4(x))
        y = F.relu(self.conv5(y))
        x = y + x


        action = F.relu(self.conv6(x))
        action = F.sigmoid(action)
        x = x.view(-1,1,1)
        value = self.value_layer(x)
        return value,x






