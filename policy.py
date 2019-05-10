import torch.nn as nn
import torch
import torch.nn.functional as F
import os
# Mock policy for testing

class Policy(nn.Module):
    WEIGHTS_PATH = "./"
    # TODO change sigmoid activation to softmax
    # TODO Stack of inputs ?
    # TODO Reformulate as Actor and Critic networks
    # TODO Input Sequence + Structure?
    # TODO One-hot encoded input or other representation ?
    # TODO Return as an N * N matrix. Better for the network
    def __init__(self,input_size):
        super().__init__()
        kernel = 8
        padding = int((input_size-1)/2)
        if input_size %2 == 0: kernel = 7

        # Input input_size (10) * 8 (2 * 4 nucleoids)
        # Output  input_size  * 20
        self.conv1 = nn.Conv2d(1,4,5,1,padding=2)

        # Input  input_size  * 20
        # Output  input_size  * 20
        self.conv2 = nn.Conv2d(4,8,5,1,padding=2)
        # Input  input_size  * 20
        # Output  input_size  * 20
        self.conv3 = nn.Conv2d(8,4,(kernel,5),1,padding=(padding,2))

        # Input  input_size  * 20
        # Output  input_size  * 20
        self.conv4 = nn.Conv2d(4,1,3,1,padding=1)
        # Input  input_size  * 20
        # Output  input_size  * 20
        kernel = input_size
        if input_size % 2 == 0: kernel = kernel -1
        self.conv5 = nn.Conv2d(1,1,kernel,1,padding=padding)
        self.output = nn.Linear(input_size*input_size,input_size*input_size)
        self.output_activation = nn.Softmax(dim=0)
        self.saved_probs = []

    def forward(self, x):
        size = x.shape[3]
        x = torch.tensor(x).type('torch.FloatTensor')
        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))


        x = F.relu(self.conv4(x))
        x= F.relu(self.conv5(x))
        #x = x.view(-1,1,1,1)
        x = self.output_activation(self.output(x.view(-1)))
        return x

    def save_weights(self, weights_name ):
        weights_fname = weights_name
        weights_fpath = os.path.join(self.WEIGHTS_PATH, weights_fname)
        torch.save({'state_dict': self.state_dict()}, weights_fpath)

    def load_weights(self, weights_name):
        state = torch.load(self.WEIGHTS_PATH +weights_name)
        self.load_state_dict(state['state_dict'])


