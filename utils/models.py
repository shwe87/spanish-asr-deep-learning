# %% [code]
# Please don't forget to mention that this was created by Shweta but please handle with care, if anything breaks it was not her responsibility.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

from torch import nn
import torch
import torch.nn.functional as F


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch.nn.init as init

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weight_init(m):
    '''
    Snippet from:
    https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
    Usage:
        model = Model()
        model.apply(weight_init)
    '''

    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_dim, dropout, bidir):
        super(LSTMLayer, self).__init__()
        #self.layer_norm = nn.LayerNorm(input_size)
        self.LSTMLayer = nn.LSTM(
            input_size=input_size, hidden_size=hidden_dim,
            num_layers=1, batch_first=True, bidirectional=bidir)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x, hidden = inputs
        x, hidden = self.LSTMLayer(x, hidden)
        #x = self.layer_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x, hidden

class ConvLayer(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, n_feats, dropout, out_channels=32, padding=1):
        super(ConvLayer, self).__init__()
        #self.layer_norm = nn.LayerNorm(n_feats//2)
        self.ConvLayer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        #x = self.layer_norm(x)
        #x = x.transpose(2, 3).contiguous()   # (batch, channel, feature, time) 
        x = self.ConvLayer(x)
        x = F.relu(x)
        #x = F.gelu(x)
        x = self.dropout(x)
        return x

class GRULayer(nn.Module):
    def __init__(self, input_size, hidden_dim, dropout, bidir, apply_layer_norm=False):
        super(GRULayer, self).__init__()
        if apply_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim*(bidir+1))
        self.GRULayer = nn.GRU(
            input_size=input_size, hidden_size=hidden_dim,
            num_layers=1, batch_first=True, bidirectional=bidir)
        self.dropout = nn.Dropout(dropout)
        self.apply_layer_norm = apply_layer_norm

    def forward(self, inputs):
        x, hidden = inputs
        if self.apply_layer_norm:
            x = self.layer_norm(x)
        #x = F.relu(x)
        x, hidden = self.GRULayer(x, hidden)
        #x = F.gelu(x)
        x = self.dropout(x)
        return x, hidden

class OnlyRNN(nn.Module):
    """
    This class will be used to create models like:
    - GRU (can be bidirectional)
    - Fully Connected Network
    - SoftMax --> Classfication problem (29 characters to decode).
    """

    def __init__(self, input_size, n_classes, hidden_dim, n_layers, n_gru_layers=1 ,drop_prob=0.2, apply_layer_norm=False, bidir=False):
        super(OnlyRNN, self).__init__()
        #output_dim = will be the alphabet + '' and space = 28 chars
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.bidir = bidir
        if self.bidir:
            self.n_direction = 2
        else:
            self.n_direction = 1
        self.apply_layer_norm = apply_layer_norm
        layers = []
        for i in range(n_gru_layers):
            if i==0:
                layers.append(GRULayer(input_size=input_size, hidden_dim=hidden_dim,
                                       dropout=drop_prob, bidir=bidir, apply_layer_norm=False))
            else:
                layers.append(GRULayer(input_size=hidden_dim*self.n_direction, hidden_dim=hidden_dim,
                                      dropout=drop_prob, bidir=bidir, apply_layer_norm=apply_layer_norm))
        self.gru_layers = nn.Sequential(*layers)
        # GRU Layer --> input (batch, channel*features, time)
        # Input size = number of features
        # With batch first --> The input is (batch, sequence, features)
        #self.gru = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True, dropout=drop_prob, bidirectional=self.bidir)
        # shape output (batch, time, num_direction * hidden_size)
        #if layer_norm:
        #    self.layer_norm = nn.LayerNorm(self.input_size)
        # (batch, channel, features, time)
        #Fully Connected 
        #if self.bidir:
        self.classifier = nn.Linear(self.hidden_dim*self.n_direction, n_classes)   #hidden_dim*2 if bidirectional
        self.dropout = nn.Dropout(0.2)

        
    def forward(self, x, hidden):
        sizes = x.size()
        x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3]) #(batch, features*channel, time)
        x = x.transpose(1,2)    #(batch, time, features)
        inputs = (x, hidden)
        out, hidden = self.gru_layers(inputs)
        # Fully Connected
        out = self.classifier(out)
        return out, hidden
    
    def init_hidden(self, batch_size, device):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of GRU
        hidden = (torch.zeros(self.n_layers*self.n_direction, batch_size, self.hidden_dim).zero_()).to(device)
        return hidden



    
class ASRConvLSTM(nn.Module):
    """
    This class will be used to create models like:
    - Conv1
    - Conv2
    - Fully Connected Network
    - LSTM
    - Classifier
    - SoftMax --> Classfication problem (28 characters to decode).
    - Layer Normalization justified: https://arxiv.org/pdf/1607.06450.pdf
     output = (seq_len, batch, num_directions * hidden_size)
    """

    def __init__(self, in_channel, lstm_input_size, hidden_dim, n_layers, n_feats, n_classes, conv_n_layers=2, lstm_n_layers=4, drop_prob=0.2, bidir=False):
        super(ASRConvLSTM, self).__init__()
        #output_dim = will be the alphabet + '' and space = 28 chars
        self.in_channel = in_channel
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_feats = n_feats
        self.lstm_input_size = lstm_input_size
        if bidir:
            self.directions = 2
        else:
            self.directions = 1
        
        # 1. Conv2d --> depth=32, kernel size=(4,4), strides=(2,2), padding=1
        # Output size (batch, 32, 64 (features/2), time/2 )
        # First conv without layer normalization
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1)
        layers = []
        for i in range(conv_n_layers):
            layers.append(ConvLayer(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, n_feats=n_feats, dropout=drop_prob))
        self.conv_layers = nn.Sequential(*layers)
        # 2. Conv2d --> in_channel depth=32, out_depth=32, kernel size=(3,3), strides=(1,1), padding=1 --> We won't modify the features vectors anymore
        # Output size (batch, channels=32, 64, time/2)
        # Let's add a fully connected layer to reduce the input size to GRU --> transpose (batch, time/2, 64* channels)
        # Input --> 64* channels
        # Output size --> 512
        self.fully_connected = nn.Linear((n_feats//2)*32, lstm_input_size)
        # LSTM  --> Input size (batch, time/2, feature/2* channels)
        #           Output size (batch, time/2, num_direction * hidden_dimension)
        # Hidden dimension --> Input (num_layers * num_directions, batch, hidden_dim=512)
        #                      Output size (num_layers * num_directions, batch, hidden_dim=512)
        layers = []
        for i in range(lstm_n_layers):
            if (i==0):
                layers.append(LSTMLayer(input_size=lstm_input_size, hidden_dim=hidden_dim, dropout= drop_prob, bidir=bidir))
            else:
                layers.append(LSTMLayer(input_size=hidden_dim*self.directions, hidden_dim=hidden_dim, dropout= drop_prob, bidir=bidir))
        self.lstm_layers = nn.Sequential(*layers)

        # Classifier:
        self.classifier = nn.Linear(self.directions*hidden_dim, n_classes )
        self.dropout = nn.Dropout(0.2)

        
    def forward(self, x, hidden):
        #First Layer --> Conv1
        #print("Before Conv 1: {}".format(x.size()))   #(batch, channel, feature, time)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3]) #(batch, features*channel, time)
        x = x.transpose(1,2)    #(batch, time, features*channel)
        #print("After View & Transpose : {}".format(x.size()))
        x = self.fully_connected(x)
        #print("After Fully Connected : {}".format(x.size()))
        x = F.relu(x)
        #x = F.gelu(x)
        x = self.dropout(x)
        # GRU Bidirectional  (batch, time, gru_input_size)
        inputs = (x, hidden)
        out, hidden = self.lstm_layers(inputs) 
        #print("After GRU : {}".format(x.size()))
        out = self.classifier(out)
        #print("After classifier {}".format(out.size()))
        return out, hidden
    
    def init_hidden(self, batch_size, device):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of GRU            
        # Implement function
        # initialize hidden state with zero weights, and move to GPU if available
        hidden = (torch.zeros(self.n_layers*self.directions, batch_size, self.hidden_dim).zero_().to(device),
                  torch.zeros(self.n_layers*self.directions, batch_size, self.hidden_dim).zero_().to(device))
        #hidden = (torch.zeros(self.n_layers*self.directions, batch_size, self.hidden_dim).zero_()).to(device)
        return hidden
    

    

    
    
class ASRConvBiGRU(nn.Module):
    """
    This class will be used to create models like:
    - Conv1
    - Conv2
    - Fully Connected Network
    - Bidirectional GRU
    - Classifier
    - SoftMax --> Classfication problem (28 characters to decode).
    - Layer Normalization justified: https://arxiv.org/pdf/1607.06450.pdf
    """

    def __init__(self, in_channel, gru_input_size, hidden_dim, n_layers, n_feats, n_classes, conv_n_layers=2, gru_n_layers=4, drop_prob=0.1, bidir=False):
        super(ASRConvBiGRU, self).__init__()
        #output_dim = will be the alphabet + '' and space = 28 chars
        self.in_channel = in_channel
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_feats = n_feats
        self.gru_input_size = gru_input_size
        self.directions = 1
        if bidir:
            self.directions = 2
        
        # 1. Conv2d --> depth=32, kernel size=(4,4), strides=(2,2), padding=1
        # Output size (batch, 32, 64 (features/2), time/2 )
        # 2. Conv2d --> in_channel depth=32, out_depth=32, kernel size=(3,3), strides=(1,1), padding=1 --> We won't modify the features vectors anymore
        # Output size (batch, channels=32, 64, time/2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1)
        layers = []
        
        for i in range(conv_n_layers):
            layers.append(ConvLayer(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, n_feats=n_feats, dropout=drop_prob))
        self.conv_layers = nn.Sequential(*layers)
        # Let's add a fully connected layer to reduce the input size to GRU --> transpose (batch, time/2, 64* channels)
        # Input --> 64* channels
        # Output size --> 512
        self.fully_connected = nn.Linear((n_feats//2)*32, gru_input_size)
        # GRU Bidrectional --> input size (batch, time/2, feature/2* channels)
        # Output size (batch, time/2, 2 * hidden_dimension)
        # Input size of hidden dimension --> (num_layers * num_directions, batch, hidden_dim=512)
        # Output size of hidden dimension --> (num_layers * num_directions, batch, hidden_dim=512)
        layers = []
        for i in range(gru_n_layers):
            if (i==0):
                layers.append(GRULayer(input_size=gru_input_size, hidden_dim=hidden_dim, dropout= drop_prob, bidir=bidir))
            else:
                layers.append(GRULayer(input_size=hidden_dim*self.directions, hidden_dim=hidden_dim, dropout= drop_prob, bidir=bidir))
        self.gru_layers = nn.Sequential(*layers)
        
        # Fully conected layers
        # Classifier:
        self.classifier = nn.Linear(self.directions*hidden_dim, n_classes)
        self.dropout = nn.Dropout(0.2)

        
    def forward(self, x, hidden):
        #First Layer --> Conv1
        x = self.conv1(x)
        x = self.conv_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3]) #(batch, features*channel, time)
        x = x.transpose(1,2)    #(batch, time, features*channel)
        x = self.fully_connected(x)
        #x = F.relu(x)
        x = F.gelu(x)
        x = self.dropout(x)
        # GRU Bidirectional  (batch, time, gru_input_size)
        inputs = (x, hidden)
        #inputs = x
        out, hidden = self.gru_layers(inputs) 
        # Fully connected layers
        out = self.classifier(out)
        return out, hidden
    
    def init_hidden(self, batch_size, device):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of GRU

        hidden = (torch.zeros(self.n_layers*self.directions, batch_size, self.hidden_dim).zero_()).to(device)
        return hidden
    
