import torch
from torch import nn
class comLSTM(nn.Module):
    def __init__(self,n_features,hidden_dim=256,output_size=1,num_layers = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features,hidden_size=hidden_dim,batch_first=True,num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim,output_size)
    def forward(self,x):
        x= x.to(torch.float32)
        x, _ = self.lstm(x)
        x = x[:,-1,:]
        x = self.fc(x)
        return x