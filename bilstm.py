import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)

    def forward(self, encoder_outputs):
        # 注意：nn.MultiheadAttention期望输入的形状为(L, B, H)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # 进行多头注意力计算，注意此处忽略了attention mask和key padding mask
        attn_output, _ = self.attention(encoder_outputs, encoder_outputs, encoder_outputs)
        # 将输出转回(B, L, H)的形状
        attn_output = attn_output.permute(1, 0, 2)
        return attn_output


class BISA_LSTM(nn.Module):
    def __init__(self, WINDOW_SIZE=1, INPUT_SIZE=54, Hidden_SIZE=1, LSTM_layer_NUM=1, num_heads=8):
        super(BISA_LSTM, self).__init__()
        self.WINDOW_SIZE = WINDOW_SIZE
        self.INPUT_SIZE = INPUT_SIZE
        self.Hidden_SIZE = Hidden_SIZE
        self.LSTM_layer_NUM = LSTM_layer_NUM
        self.BN = nn.BatchNorm1d(self.WINDOW_SIZE)
        self.lstm = nn.LSTM(input_size=INPUT_SIZE,
                            hidden_size=Hidden_SIZE,
                            num_layers=LSTM_layer_NUM,
                            batch_first=True,
                            bidirectional=True)
        self.attention = MultiHeadAttention(Hidden_SIZE * 2, num_heads)  # 注意因为是双向LSTM，所以隐藏层维度要乘以2
        self.out = nn.Sequential(nn.Linear(Hidden_SIZE * 2, 4), nn.Softmax(dim=1))

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.BN(x)
        r_out, hidden = self.lstm(x, None)  # x(batch,time_step,input_size)
        r_out = self.attention(r_out)
        out = self.out(r_out[:, -1, :])  # 只取最后一个时间步的输出作为全连接层的输入
        return out
