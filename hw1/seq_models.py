"""
   seq_models.py
   COMP9444, CSE, UNSW
"""

import torch
import torch.nn as nn
import math


class SRN_model(nn.Module):
    def __init__(self, num_input, num_hid, num_out, batch_size=1):
        super().__init__()
        self.num_hid = num_hid
        self.batch_size = batch_size
        self.H0= nn.Parameter(torch.Tensor(num_hid))
        self.W = nn.Parameter(torch.Tensor(num_input, num_hid))
        self.U = nn.Parameter(torch.Tensor(num_hid, num_hid))
        self.hid_bias = nn.Parameter(torch.Tensor(num_hid))
        self.V = nn.Parameter(torch.Tensor(num_hid, num_out))
        self.out_bias = nn.Parameter(torch.Tensor(num_out))
        

    def init_hidden(self):
        H0 = torch.tanh(self.H0)
        return(H0.unsqueeze(0).expand(self.batch_size,-1))
 
    def forward(self, x, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        batch_size, seq_size, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t = self.init_hidden().to(x.device)
        else:
            h_t = init_states
            
        for t in range(seq_size):
            x_t = x[:, t, :]
            c_t = x_t @ self.W + h_t @ self.U + self.hid_bias
            h_t = torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from (sequence, batch, feature)
        #           to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0,1).contiguous()
        output = hidden_seq @ self.V + self.out_bias
        return hidden_seq, output

class LSTM_model(nn.Module):
    def __init__(self,num_input,num_hid,num_out,batch_size=1,num_layers=1):
        super().__init__()
        self.num_hid = num_hid
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.W = nn.Parameter(torch.Tensor(num_input, num_hid * 4))
        self.U = nn.Parameter(torch.Tensor(num_hid, num_hid * 4))
        self.hid_bias = nn.Parameter(torch.Tensor(num_hid * 4))
        self.V = nn.Parameter(torch.Tensor(num_hid, num_out))
        self.out_bias = nn.Parameter(torch.Tensor(num_out))
        self.init_weights()
        self.context = self.hid = []
        self.i_t = self.f_t = self.g_t = self.o_t = []

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.num_hid)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def init_hidden(self):
        return(torch.zeros(self.num_layers, self.batch_size, self.num_hid),
               torch.zeros(self.num_layers, self.batch_size, self.num_hid))

    def forward(self, x, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        batch_size, seq_size, _ = x.size()
        hidden_seq = []
        self.context = self.hid = []
        self.i_t = self.f_t = self.g_t = self.o_t = []
        if init_states is None:
            h_t, c_t = (torch.zeros(batch_size,self.num_hid).to(x.device), 
                        torch.zeros(batch_size,self.num_hid).to(x.device))
        else:
            h_t, c_t = init_states
         
        NH = self.num_hid
        for t in range(seq_size):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.hid_bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :NH]),     # input gate
                torch.sigmoid(gates[:, NH:NH*2]), # forget gate
                torch.tanh(gates[:, NH*2:NH*3]),  # new values
                torch.sigmoid(gates[:, NH*3:]),   # output gate
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
            # CHANGED
            # self.context.append(c_t.unsqueeze(0))
            # self.hid.append(h_t.unsqueeze(0))
            # self.i_t.append(i_t.unsqueeze(0))
            # self.f_t.append(f_t.unsqueeze(0))
            # self.g_t.append(g_t.unsqueeze(0))
            # self.o_t.append(o_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0,1).contiguous()
        output = hidden_seq @ self.V + self.out_bias
        self.context = c_t
        self.hid = h_t
        self.i_t = i_t
        self.f_t = f_t
        self.g_t = g_t
        self.o_t = o_t
        # self.context = torch.cat(self.context, dim=0)
        # self.context = self.context.transpose(0,1).contiguous()
        # reshape from (sequence, batch, feature)
        #           to (batch, sequence, feature)
        # self.hid = self.hid.transpose(0,1).contiguous()
        # self.hid = torch.cat(self.hid, dim=0)
        
        # self.i_t = self.i_t.transpose(0,1).contiguous()
        # self.i_t = torch.cat(self.i_t, dim=0)
        # self.f_t = self.f_t.transpose(0,1).contiguous()
        # self.f_t = torch.cat(self.f_t, dim=0)
        # self.g_t = self.g_t.transpose(0,1).contiguous()
        # self.g_t = torch.cat(self.g_t, dim=0)
        # self.o_t = self.o_t.transpose(0,1).contiguous()
        # self.o_t = torch.cat(self.o_t, dim=0)
        return hidden_seq, output
