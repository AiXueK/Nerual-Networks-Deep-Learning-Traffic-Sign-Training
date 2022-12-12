"""
   crown.py
   COMP9444, CSE, UNSW
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Full3Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full3Net, self).__init__()
        self.hid1 = self.hid2 = None
        self.hidLayer1 = nn.Linear(2, hid)
        self.hidLayer2 = nn.Linear(hid, hid)
        self.outputLayer = nn.Linear(hid, 1)

    def forward(self, input):
        self.hid1 = torch.tanh(self.hidLayer1(input))
        self.hid2 = torch.tanh(self.hidLayer2(self.hid1))
        return torch.sigmoid(self.outputLayer(self.hid2))

class Full4Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full4Net, self).__init__()
        self.hid1 = self.hid2 = self.hid3 = None
        self.hidLayer1 = nn.Linear(2, hid)
        self.hidLayer2 = nn.Linear(hid, hid)
        self.hidLayer3 = nn.Linear(hid, hid)
        self.outputLayer = nn.Linear(hid, 1)

    def forward(self, input):
        self.hid1 = torch.tanh(self.hidLayer1(input))
        self.hid2 = torch.tanh(self.hidLayer2(self.hid1))
        self.hid3 = torch.tanh(self.hidLayer3(self.hid2))
        return torch.sigmoid(self.outputLayer(self.hid3))

class DenseNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(DenseNet, self).__init__()
        self.hid1 = self.hid2 = None
        self.hidLayer1 = nn.Linear(2, num_hid)
        self.hidLayer21 = nn.Linear(2, num_hid)
        self.hidLayer22 = nn.Linear(num_hid, num_hid, bias=False)
        self.outputLayer1 = nn.Linear(2, 1)
        self.outputLayer2 = nn.Linear(num_hid, 1, bias=False)
        self.outputLayer3 = nn.Linear(num_hid, 1, bias=False)

    def forward(self, input):
        self.hid1 = torch.tanh(self.hidLayer1(input))
        self.hid2 = torch.tanh(self.hidLayer21(input) + self.hidLayer22(self.hid1))
        return torch.sigmoid(self.outputLayer1(input) + self.outputLayer2(self.hid1)
                             + self.outputLayer3(self.hid2))
