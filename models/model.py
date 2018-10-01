#!/usr/bin/env python
# coding=UTF-8
from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init as init
import torchvision
import numpy as np
from torch.autograd import Variable

__all__ = ['LstmModel']


class LstmModel(nn.Module):
    def __init__(self, use_gpu=True):
        super(LstmModel, self).__init__()

        self.Tensor = torch.cuda.FloatTensor if use_gpu else torch.Tensor
        self.gru_layer1 = nn.GRU(input_size=7, hidden_size=500,
                                 num_layers=1, batch_first=True)
        self.gru_layer2 = nn.GRU(input_size=500, hidden_size=500,
                                 num_layers=1, batch_first=True)
        self.gru_layer3 = nn.GRU(input_size=500, hidden_size=1,
                                 num_layers=1, batch_first=True)
        self.tanh = nn.Tanh()
        # self.out_bn1 = nn.BatchNorm1d(100)
        # self.out_bn2 = nn.BatchNorm1d(300)
        self.use_gpu = use_gpu

        self.l2_criterion = nn.MSELoss()
        # L2 regularization

    def init_weights(self):
        # LSTM Unit: numlayer = 1, initialization
        init.orthogonal_(self.gru_layer1.all_weights[0][0], gain=np.sqrt(2.0))
        init.orthogonal_(self.gru_layer1.all_weights[0][1], gain=np.sqrt(2.0))
        init.uniform_(self.gru_layer1.all_weights[0][2], 1, 0.1)
        init.uniform_(self.gru_layer1.all_weights[0][3], 1, 0.1)

        init.orthogonal_(self.gru_layer2.all_weights[0][0], gain=np.sqrt(2.0))
        init.orthogonal_(self.gru_layer2.all_weights[0][1], gain=np.sqrt(2.0))
        init.uniform_(self.gru_layer2.all_weights[0][2], 1, 0.1)
        init.uniform_(self.gru_layer2.all_weights[0][3], 1, 0.1)

        init.orthogonal_(self.gru_layer3.all_weights[0][0], gain=np.sqrt(2.0))
        init.orthogonal_(self.gru_layer3.all_weights[0][1], gain=np.sqrt(2.0))
        init.uniform_(self.gru_layer3.all_weights[0][2], 1, 0.1)
        init.uniform_(self.gru_layer3.all_weights[0][3], 1, 0.1)

    def init_hidden(self, input_batch):
        hidden1 = torch.Tensor(
            torch.zeros(1, input_batch, 500))
        hidden2 = torch.Tensor(
            torch.zeros(1, input_batch, 500))
        hidden3 = torch.Tensor(
            torch.zeros(1, input_batch, 1))
        return (hidden1.cuda(), hidden2.cuda(), hidden3.cuda()) \
            if self.use_gpu else (hidden1, hidden2, hidden3)

    def layer_regularization(self):
        # lstm_layer L2 regularization
        target_tensor = self.Tensor(self.gru_layer1.all_weights[0][0].size()).fill_(0)
        target_tensor = Variable(
            target_tensor, requires_grad=False)
        l2_regular_loss = self.l2_criterion(self.gru_layer1.all_weights[0][0], target_tensor)
        target_tensor = self.Tensor(self.gru_layer1.all_weights[0][1].size()).fill_(0)
        target_tensor = Variable(
            target_tensor, requires_grad=False)
        l2_regular_loss += self.l2_criterion(self.gru_layer1.all_weights[0][1], target_tensor)

        target_tensor = self.Tensor(self.gru_layer2.all_weights[0][0].size()).fill_(0)
        target_tensor = Variable(
            target_tensor, requires_grad=False)
        l2_regular_loss += self.l2_criterion(self.gru_layer2.all_weights[0][0], target_tensor)
        target_tensor = self.Tensor(self.gru_layer2.all_weights[0][1].size()).fill_(0)
        target_tensor = Variable(
            target_tensor, requires_grad=False)
        l2_regular_loss += self.l2_criterion(self.gru_layer2.all_weights[0][1], target_tensor)
        return l2_regular_loss

    def forward(self, x, hidden01, hidden02, hidden03):
        # input_batch = x.shape[0]

        gru1_output, hidden1 = self.gru_layer1(x, hidden01)
        # gru1_output = self.out_bn1(gru1_output.squeeze(0)).unsqueeze(0)
        gru2_output, hidden2 = self.gru_layer2(gru1_output, hidden02)
        # gru2_output = self.out_bn2(gru2_output.squeeze(0)).unsqueeze(0)
        gru3_output, hidden3 = self.gru_layer3(gru2_output, hidden03)
        gru_theta = gru3_output.squeeze(2)  # size:[in_batch,seq_lenth]

        if not self.training:
            return gru_theta, hidden1, hidden2, hidden3

        l2_regular_loss = self.layer_regularization()

        return gru_theta, hidden1, hidden2, hidden3, l2_regular_loss
