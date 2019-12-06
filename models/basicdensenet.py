#!/usr/bin/python3
"""
file: basicdensenet.py

Creates a PyTorch Module of a baisc dense network 
for use in solving PDE via the Deep Ritz Method
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

class PowerReLU(nn.Module):
    """
    Implements simga(x)^(power)
    Applies a power of the rectified linear unit element-wise.

    NOTE: inplace may not be working.
    Can set inplace for inplace operation if desired.
    BUT I don't think it is working now.

    INPUT:
        x -- size (N,*) tensor where * is any number of additional
             dimensions
    OUTPUT:
        y -- size (N,*)
    """
    def __init__(self,inplace=False,power=3):
        super(PowerReLU,self).__init__()
        self.inplace = inplace
        self.power = power

    def forward(self,input):
        y = F.relu(input,inplace=self.inplace)
        return y.pow(self.power)

class Block(nn.Module):
    """
    IMplementation of the block used in the Deep Ritz 
    Paper
    
    Parameters:
    in_N  -- dimension of the input
    width -- number of nodes in the interior middle layer
    out_N -- dimension of the output
    phi   -- activation function used
    """
    def __init__(self,in_N,width,out_N,phi=PowerReLU()):
        super(Block,self).__init__()
        # create the necessary linear layers
        self.L1= nn.Linear(in_N,width)
        self.L2 = nn.Linear(width,out_N)
        # choose appropriate activation function
        self.phi = phi

    def forward(self,x):
        return self.phi(self.L2(self.phi(self.L1(x)))) + x

class drrnn(nn.Module):
    """
    drrnn -- Deep Ritz Residual Neural Network

    Implements a network with the architecture used in the
    deep ritz method paper
    
    Parameters:
        in_N  -- input dimension
        out_N -- output dimension
        m     -- width of layers that form blocks
        depth -- number of blocks to be stacked
        phi   -- the activation function
    """
    def __init__(self,in_N,m,out_N,depth=4,phi=PowerReLU()):
        super(drrnn,self).__init__()
        # set parameters
        self.in_N = in_N
        self.m = m
        self.out_N = out_N
        self.depth = depth
        self.phi = phi

        # list for holding all the blocks
        self.stack = nn.ModuleList()

        # add first layer to list
        self.stack.append(nn.Linear(in_N,m))

        # add middle blocks to list 
        for i in range(depth):
            self.stack.append(Block(m,m,m))
        
        # add output linear layer
        self.stack.append(nn.Linear(m,out_N))

    def forward(self,x):
        # first layer
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x



if __name__ == "__main__":
    # test PowerReLU
    x = torch.linspace(-1,1,1000)
    Fun = PowerReLU()
    y = Fun(x)
    # optionally print the result
    #plt.plot(x.numpy(),y.numpy())
    #plt.show()

    # test drrnn
    in_N = 2
    m = 10
    out_N = 1
    net = drrnn(in_N,m,out_N)

    # make 1000 test points
    x = torch.rand(1000,in_N)
    
    # test forward pass of drrnn
    y = net(x)

