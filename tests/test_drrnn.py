#!/usr/bin/python3
"""
file: test_drrnn.py

test of the Deep Ritz Residual Neural Network

Questions:
    - Can we recreate the results in the Original Deep Ritz Paper?
"""
import torch
from context import models


def test_instance():
    in_N = 2
    m = 10
    out_N = 1
    net = models.drrnn(in_N,m,out_N)
    print("(+) Successfully made an instance")

def test_forward():
    in_N = 2
    m = 10
    out_N = 1
    net = models.drrnn(in_N,m,out_N)
    
    # get 1000 sample data
    x = torch.rand(1000,in_N)
    y = net(x)
    
    print("(+) Forward Pass worked")

if __name__ == "__main__":
    print("TESTING class drrnn:")
    test_instance()
    test_forward()
