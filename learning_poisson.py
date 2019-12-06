#!/usr/bin/python3
r"""
file = learning_poisson.py

TO DO:
    - Sample from boundary
    - custom loss function?

Script for learning the solution to a poisson equation
    $-\Nabla u(x) = 1, x \in \Omega$
    $        u(x) = 0, x \in \partial\Omega$

"""
import math
import numpy as np
import torch
import torch.optim as optim
# import neural network models
import models

# set path to saved parameters of trained network
PATH_TO_PARAMS = "./data/poisson_drrnn_params.pt"

def train():
    # instantiate a network
    in_N = 2
    m = 10
    out_N = 1
    net = models.drrnn(in_N,m,out_N)

    # set up optimizer and loss
    # -- using Mean-square error for now
    criterion = torch.nn.MSELoss()
    # tell optimizer which parameters to update
    # -- as well as set up learning rate
    optimizer = optim.Adam(net.parameters(),lr=1e-2)

    # define number of epochs to take
    eN = 200
    # empty list to store train losses
    train_losses = []
    for epoch in range(eN):
        # get data for this iteration
        # -- x is the inputs
        # -- target will be the truth
        x = get_interior_points()
        # find truth for these poitns
        target = poisson_solution_2d(x)
        prediction = net(x)
        # find loss
        loss = criterion(prediction,target)
        # append to losses
        train_losses.append(loss.item())
        # print the progress
        print_progress(epoch,loss.item())
        # zero gradients
        optimizer.zero_grad()
        # backprop
        loss.backward()
        # update parameters
        optimizer.step()

    print("Completed Training")
    print("Saving Model")
    torch.save(net.state_dict(),PATH_TO_PARAMS)

def print_progress(epoch,loss):
    if epoch % 25 == 0:
        print("epoch: %4d -- loss: %e"%(epoch,loss))

def get_interior_points(N=100,d=2):
    """
    randomly sample N points from interior of [-1,1]^d
    """
    return torch.rand(N,d)

def get_boundary_points():
    """
    HOW TO RANDOMLY SAMPLE BOUNDARY
    """
    pass

def poisson_solution_2d(x):
    """
    returns u(r,theta) = r^(1/2)*sin(theta/2)
    where (r,theta) is the polar coordinates of 
    x = (x_1,x_2)
    """
    r = np.sqrt(x[:,0:1]**2 + x[:,1:2]**2)
    theta = np.arctan2(x[:,0:1],x[:,1:2])

    return r**(1/2)*np.sin(theta/2)

if __name__ == "__main__":
    train()
