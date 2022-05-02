"""
    © 2022 This work is licensed under a CC-BY-NC-SA license.
    Title: *"Error-based or target-based? A unifying framework for learning in recurrent spiking networks"*
    Authors: Cristiano Capone¹, Paolo Muratore² & Pier Stanislao Paolucci¹

    ¹INFN, Sezione di Roma, Italy RM 00185
    ²SISSA - Internation School for Advanced Studies, Trieste, Italy
"""

import torch
import numpy as np

from math import pi
from goal import GOAL

from tqdm import trange
from typing import Tuple

# Bbuild a k-D trajectory
def build_target(n_out, T = 150, norm = True, off = 0):
    t = torch.linspace(0, 2 * pi, T)
    Fs = [1, 2, 3, 5]

    tout = []
    for _ in range(n_out):
        As = torch.zeros(4).uniform_(0, 1)
        Ps = torch.zeros(4).uniform_(0, 2 * pi)

        tout += [torch.stack([A * torch.sin(f * t + p) for A, f, p in zip(As, Fs, Ps)]).sum(0)]

    tout = torch.stack(tout, dim = -1).reshape(T, 1, n_out)
    tout[:off].mul_(0.)

    return tout / tout.max() if norm else tout 

def build_clock(n_clk, T = 150, off = 0):
    clk = torch.zeros(T, n_clk)

    span = T // n_clk

    for k in range(n_clk):
        clk[k * span : (k+1) * span, k] = 1

    clk[:off].mul_(0.)

    return clk.reshape(T, 1, n_clk)

def train(par : dict):
    n_inp = par['n_inp']
    n_rec = par['n_rec']
    n_out = par['n_out']

    device = par['device']

    targ = par['targ']
    clck = par['clck']

    out_lr = par['out_lr']
    batch_size = par['batch_size']

    out_epochs, rec_epochs = par['epochs']

    net = GOAL(**par)

    # Set the inputs
    net.eval()
    tdrive = targ @ (torch.randn(n_out, n_rec, device = device) * par['sigma_trg'])

    # * Train the readout
    z_targ, _ = net(clck, drive = tdrive)
    hz_targ = net.filter(z_targ)

    out_iter = trange(out_epochs, desc = 'Readout Training | MSE: --- ', leave = False) if par['verbose'] else\
                range(out_epochs)

    rec_iter = trange(rec_epochs, desc = 'Recurrent Training | MSE: --- | ΔZ: --- ', leave = False) if par['verbose'] else\
                range(rec_epochs)

    for _ in out_iter:
        rout = hz_targ @ net.W_ho
        with torch.no_grad():
            net.W_ho += out_lr * torch.einsum('tbo,tbh->ho', (targ - rout), hz_targ) / batch_size
            
        mse = torch.mean((targ - rout)**2).detach()

        if par['verbose']:
            msg = f'Readout Training | MSE: {mse:.3f}'
            out_iter.set_description(msg)

    # Store the output MSE
    MSE_out = mse

    # Now that readout training is completed we build the OnlineGrad
    net.rec_opt.build(hz_targ)
    
    # * Train the recurrent network
    net.train()
    inputs = clck.unbind(0)

    MSE_rec = np.zeros(rec_epochs)
    ΔZ = np.zeros(rec_epochs)

    for epoch in rec_iter:
        Z, rout = net(inputs, target = hz_targ)

        mse = torch.mean((targ - rout.detach())**2).detach()
        δz  = torch.abs(z_targ - Z.detach()).sum()

        MSE_rec[epoch] = mse
        ΔZ[epoch] = δz

        if par['verbose']:
            msg = f'Recurrent Training | MSE: {mse:.3f} | ΔZ: {δz}'
            rec_iter.set_description(msg)


    return MSE_out, MSE_rec, ΔZ

def scan(par : dict, pair : Tuple[int, float]):
    par['rank']  = pair[0]
    par['tau_o'] = pair[1]
    
    return train(par)


dt = 0.001
default = {'dt'    : dt,

       'tau_m' : 8 * dt,
       'tau_s' : 2 * dt,
       'tau_o' : 5 * dt,
       
       'sigma_inp' : 22,
       'sigma_trg' : 8,

       'sigma_rec' : 0,

       'reset' : -20,
       'bias'  : -4,
       'Vo'    : -4,

       'n_inp' : 5,
       'n_rec' : 150,
       'n_out' : 3, 

        't_seq' : 100,

       'resample' : False,
       'batch_size' : 1,

       'rec_lr' : 0.05,
       'out_lr' : 0.0025
       }