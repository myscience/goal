import torch
import torch.jit as jit
from torch.nn import Parameter

from torch import Tensor
from typing import Optional, Tuple, Callable
from collections import namedtuple

from math import exp

LIFState = namedtuple('LIFState', ['v', 'z', 'hz', 'ej'])

class LIF(jit.ScriptModule):
    def __init__(self,
        n_inp : int,            # Number of input neurons
        n_rec : int,            # Number of recurrent neurons

        tau_m : float = 1,      # Membrane time constant
        tau_s : float = 1,      # Spike time constant
        dt    : float = 1,      # Time integration step 
        thr   : float = 0,      # Spike generation threshold
        reset : float = -1,     # Reset voltage potential
        bias  : float = 0,      # Bias external current
        scale : float = 1.,     # Scaling factor for surrogate gradients

        sigma_inp : float = 1., # Variance of input matrix
        sigma_rec : float = 1., # Variance of recurrent matrix 

        **kwargs
        ) -> None:
        super(LIF, self).__init__()

        self.inp_size = n_inp
        self.rec_size = n_rec
        self.v_thr    = thr
        self.v_bias   = bias
        self.v_reset  = reset
        self.scale    = scale

        self.itau_m = exp(-dt / tau_m)
        self.itau_s = exp(-dt / tau_s)

        self.W_ih = Parameter(torch.randn(n_inp, n_rec) * sigma_inp, requires_grad = False)
        self.W_hh = Parameter(torch.randn(n_rec, n_rec) * sigma_rec, requires_grad = False)

        self.saved_tensors = torch.empty(0)

    @jit.script_method
    def forward(self, 
        input : Tensor,
        state : LIFState,
        drive : Optional[Tensor] = None
        ) -> Tuple[Tensor, LIFState]:

        # Save input tensor for backward pass
        self.saved_tensors = input.clone()

        # Unpack the state tuple into membrane potential and spike
        v, z, hz, ej = state

        # Compute online the elegibility trace
        nhz = hz * self.itau_s + (1 - self.itau_s) * z
        nej = ej * self.itau_m + (1 - self.itau_m) * nhz

        # Compute the input current
        I_t = torch.mm(input, self.W_ih) + torch.mm(nhz, self.W_hh) + self.v_bias
        I_reset = z * self.v_reset
        I_drive = torch.zeros_like(I_t) if drive is None else drive

        # Update the membrane potential
        nv = v * self.itau_m + (1 - self.itau_m) * (I_t + I_drive) + I_reset

        # Generate the spike signal
        nz = torch.gt(nv, self.v_thr).to(z.dtype)
        
        # Compose the new output
        return nz, LIFState(nv, nz, nhz, nej)

    # This function is highly inspired from:
    # https://github.com/fzenke/spytorch/blob/main/notebooks/SpyTorchTutorial4.ipynb
    @staticmethod
    @jit.script_method
    def backward(self, grad_out):
        input, = self.saved_tensors
        grad_inp = grad_out.clone()

        grad = grad_inp / (self.scale * torch.abs(input) + 1.)**2
        
        return grad