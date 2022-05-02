"""
    © 2022 This work is licensed under a CC-BY-NC-SA license.
    Title: *"Error-based or target-based? A unifying framework for learning in recurrent spiking networks"*
    Authors: Cristiano Capone¹*, Paolo Muratore²* & Pier Stanislao Paolucci¹

    *: These authors equally contributed to the article

    ¹INFN, Sezione di Roma, Italy RM 00185
    ²SISSA - Internation School for Advanced Studies, Trieste, Italy
"""

import torch 
import torch.nn as nn
import torch.jit as jit

from torch.nn import Parameter
from torch import Tensor

from typing import Tuple, List, Optional, Union, Callable

from math import exp, sqrt

from lif import LIF, LIFState
from ongrad import OptSpike, OptOutput

class GOAL(jit.ScriptModule):
    '''
        Pytorch implementation of a Recurrent Spiking Network model, with
        rank & timescale error-feedbacks control for adjustable learning
        regimes (from purely error-based to purely target-based).

        Online learning is implemented via a forward hook.
    '''
    def __init__(self,
        n_inp : int,    # Number of input neurons
        n_rec : int,    # Number of recurrent neurons
        n_out : int,    # Number of output neurons

        Vo : float = 0.,         # Initial voltage condition 

        dt    : float = 1,       # Time integration step 
        tau_o : float = 1,       # Readout time constant

        rec_lr : float = 1,      # Recurrent optimizer learning rate
        out_lr : float = 1,      # Readout   optimizer learning rate

        rank  : int = None,      # The rank of error feedback matrix
        targ  : Tensor = None,   # The output target to aim for

        feedback : str = 'spk',  # The kind of feedback to use (Either Eq (8) or Eq (10) of main paper)
        resample : bool = True,  # Flag to signal whether to resample the feedback matrix at each call

        device : Optional[Union[str, torch.device]] = None,  # Hardware device where to run computations

        **kwargs        # Additional kwargs argument to pass to cell constructor
        ):
        super().__init__()

        self.inp_size = n_inp
        self.rec_size = n_rec
        self.out_size = n_out

        # Store the output target
        self.out_targ = targ

        self.itau_o = exp(-dt / tau_o)

        self.Vo = Vo

        # Select the device where to run all the computations
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        device = torch.device(device)

        # Instantiate the recurrent module
        self.cell = LIF(n_inp, n_rec, dt = dt, **kwargs).to(device)

        # Initialize the readout matrix
        self.W_ho = Parameter(torch.randn(n_rec, n_out, device = device) / sqrt(n_rec), requires_grad = False)

        # Initialize the optimizers
        self.rec_lr = rec_lr
        self.out_lr = out_lr

        # NOTE: Online Gradient inner matrices should be constructed only **after** 
        #       the readout matrix self.W_ho has been properly trained.
        gradpar = {'itau' : self.itau_o, 'lr' : rec_lr, 'rank' : rank, 
                   'resample' : resample, 'device' : device, 'out_targ' : targ}

        # Select the type of feedback computation
        if feedback == 'spk':
            self.rec_opt = OptSpike(self.cell.W_hh, self.W_ho, **gradpar)
        elif feedback == 'out':
            self.rec_opt = OptOutput(self.cell.W_hh, self.W_ho, **gradpar)
        else:
            raise ValueError(f'Unknown feedback: {feedback}. Can be either "spk" or "out".')

        self.device = device

    # @jit.script_method
    def forward(self,
        input  : List[Tensor],
        state  : LIFState = None,
        drive  : Tensor = None,
        target : Tensor = None
        ) -> Tuple[Tensor, Tensor]:

        if state is None: state = self.zero_state()

        outputs = jit.annotate(List[Tensor], [state.z])

        # If online training, we register the update function as
        # a PyTorch forward hook for the cell module
        if self.training:
            self.rec_opt.set_target(target)
            hook = self.cell.register_forward_hook(self.rec_opt)

        # Unpack the input along the time dimension
        for t, inp in enumerate(input):
            out, state = self.cell(inp, state, drive = None if drive is None else drive[t])

            outputs += [out]

        if self.training: 
            self.rec_opt.clear()
            hook.remove()

        outputs = torch.stack(outputs)

        return outputs, self.read(outputs, self.itau_o)

    @jit.script_method
    def read(self, z : Tensor, itau : float) -> Tensor:
        hz = self.filter(z, itau)

        return torch.einsum('tbh,ho->tbo', hz, self.W_ho).detach()

    @jit.script_method
    def filter(self, outs : Tensor, itau : Optional[float] = None) -> Tensor:
        if itau is None: itau = self.itau_o

        rout = torch.zeros_like(outs)
        
        # Filter the output using readout time constant
        for t, out in enumerate(outs):
            rout[t] = rout[t - 1] * itau + (1 - itau) * out

        return rout.detach()

    @jit.script_method
    def zero_state(self, batch : int = 1):
        shape = (batch, self.rec_size)

        v0  = torch.ones (shape, device = self.device) * self.Vo

        z0  = torch.zeros(shape, device = self.device)
        hz0 = torch.zeros(shape, device = self.device)
        ej0 = torch.zeros(shape, device = self.device)

        v0.requires_grad_(False) 
        z0.requires_grad_(False) 
        hz0.requires_grad_(False) 
        ej0.requires_grad_(False) 

        return LIFState(v0, z0, hz0, ej0)