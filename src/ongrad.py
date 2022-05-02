"""
    © 2022 This work is licensed under a CC-BY-NC-SA license.
    Title: *"Error-based or target-based? A unifying framework for learning in recurrent spiking networks"*
    Authors: Cristiano Capone¹*, Paolo Muratore²* & Pier Stanislao Paolucci¹

    *: These authors equally contributed to the article

    ¹INFN, Sezione di Roma, Italy RM 00185
    ²SISSA - Internation School for Advanced Studies, Trieste, Italy
"""

import torch
import torch.jit as jit

from abc import ABC, abstractmethod
from torch import Tensor
from torch.optim import SGD
from typing import Optional, Union, Tuple, List

from lif import LIF, LIFState

class OptimOn(ABC):
    '''
        A callable Python object that implements the online gradient computation & update.
    '''

    def __init__(self,
        W_hh : Tensor,
        W_ho : Tensor,
        itau : float = 1.,
        lr   : float = 0.01,

        out_targ : Optional[Tensor] = None,  # Output target for the optimizer.
        spk_targ : Optional[Tensor] = None,  # Spikes target for the optimizer.
        device : Optional[Union[str, torch.device]] = None,  # Hardware device where to run computations

        rank : int = None,  # The rank of the feedback matrix D

        resample : bool = True  # Flag to signal whether to resample the feedback matrix at each call
        ) -> None:
        super(OptimOn, self).__init__()

        # Get the shape of network
        n_rec, n_out = W_ho.shape

        self.rank = rank
        self.n_out = n_out

        # Select a working device
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        device = torch.device(device)

        self.t  = 0
        self.hz = torch.zeros(n_rec, device = device, requires_grad = False)

        self.W_hh = W_hh
        self.W_ho = W_ho

        self.itau = itau

        self.lr   = lr

        # Create the optimizer
        self.optim = SGD([W_hh], lr)

        self.out_targ = torch.empty(0) if out_targ is None else out_targ
        self.spk_targ = torch.empty(0) if spk_targ is None else spk_targ

        self.has_targ = False if spk_targ is None else True

        self.n_rec = n_rec
        self.device = device

        self.resample = resample and rank is not None

        # NOTE: Online Gradient inner matrices should be constructed only **after** 
        #       the readout matrix W_ho has been properly trained. Here we init
        #       the built flag to False to ensure that it optimizer can only be called
        #       after readout training has taken place
        self.built = False

    def set_target(self, targ : Tensor):
        self.spk_targ = targ
        self.has_targ = True
        
        self.hz      *= 0 
        self.t        = 0

    def clear(self):
        self.hz      *= 0 
        self.t        = 0
        self.spk_targ = torch.empty_like(self.spk_targ)
        self.has_targ = False

    @abstractmethod
    def __call__(self,
        module : LIF, 
        minp : Tuple[Tensor, LIFState], 
        mout : Tuple[Tensor, LIFState]
        ):
        return

    @abstractmethod
    def build(self, targ : Tensor = None):
        return
        

class OptSpike(OptimOn):
    '''
        A callable Python object that implements the online gradient computation & update.
    '''
    def __init__(self, *args, **kwargs) -> None:
        super(OptSpike, self).__init__(*args, **kwargs)

    def __call__(self,
        module : LIF, 
        minp : Tuple[Tensor, LIFState], 
        mout : Tuple[Tensor, LIFState]
        ):

        if not self.has_targ:
            raise AttributeError('Cannot perform online update without a spike target.')

        if not self.built:
            msg = '''Online Gradient has not been properly built. You should call the
                     OnlineGrad.built() method to construct the feedback matrix. This
                     is done as proper feedback need a fully trained readout matrix so
                     building needs to wait for such training to occur.
                  '''
            raise ValueError(msg)

        # If resample is demaned we re-build the feedback matrix at each iteration
        if self.resample: self.build()

        _, state = mout
        v, z, _, ej = state

        t = self.t

        # Filter the spikes with readout constant
        self.hz = self.hz * self.itau + (1 - self.itau) * z

        with torch.no_grad():
            # Compute the (internal) error
            err = self.spk_targ[t+1] - self.hz

            # Compute the pseudo-derivative of the spike
            pseudo = torch.sigmoid(v) * (1 - torch.sigmoid(v))

            # Compute the gradient
            module.W_hh.grad = -torch.einsum('bi,bj->ji', (err @ self.D) * pseudo, ej)
            
            # grad = torch.einsum('bi,bj->ji', (err @ self.D) * pseudo, ej)
            # module.W_hh.add_(grad, alpha = self.lr)
            
            # Update the weights via optimizer
            self.optim.step()

            # Enforce no self-connections
            module.W_hh.fill_diagonal_ (0)

        # Update time
        self.t += 1

    def build(self, targ : Tensor = None):
        # Compute augmented matrix B based on trained readout matrix and a random Gaussian
        # component so to reach the desired rank. 
        std = 2 * self.W_ho.std()
        B = torch.randn(self.n_rec, self.n_rec, device = self.device, requires_grad = False) * std
        B[:, :self.n_out] = self.W_ho.detach().clone()

        # Construct the feedback matrix based on matrix B and provided rank. If rank is None,
        # the system falls back to LTTS training (purely diagonal feedback)
        self.D = torch.mm(B[:, :self.rank], B[:, :self.rank].T) if self.rank is not None else\
                 torch.eye(self.n_rec, device = self.device) * torch.mean(torch.diag(B @ B.T))

        self.built = True

class OptOutput(OptimOn):
    '''
        This class instantiate the Online Optimizer by implementing Eq. (8) of the
        paper. In particular, errors are computed at the output-level and projected
        into the network via the (augmented) feedback matrix B+.
    '''
    def __init__(self, *args, **kwargs) -> None:
        super(OptOutput, self).__init__(*args, **kwargs)

    def __call__(self,
        module : LIF, 
        minp : Tuple[Tensor, LIFState], 
        mout : Tuple[Tensor, LIFState]
        ):

        if not self.has_targ:
            raise AttributeError('Cannot perform online update without both targets.')

        if not self.built:
            msg = '''Online Gradient has not been properly built. You should call the
                     OnlineGrad.built() method to construct the feedback matrix. This
                     is done as proper feedback need a fully trained readout matrix so
                     building needs to wait for such training to occur.
                  '''
            raise ValueError(msg)

        # If resample is demaned we re-build the feedback matrix at each iteration
        if self.resample: self.build()

        _, state = mout
        v, z, _, ej = state

        t = self.t

        # Filter the spikes with readout constant
        self.hz = self.hz * self.itau + (1 - self.itau) * z

        with torch.no_grad():
            # Compute the (internal) error
            err = self.Ytarg[t+1] - self.hz @ self.B

            # Compute the pseudo-derivative of the spike
            pseudo = torch.sigmoid(v) * (1 - torch.sigmoid(v))

            # Compute the gradient
            module.W_hh.grad = -torch.einsum('bi,bj->ji', (err @ self.B.T) * pseudo, ej)
            
            # Update the weights via optimizer
            self.optim.step()

            # grad = torch.einsum('bi,bj->ji', (err @ self.B.T) * pseudo, ej)
            # module.W_hh.add_(grad, alpha = self.lr)

            # Enforce no self-connections
            module.W_hh.fill_diagonal_ (0)

        # Update time
        self.t += 1

    def build(self, spk_targ : Tensor = None):
        if spk_targ is None and not self.has_targ:
            raise ValueError('Cannot build the feedback matrix without output and spike target')

        starg = self.spk_targ if spk_targ is None else spk_targ

        # Compute augmented matrix B+ based on trained readout matrix and a random Gaussian
        # component so to reach the desired rank. 
        # NOTE: We guaranteed that rank is not None as we check it before class instantiation 
        std = 2 * self.W_ho.std()
        self.B = torch.randn(self.n_rec, self.rank, device = self.device, requires_grad = False) * std
        self.B[:, :self.n_out] = self.W_ho.detach().clone()

        # Compute the Y* of equation (8)
        self.Ytarg = torch.einsum('tbi,ik->tbk', starg, self.B)
        self.Ytarg[..., :self.n_out] = self.out_targ.clone()

        self.built = True