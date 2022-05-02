"""
    © 2022 This work is licensed under a CC-BY-NC-SA license.
    Title: *"Error-based or target-based? A unifying framework for learning in recurrent spiking networks"*
    Authors: Cristiano Capone¹*, Paolo Muratore²* & Pier Stanislao Paolucci¹

    *: These authors equally contributed to the article

    ¹INFN, Sezione di Roma, Italy RM 00185
    ²SISSA - Internation School for Advanced Studies, Trieste, Italy
"""


import numpy as np
import utils as ut
from tqdm import trange
from optimizer import Adam, SimpleGradient

class GOAL:
    '''
        Numpy implementation of a Recurrent Spiking Network model, with
        rank & timescale error-feedbacks control for adjustable learning
        regimes (from purely error-based to purely target-based).

        Online learning is implemented via a forward hook.
    '''

    def __init__ (self, par):
        # This are the network size N, input I, output O and max temporal span T
        self.N, self.I, self.O, self.T = par['shape']

        self.dt = par['dt']

        self.itau_m = np.exp (-self.dt / par['tau_m']) 
        self.itau_s = np.exp (-self.dt / par['tau_s']) 
        self.itau_ro = np.exp (-self.dt / par['tau_ro'])
        self.itau_star = np.exp (-self.dt / par['tau_star']) 

        self.dv = par['dv'] 

        # This is the network connectivity matrix
        self.J = np.random.normal (0., par['sigma_Jrec'], size = (self.N, self.N))
        # self.J = np.zeros ((self.N, self.N)) 

        # This is the network input, teach and output matrices
        self.Jin = np.random.normal (0., par['sigma_input'], size = (self.N, self.I)) 
        self.Jteach = np.random.normal (0., par['sigma_teach'], size = (self.N, self.O)) 

        # This is the hint signal
        try:
            self.H = par['hint_shape']
            self.Jhint = np.random.normal (0., par['sigma_hint'], size = (self.N, self.H)) 
        except KeyError:
            self.H = 0
            self.Jhint = None

        self.Jout = np.zeros ((self.O, self.N)) 

        # Remove self-connections
        np.fill_diagonal (self.J, 0.) 

        # Impose reset after spike
        self.s_inh = -par['s_inh'] 
        self.Jreset = np.diag (np.ones (self.N) * self.s_inh) 

        # This is the external field
        h = par['h'] 

        assert type (h) in (np.ndarray, float, int)
        self.h = h if isinstance (h, np.ndarray) else np.ones (self.N) * h 

        # Membrane potential
        self.H = np.ones (self.N) * par['Vo'] 
        self.Vo = par['Vo'] 

        # These are the spikes train and the filtered spikes train
        self.S = np.zeros (self.N) 
        self.S_hat = np.zeros (self.N) 

        # This is the single-time output buffer
        self.out = np.zeros (self.N) 

        # Here we save the params dictionary
        self.par = par 

    def _sigm (self, x, dv = None):
        if dv is None:
            dv = self.dv 

        # If dv is too small, no need for elaborate computation, just return
        # the theta function
        if dv < 1 / 30:
            return x > 0 

        # Here we apply numerically stable version of signoid activation
        # based on the sign of the potential
        y = x / dv 

        out = np.zeros (x.shape) 
        mask = x > 0 
        out [mask] = 1. / (1. + np.exp (-y [mask])) 
        out [~mask] = np.exp (y [~mask]) / (1. + np.exp (y [~mask])) 

        return out 

    def _dsigm (self, x, dv = None):
        if dv is None: dv = self.dv
        # return self._sigm (x, dv = dv) * (1. - self._sigm (x, dv = dv))
        y = x / dv 

        out = np.zeros (x.shape) 
        mask = x > 0 
        out [mask] = 1. / (1. + np.exp (-y [mask])) 
        out [~mask] = np.exp (y [~mask]) / (1. + np.exp (y [~mask])) 

        return out * (1 - out)

    def step (self, inp, t):
        itau_m = self.itau_m 
        itau_s = self.itau_s 

        self.S_hat [:] = self.S_hat [:] * itau_s + self.S [:] * (1. - itau_s)

        self.H [:] = self.H [:] * itau_m + (1. - itau_m) * (self.J @ self.S_hat [:] + self.Jin @ inp + self.h)\
                                                          + self.Jreset @ self.S [:] 

        self.S [:] = self._sigm (self.H [:], dv = self.dv) - 0.5 > 0. 

        # Here we use our policy to suggest a novel action given the system
        # current state
        action = self.policy (self.S) 

        # Here we return the chosen next action
        return action, self.S.copy ()

    def policy (self, state):
        self.out = self.out * self.itau_ro  + state * (1 - self.itau_ro) 

        return self.Jout @ self.out 

    def compute (self, inp, init = None, rec = True):
        '''
            This function is used to compute the output of our model given an
            input.
            Args:
                inp : numpy.array of shape (N, T), where N is the number of neurons
                      in the network and T is the time length of the input.
                init: numpy.array of shape (N, ), where N is the number of
                      neurons in the network. It defines the initial condition on
                      the spikes. Should be in range [0, 1]. Continous values are
                      casted to boolean.
            Keywords:
                Tmax: (default: None) Optional time legth of the produced output.
                      If not provided default is self.T
                Vo  : (default: None) Optional initial condition for the neurons
                      membrane potential. If not provided defaults to external
                      field h for all neurons.
                dv  : (default: None) Optional different value for the dv param
                      to compute the sigmoid activation.
        '''
        # Check correct input shape
        assert inp.shape[0] == self.N 

        N, T = inp.shape 

        itau_m = self.itau_m 
        itau_s = self.itau_s 

        self.reset (init) 

        Sout = np.zeros ((N, T)) 

        for t in range (T - 1):
            self.S_hat [:] = self.S_hat [:] * itau_s + self.S [:] * (1. - itau_s)

            self.H [:] = self.H * itau_m + (1. - itau_m) * ((self.J @ self.S_hat [:] if rec else 0)
                                                            + inp [:, t] + self.h)\
                                                         + self.Jreset @ self.S 

            self.S [:] = self._sigm (self.H, dv = self.dv) - 0.5 > 0. 
            Sout [:, t + 1] = self.S.copy () 

        return Sout, self.Jout @ ut.sfilter (Sout, itau = self.itau_ro) 

    def implement (self, experts, hint = None, adapt = False, rec = False):
        if (self.J != 0).any () and rec:
            print ('WARNING: Implement expert with non-zero recurrent weights\n') 

        # First we parse experts input-output pair
        inps = np.array ([exp[0] for exp in experts]) 
        outs = np.array ([exp[1] for exp in experts]) 

        # Here we extract the maxima of input and output which are used to balance
        # the signal injection using the sigma_input and sigma_teach variables
        if adapt:
            self.sigma_input = 5. / np.max (np.abs (inps)) 
            self.sigma_teach = 5. / np.max (np.abs (outs)) 

            self.par['sigma_input'] = self.sigma_input 
            self.par['sigma_teach'] = self.sigma_teach 

            self.Jin = np.random.normal (0., self.sigma_input, size = (self.N, self.I)) 
            self.Jteach = np.random.normal (0., self.sigma_teach, size = (self.N, self.O)) 

        # We then build a target network dynamics
        Inp = np.einsum ('ij, njt->nit', self.Jin, inps) 
        Tar = np.einsum ('ij, njt->nit', self.Jteach, outs)

        tInp = Inp + Tar + (0 if hint is None else self.Jhint @ hint)

        Targ = [self.compute (t_inp, rec = rec)[0] for t_inp in tInp] 

        return Targ, Inp 

    def train(self, 
        input, 
        targets, 
        epochs = (100, 500), 
        rank = None, 
        Jext = None, 
        verbose = True,
        sample_at = None
        ):
        # Here we clone this behaviour
        itau_m = self.itau_m 
        itau_s = self.itau_s 
        itau_ro = self.itau_ro

        alpha = self.par['alpha'] 
        alpha_rout = self.par['alpha_rout'] 
        alpha_reg = 0.1

        # Separate the targets into: in & out targets
        int_targ, out_targ = targets

        dH = np.zeros (self.N)
        S_pred = np.zeros (self.N) 

        # int_err = np.zeros (epochs[1] if sample_at is None else np.sum(sample_at))
        # out_err = np.zeros (epochs[1] if sample_at is None else np.sum(sample_at))
        int_err = []
        out_err = []

        hin_targ = ut.sfilter(int_targ, self.itau_ro)

        # Readout Training from in-target
        optim = Adam (alpha = alpha_rout, drop = 1., drop_time = epochs[0]) 

        iterator = trange(epochs[0], desc = 'Readout Training | MSE: --- ') if verbose else\
                   range(epochs[0])

        for epoch in iterator:
            model_out = self.Jout @ hin_targ

            dJ = (out_targ - model_out) @ hin_targ.T


            self.Jout += alpha_rout * dJ
            # self.Jout = optim.step(self.Jout, dJ)


            mse = np.mean((out_targ[:, 8:] - model_out[:, 8:])**2)

            if verbose:
                msg = f'Readout Training | MSE: {mse:.3f} '
                iterator.set_description(msg)

        lim_err = mse

        if Jext is None:
            # If no external matrix was provided, create our own Jaug matrix
            Jaug = np.random.normal(0., np.std(self.Jout) * 2, size = (rank, self.N))
        else:
            # Otherwise, pick the externally provided one and scale it
            Jaug = Jext.copy() * np.std(self.Jout)
            Jaug[:self.Jout.shape[0]] = self.Jout.copy()
            
        # Rank None selects for the pure LTTS case where the B matrix is purely diagonal
        # Otherwise, we set compute B accordingly to the provided rank
        B = np.eye(self.N) * np.mean(np.diag(Jaug.T @ Jaug)) if rank is None else\
            Jaug[:rank].T @ Jaug[:rank]

        # Recurrent training based on trained readout matrix
        iterator = trange(epochs[1], desc = 'Rec. Training | ΔS: --- | MSE: --- ') if verbose else\
                   range(epochs[1])
        for epoch in iterator:
            self.reset () 
            dH *= 0

            S_pred *= 0

            int_pred = [S_pred]
            DH = [dH]

            for t in range (self.T - 1):
                self.S_hat = self.S_hat * itau_s + self.S * (1. - itau_s) 
                self.H = self.H * itau_m + (1. - itau_m) * (self.J @ self.S_hat + input [:, t] + self.h)\
                                                + self.Jreset @ self.S 

                dH = dH  * itau_m + (1. - itau_m) * self.S_hat 

                self.S = self.H > 0. 
                S_pred = S_pred * itau_ro + self.S * (1. - itau_ro)

                int_pred += [self.S]
                DH       += dH

                dJ = np.outer((B @ (hin_targ[:, t+1] - S_pred)) * self._dsigm(self.H), dH)

                self.J += alpha * dJ
                np.fill_diagonal(self.J, 0.)

            # Here we implement a firing-rate regularization
            fr_targ = np.mean(int_targ, axis = 1, keepdims = True)
            fr_pred = np.mean(int_pred, axis = 0, keepdims = True).T

            dJ = (fr_targ - fr_pred) @ DH
            self.J += alpha_reg * dJ
            np.fill_diagonal(self.J, 0.)


            if sample_at[epoch]:
                # Here we track the training measures: internal error and output error
                S, O = self.compute(input)

                dS  = np.abs(int_targ - S).sum()
                mse = np.mean((out_targ[:, 8:] - O[:, 8:])**2)

                int_err += [dS]
                out_err += [mse]

            if verbose:
                msg = f'Rec. Training | ΔS: {dS} | MSE: {mse:.3f} '
                iterator.set_description(msg)

        return int_err, out_err, lim_err


    def reset (self, init = None):
        self.S [:] = init if init else np.zeros (self.N) 
        self.S_hat [:] = self.S [:] * self.itau_s if init else np.zeros (self.N) 

        self.out [:] *= 0 

        self.H [:] = self.Vo 

    def forget (self, J = None, Jout = None):
        self.J = np.random.normal (0., self.par['sigma_Jrec'], size = (self.N, self.N)) if J is None else J.copy()
        self.Jout = np.zeros ((self.O, self.N)) if Jout is None else Jout.copy()

    def save (self, filename):
        # Here we collect the relevant quantities to store
        data_bundle = (self.Jin, self.Jteach, self.Jout, self.J, self.par)

        np.save (filename, np.array (data_bundle, dtype = np.object)) 

    @classmethod
    def load (cls, filename):
        data_bundle = np.load (filename, allow_pickle = True) 

        Jin, Jteach, Jout, J, par = data_bundle 

        obj = GOAL (par) 

        obj.Jin = Jin.copy () 
        obj.Jteach = Jteach.copy () 
        obj.Jout = Jout.copy () 
        obj.J = J.copy () 

        return obj 
