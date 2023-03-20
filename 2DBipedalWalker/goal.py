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
from optimizer import Adam

def sigm ( x, dv ):

	if dv < 1 / 30:
		return x > 0;
	y = x / dv;
    #y = x / 10.;

	out = 1.5*(1. / (1. + np.exp (-y*3. )) - .5);

	return out;


class LTTS:

    def __init__ (self, par):
        # This are the network size N, input I, output O and max temporal span T
        self.N, self.I, self.O, self.T = par['shape'];
        net_shape = (self.N, self.T);

        self.dt = par['dt']#1. / self.T;
        self.itau_m = self.dt / par['tau_m'];
        self.itau_s = np.exp (-self.dt / par['tau_s']);
        self.itau_ro = np.exp (-self.dt / par['tau_ro']);

        self.dv = par['dv'];

        # This is the network connectivity matrix
        self.J = np.zeros ((self.N, self.N));

        # This is the network input, teach and output matrices
        self.Jin = np.random.normal (0., par['sigma_input'], size = (self.N, self.I));
        self.Jteach = np.random.normal (0., par['sigma_teach'], size = (self.N, self.O));

        self.Jout = np.random.normal (0., 0.1, size = (self.O,self.N));#np.zeros ((self.O, self.N));

        # Remove self-connections
        np.fill_diagonal (self.J, 0.);

        # Impose reset after spike
        self.s_inh = -par['s_inh'];
        self.Jreset = np.diag (np.ones (self.N) * self.s_inh);

        # This is the external field
        h = par['h'];

        assert type (h) in (np.ndarray, float, int)
        self.h = h if isinstance (h, np.ndarray) else np.ones (self.N) * h;

        # Membrane potential
        self.H = np.ones (self.N) * par['Vo'];

        self.Vo = par['Vo'];

        # These are the spikes train and the filtered spikes train
        self.S = np.zeros (self.N);
        self.S_hat = np.zeros (self.N);
        self.dH = np.zeros (self.N);

        # This is the single-time output buffer
        self.out = np.zeros (self.N);

        # Here we save the params dictionary
        self.par = par;

    def _sigm (self, x, dv = None):
        if dv is None:
            dv = self.dv;

        # If dv is too small, no need for elaborate computation, just return
        # the theta function
        if dv < 1 / 30:
            return x > 0;

        # Here we apply numerically stable version of signoid activation
        # based on the sign of the potential
        y = x / dv;

        out = np.zeros (x.shape);
        mask = x > 0;
        out [mask] = 1. / (1. + np.exp (-y [mask]));
        out [~mask] = np.exp (y [~mask]) / (1. + np.exp (y [~mask]));

        return out;

    def _dsigm (self, x, dv = None):
        return self._sigm (x, dv = dv) * (1. - self._sigm (x, dv = dv));

    def step (self, inp, t, probabilistic = False):
        itau_m = self.itau_m;
        itau_s = self.itau_s;

        self.S_hat [:] = (self.S_hat [:] * itau_s + self.S [:] * (1. - itau_s))
        self.dH [:] = self.dH  * (1. - itau_m) + itau_m * self.S_hat;

        self.H [:] = self.H [:] * (1. - itau_m) + itau_m * (self.J @ self.S_hat [:] + self.Jin @ inp + self.h)\
                                                          + self.Jreset @ self.S [:];

        dv_prob = .1
        if probabilistic:
            #print('here prob')
            self.S [:] = np.heaviside(np.random.uniform(low=0.0, high=1.0, size=np.shape(self.H [:])) - (1-self._sigm (self.H [:], dv = dv_prob )),0)
        else:
            self.S [:] = self._sigm (self.H [:], dv = self.dv) - 0.5 > 0.

        # Here we use our policy to suggest a novel action given the system
        # current state
        action = self.policy (self.S);

        dJ = np.outer ((self.S [:] - self._sigm (self.H [:], dv = dv_prob ) ) , self.dH);#
        #print( self._sigm (self.H [:], dv = dv_prob ) )
        #print(np.std(dJ))

        # Here we return the chosen next action
        return action, self.S.copy () , dJ , self.H [:]

    def policy (self, state):
        self.out = self.out * self.itau_ro  + state * (1 - self.itau_ro);

        return self.Jout @ self.out;

    def compute (self, inp, init = None, rec = True):
        assert inp.shape[0] == self.N;
        N, T = inp.shape;

        itau_m = self.itau_m;
        itau_s = self.itau_s;

        self.reset (init);

        Sout = np.zeros ((N, T));
        self.H [:] = self.H [:]*0+self.Vo
        self.S_hat [:] = self.S_hat [:]*0
        self.S [:] = self.S_hat [:]*0

        for t in range (T - 1):
            self.S_hat [:] = self.S_hat [:] * itau_s + self.S [:] * (1. - itau_s)

            self.H [:] = self.H * (1. - itau_m) + itau_m * ((self.J @ self.S_hat [:] if rec else 0)
                                                            + inp [:, t] + self.h)\
                                                         + self.Jreset @ self.S;

            self.S [:] = self._sigm (self.H, dv = self.dv) - 0.5 > 0.;
            Sout [:, t+1] = self.S.copy ();

        return Sout, self.Jout @ ut.sfilter (Sout, itau = self.par['beta_ro']);

    def implement (self, s_coll, a_coll, time_steps, adapt = False):
        if (self.J != 0).any ():
            print ('WARNING: Implement expert with non-zero recurrent weights\n');

        # First we parse experts input-output pair
        inps = s_coll#np.array ([exp[0] for exp in experts]);
        outs = a_coll#np.array ([exp[1] for exp in experts]);

        # Here we extract the maxima of input and output which are used to balance
        # the signal injection using the sigma_input and sigma_teach variables
        if adapt:
            self.sigma_input = 5. / np.max (np.abs (inps));
            self.sigma_teach = 5. / np.max (np.abs (outs));

            self.par['sigma_input'] = self.sigma_input;
            self.par['sigma_teach'] = self.sigma_teach;

            self.Jin = np.random.normal (0., self.sigma_input, size = (self.N, self.I));
            self.Jteach = np.random.normal (0., self.sigma_teach, size = (self.N, self.O));

        # We then build a target network dynamics

        #inps = inps.reshape(24,time_steps)

        Inp = self.Jin@inps#np.einsum ('ij, njt->nit', self.Jin, inps);
        tInp = Inp + self.Jteach@outs#np.einsum ('ij, njt->nit', self.Jteach, outs);

        #Targ = [self.compute (t_inp , rec = False)[0] for t_inp in tInp];
        #Targ = self.compute (tInp , rec = False)[0]
        Targ = self.compute (tInp , rec = True)[0]

        return Targ, Inp;

    def implement_LSM (self, s_coll, a_coll, time_steps, adapt = False):
        if (self.J != 0).any ():
            print ('WARNING: Implement expert with non-zero recurrent weights\n');

        # First we parse experts input-output pair
        inps = s_coll#np.array ([exp[0] for exp in experts]);
        outs = a_coll#np.array ([exp[1] for exp in experts]);

        # Here we extract the maxima of input and output which are used to balance
        # the signal injection using the sigma_input and sigma_teach variables
        if adapt:
            self.sigma_input = 5. / np.max (np.abs (inps));
            self.sigma_teach = 5. / np.max (np.abs (outs));

            self.par['sigma_input'] = self.sigma_input;
            self.par['sigma_teach'] = self.sigma_teach;

            self.Jin = np.random.normal (0., self.sigma_input, size = (self.N, self.I));
            self.Jteach = np.random.normal (0., self.sigma_teach, size = (self.N, self.O));

        # We then build a target network dynamics

        #inps = inps.reshape(24,time_steps)

        Inp = self.Jin@inps#np.einsum ('ij, njt->nit', self.Jin, inps);
        tInp = Inp + self.Jteach@outs#np.einsum ('ij, njt->nit', self.Jteach, outs);

        print("inp" + str( "{:+0.2f}".format(np.shape(Inp)[0]  ) ) )
        print("inp" + str( "{:+0.2f}".format(np.shape(Inp)[1]  ) ) )

        print(np.shape(tInp))

        #Targ = [self.compute (t_inp , rec = False)[0] for t_inp in tInp];
        Targ = self.compute (tInp , rec = True)[0]

        return Targ, Inp;

    def clone (self, s_coll, a_coll , targets, sigma_state,B, epochs = 500, rank = None, clumped = True):
        import matplotlib.pyplot as plt

        itau_m = self.itau_m;
        itau_s = self.itau_s;

        alpha = self.par['alpha'];
        alpha_rout = self.par['alpha_rout'];
        beta_ro = self.par['beta_ro'];
        beta_targ = 1-self.par['beta_targ'];
        S_rout = ut.sfilter (targets, itau = beta_ro)# for targ in targets];s

        adam_out = Adam (alpha = alpha_rout, drop = 0.9, drop_time = epochs // 10);

        # Here we train the network - online mode

        adam = Adam (alpha = alpha, drop = 0.9, drop_time = epochs // 10 * self.T);

        targets = np.array (targets)
        inps = np.array (self.Jin @ (s_coll +np.random.normal(0,0.0 ,size = (self.par["shape"][1],np.shape(s_coll)[1]) )));
        outs = np.copy(np.array (a_coll));
        #outs = P@np.array (a_coll);
        outs_nl = B@S_rout
        #outs_nl[4:-1,:] = sigm(np.array (outs[4:-1,:] ), 1.);
        #np.std(a_out[:,1:-1] - outs[:,0:-2] )
        print("estimated error = " + str(np.std(outs_nl[0:self.O,1:-1] - a_coll[:,0:-2])))
        #outs_nl[0:self.O,:] = np.copy(a_coll)

        #print(np.shape(outs_nl))
        #outs_nl = sigm(np.array (outs), 1.);

        dH = np.zeros (self.N);

        for epoch in range(epochs):#trange (epochs, leave = False, desc = 'Cloning'):
            #shuffle ((inps, outs, targets, S_rout));
            #for t in range(1):#out, s_out in zip (outs, S_rout):
            for nums in range(1):#inp, out, targ in zip (inps, outs, targets):
                self.reset ();
                dH *= 0;
                dj_tot = 0
                error = 0
                error_ro =0

                self.H [:] = self.H [:]*0+self.Vo
                self.S_hat [:] = self.S_hat [:]*0
                self.S [:] = self.S_hat [:]*0
                S_targ_hat = self.S_hat [:]*0
                S_pred_hat = self.S_hat [:]*0

                s_out = np.zeros((self.N,self.T),float)
                inps = np.array (self.Jin @ (s_coll +np.random.normal(0 , sigma_state ,size = (self.par["shape"][1],np.shape(s_coll)[1]) )));
                dJ = np.zeros((self.N,self.N),float)

                error_prop = 0
                error_prop_appr = 0
                error_ds = 0
                DH = 0

                for t in range (self.T - 1):

                    if clumped:

                        self.S_hat [:] = self.S_hat * itau_s + targets [:, t] * (1. - itau_s)
                        self.H [:] = self.H * (1. - itau_m) + itau_m * (self.J @ self.S_hat + inps [:, t] + self.h) + self.Jreset @ targets [:, t];

                    else:

                        self.S_hat [:] = self.S_hat * itau_s + s_out[:,t]* (1. - itau_s)
                        self.H [:] = self.H * (1. - itau_m) + itau_m * (self.J @ self.S_hat + inps [:, t] + self.h) + self.Jreset @ s_out[:,t];

                    dH [:] = dH  * (1. - itau_m) + itau_m * self.S_hat;
                    DH+=dH
                    s_out[:,t+1] =  np.heaviside (self.H,0)
                    self.S = s_out[:,t+1]

                    itau_targ = 0.1;

                    #print(beta_targ)

                    S_pred_hat = S_pred_hat * (1. - beta_targ) + s_out [:, t+1] * (beta_targ)
                    S_targ_hat = S_targ_hat * (1. - beta_targ) + targets [:, t+1] * ( beta_targ)

                    if rank>0:

                        #dJ = np.outer (D @(S_targ_hat - S_pred_hat)*self._dsigm (self.H, dv = 1), dH);#

                        err = B @ (S_targ_hat -  S_pred_hat)
                        err_eb = ( outs_nl[:,t] - B @ S_pred_hat)#R @ S_pred_hat)

                        error_prop_appr += np.std((err)[0:self.O])**2
                        error_prop += np.std((err_eb)[0:self.O])**2
                        error_ds += np.std(S_targ_hat -  S_pred_hat)**2

                        #dJ = np.outer ( B.T@err*self._dsigm (self.H, dv = 1) , dH);#
                        dJ = np.outer ( B.T@err*self._dsigm (self.H, dv = 1) , dH);#
                        #dJ = np.outer ( err_eb@B*self._dsigm (self.H, dv = 1) , dH);

                    else:
                        dJ = np.outer ((S_targ_hat - S_pred_hat) , dH);#
                    #dJ = np.outer (D @ (targets [:, t+1] - self._sigm(self.H, dv = self.dv)), dH);
                    #l2 = 10^-3
                    self.J += dJ*alpha #- l2*self.J
                    #self.J = adam.step(self.J,dJ)

                    S_rout[:,t+1] = S_rout[:,t]*beta_ro + (1-beta_ro)* s_out[:,t+1]
                    a_out = self.Jout @ S_rout

                    dj_tot += np.sum( np.abs(dJ) )
                    error += np.sum(np.abs( targets [:, t+1] - np.heaviside (self.H,0) ))

                    np.fill_diagonal (self.J, 0.);

                ftarg = np.mean(targets,1)
                fav = np.mean(s_out,1)

                dJ = np.outer ((ftarg - fav) , DH);#
                self.J += dJ*0.05
                np.fill_diagonal (self.J, 0.);

                print(np.mean(fav))
                print(np.mean(ftarg))

                stdMax = 3.
                stdJ = np.std(self.J)
                #if stdJ>stdMax:
                    #self.J = self.J/stdJ*stdMax
                self.J[self.J>10.]=10.
                self.J[self.J<-10.]=-10.
                print(stdJ)

                error_ro = np.sum( np.abs(a_out[0:4,1:-1] - outs[0:4,0:-2]) )

        #print(dj_tot)
        print("DS = " + str( "{: 0.0f}".format(error) ))
        print("out error appr" + str( "{: 0.5f}".format(error_prop_appr) ))
        print("out error" + str( "{: 0.2f}".format(error_prop) ))

        return error ,error_prop_appr/self.T,error_prop/self.T;

    def find_error (self, a_coll , targets, epochs = 500, rank = None):

        beta_ro = self.par['beta_ro'];
        S_rout = ut.sfilter (targets, itau = beta_ro)# for targ in targets];s
        a_out = self.Jout @ S_rout
        error = np.std(a_out[:,1:-1] - a_coll[:,0:-2] )
        print("find error = " + str(error))
        return error

    def clone_ro (self, a_coll , targets, epochs = 500, rank = None):
        import matplotlib.pyplot as plt


        T = np.shape(a_coll)[1]
        itau_m = self.itau_m;
        itau_s = self.itau_s;

        alpha = self.par['alpha'];
        alpha_rout = self.par['alpha_rout'];
        beta_ro = self.par['beta_ro'];

        S_rout = ut.sfilter (targets, itau = beta_ro)# for targ in targets];

        adam_out = Adam (alpha = alpha_rout, drop = 0.9, drop_time = epochs // 10);

        # Here we train the network - online mode
        adam = Adam (alpha = alpha, drop = 0.9, drop_time = epochs // 10 * self.T);

        targets = np.array (targets)
        #inps = np.array (self.Jin @ s_coll);
        outs = np.array (a_coll);

        dH = np.zeros (self.N);

        self.reset ();

        s_out = np.zeros((self.N,T),float)

        # Here we train the network
        for epoch in range(epochs):#  trange (epochs, leave = False, desc = 'Cloning'):
            S_rout = ut.sfilter (targets, itau = beta_ro)
            a_out = self.Jout @ S_rout
            #    a_out1 = Jout1 @ S_rout
            #    a_out2 = Jout2 @ S_rout

            dJ = (outs[:,0:-2] - a_out[:,1:-1]) @  S_rout[:,1:-1].T ;
            #    dJ1 = (outs - a_out1) @  S_rout.T ;
            #    dJ2 = (outs[:,1:-1] - a_out2[:,0:-2]) @  S_rout[:,0:-2].T ;

            #self.Jout += dJ*alpha_rout  #
            self.Jout = adam_out.step (self.Jout, dJ)
            #    Jout1 += dJ1*alpha_rout  #
            #    Jout2 += dJ2*alpha_rout  #

                #adam_out.step (self.Jout, dJ);

            #error = np.mean( np.abs(a_out[:,1:-1] - outs[:,0:-2]) )
            error = np.std(a_out[:,1:-1] - outs[:,0:-2] )

        plt.figure()
        plt.plot(a_out[0,:])
        plt.plot(outs[0,:])
        plt.savefig("outs_train.eps", format='eps')

            #error1 = np.sum( np.abs(a_out1 - outs))
            #error2 = np.sum( np.abs(a_out2[:,0:-2] - outs[:,1:-1]))

            #if np.mod(epoch,100)==0:
            #    print(error)
            #    print(error1)
            #    print(error2)


        return a_out,error;

    def reset (self, init = None):
        self.S [:] = init if init else np.zeros (self.N);
        self.S_hat [:] = self.S [:] * self.itau_s if init else np.zeros (self.N);

        self.out [:] *= 0;

        self.H [:] *= 0.;
        self.H [:] += self.Vo;


    def save (self, filename):
        # Here we collect the relevant quantities to store
        data_bundle = (self.Jin, self.Jteach, self.Jout, self.J, self.par)

        np.save (filename, np.array (data_bundle, dtype = np.object));

    @classmethod
    def load (cls, filename):
        data_bundle = np.load (filename, allow_pickle = True);

        Jin, Jteach, Jout, J, par = data_bundle;

        obj = LTTS (par);

        obj.Jin = Jin.copy ();
        obj.Jteach = Jteach.copy ();
        obj.Jout = Jout.copy ();
        obj.J = J.copy ();

        return obj;
