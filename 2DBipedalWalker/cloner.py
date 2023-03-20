"""
    © 2022 This work is licensed under a CC-BY-NC-SA license.
    Title: *"Error-based or target-based? A unifying framework for learning in recurrent spiking networks"*
    Authors: Cristiano Capone¹*, Paolo Muratore²* & Pier Stanislao Paolucci¹

    *: These authors equally contributed to the article

    ¹INFN, Sezione di Roma, Italy RM 00185
    ²SISSA - Internation School for Advanced Studies, Trieste, Italy
"""

import goal
import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt,loadtxt
from tqdm import trange
import test_cloner
import os.path
import os

folder = "RewardsErrors"

time_steps = 400
t0 = 0

N, I, O, T = 500, 24, 4, time_steps;
shape = (N, I, O, T);


n_examples = 1
n_examples_validation = 1

num_iterations = 100
num_iterations_out = 15

epochs_out =  150

# Here we define our model

dt = .001
tau_m = 4. * dt
tau_s = 2. * dt
tau_ro = 3. * dt
tau_targ = tau_ro
beta_s  = np.exp (-dt / tau_s)
beta_ro = np.exp (-dt / tau_ro)
beta_targ = np.exp (-dt / tau_targ)

sigma_teach = 1.
sigma_input = 2.
offT = 0
dv = 1 / 5.
alpha = .03*.25*.5
alpha_rout = .0015*.25
Vo = -4
h = -4
s_inh = 20

# Load Data To Clone
steps = time_steps

Jteach = np.random.normal(0,1,size=(N,O))
Jin = np.random.normal(0,1,size=(N,I))
B = np.random.normal(0,1,size=(N,N))*.1
J = np.random.normal(0,1,size=(N,N))*.1


rank_vals = [4,500]
tau_vals = [0.1,3.,10.]

n_reps = 10

for n_rep in range(n_reps):
	for n_rank in range(len(rank_vals)):
		for n_tau in range(len(tau_vals)):
			rank = rank_vals[n_rank]
			tau_ro = tau_vals[n_tau]*dt

			beta_ro = np.exp (-dt / tau_ro)
			tau_targ = tau_ro
			beta_targ = beta_ro

			alpha_rank = alpha/np.sqrt(rank)*np.sqrt(500)

			# Here we build the dictionary of the simulation parameters
			par = {'tau_m' : tau_m, 'tau_s' : tau_s, 'tau_ro' : tau_ro, 'beta_ro' : beta_ro,'beta_targ' : beta_targ,
				   'dv' : dv, 'alpha' : alpha_rank, 'Vo' : Vo, 'h' : h, 's_inh' : s_inh,
				   'N' : N, 'T' : T, 'dt' : dt, 'offT' : offT, 'alpha_rout' : alpha_rout,
				   'sigma_input' : sigma_input, 'sigma_teach' : sigma_teach, 'shape' : shape};

			del goal
			import goal
			ltts = goal.LTTS (par);

			ltts.J *=0.
			ltts.Jout *=0.

			np.fill_diagonal (ltts.J, 0.);

			a_coll = []
			s_coll = []

			demonstrations_nums = [1]
			validation_nums = [3]

			a_coll_load = [ loadtxt("action_state/action_" + str( demonstrations_nums[n_batch]  ) + ".csv") for n_batch in range(n_examples)]
			s_coll_load = [ loadtxt("action_state/state_" + str( demonstrations_nums[n_batch]  ) + ".csv") for n_batch in range(n_examples)]

			a_coll_val_load = [ loadtxt("action_state/action_" + str( validation_nums[n_batch]  ) + ".csv") for n_batch in range(n_examples_validation)]
			s_coll_val_load = [ loadtxt("action_state/state_" + str( validation_nums[n_batch]  ) + ".csv") for n_batch in range(n_examples_validation)]

			itargets = [ltts.implement (s_coll_load[n_batch][0:I,0+t0:time_steps+t0],a_coll_load[n_batch][:,0+t0:time_steps+t0],time_steps)[0] for n_batch in range(n_examples)]
			inputs = [ltts.implement (s_coll_load[n_batch][0:I,0+t0:time_steps+t0],a_coll_load[n_batch][:,0+t0:time_steps+t0],time_steps)[1] for n_batch in range(n_examples)]

			itargets_val = [ltts.implement (s_coll_val_load[n_batch][0:I,0+t0:time_steps+t0],a_coll_val_load[n_batch][:,0+t0:time_steps+t0],time_steps)[0] for n_batch in range(n_examples_validation)]
			inputs_val = [ltts.implement (s_coll_val_load[n_batch][0:I,0+t0:time_steps+t0],a_coll_val_load[n_batch][:,0+t0:time_steps+t0],time_steps)[1] for n_batch in range(n_examples_validation)]

			a_aggr = np.zeros((O,0),dtype=float)
			s_aggr = np.zeros((I,0),dtype=float)
			S_aggr = np.zeros((N,0),dtype=float)
			
			#Here the internal target of spikes are defined
			
			for n_batch in range(n_examples):

				a_coll = a_coll_load[n_batch][:,0+t0:time_steps+t0]
				a_coll[:,0:offT] = 0
				a_aggr = np.concatenate( (a_aggr.T,a_coll.T) ).T
				S_gen, action = ltts.compute (inputs[n_batch][:,:]);
				S_aggr = np.concatenate( (S_aggr.T,itargets[n_batch].T) ).T
				
			#Here a rudimentary RO is learned to project errors with weights that are not totally random


			for kk in trange (20, leave = False, desc = 'RO'):
				a_out,error = ltts.clone_ro ( a_aggr , S_aggr, epochs = 500)
				#print(np.std(ltts.Jout))
				print(error)
			
			
			stdB_0 = 0.4
			ltts.par["alpha"] = ltts.par["alpha"] / np.std(ltts.Jout)*stdB_0
			
			
			B = np.random.normal(0, np.std(ltts.Jout)*2 ,size=(rank,N))
			
			if rank>0:

				B[0:O,:] = np.copy(ltts.Jout)#B[0:rank,:]#
			
			print(np.std(B))
			
			## Here the internal targets are learned by adjusting recurrent weights following the goal learning rule

			if rank>-1:
				DDDS = []
				DY = []
				DY_APPR = []
				stdJ = []

				for kk in trange (num_iterations, leave = False, desc = 'Cloning'):

					internal_error=0
					internal_error_val=0
					external_error=0
					external_error_val=0

					
					for n_batch in range(n_examples):

						a_coll = a_coll_load[n_batch][:,0+t0:time_steps+t0]
						a_coll[:,0:offT] = 0
						s_coll = s_coll_load[n_batch][0:I,0+t0:time_steps+t0]

						DDS , dy_appr, dy = ltts.clone ( s_coll, a_coll , itargets[n_batch][:,:], 0. ,B, epochs = 1,rank = rank,clumped = False);

						S_gen, action = ltts.compute (inputs[n_batch][:,:]);
						internal_error += np.mean(np.abs(S_gen - itargets[n_batch][:,:]))
						#external_error += np.mean( np.abs(a_out[:,1:-1] - a_coll[:,0:-2]) ) #np.mean(a_coll[:,] - action)

					DDDS.append(DDS)
					DY_APPR.append(dy_appr)
					DY.append(dy)
					stdJ.append(np.std(ltts.J))

					plt.figure(figsize=(12,3))
					plt.subplot(131)
					plt.plot(DDDS)
					plt.ylabel('DS')
					plt.xlabel('# iter')

					plt.subplot(132)
					plt.plot(DY_APPR)
					plt.ylabel('mse/appr')
					plt.xlabel('# iter')

					plt.plot(DY)

					plt.subplot(133)
					plt.plot(stdJ)
					plt.ylabel('std J')
					plt.xlabel('# iter')

					plt.savefig(os.path.join(folder,"DS" + str(N) + "_si" + str(sigma_input) + "_st" + str(sigma_teach) + "_" + str(n_rep)  + "_n_ex" +str(n_examples) + "_rank_" +str(rank) + "_tau_" + str(tau_targ)  +".png"), format='png')
					plt.savefig(os.path.join(folder,"DS" + str(N) + "_si" + str(sigma_input) + "_st" + str(sigma_teach) + "_" + str(n_rep)  + "_n_ex" +str(n_examples) + "_rank_" +str(rank) + "_tau_" + str(tau_targ)  +".eps"), format='eps')

			#print(np.std(ltts.J))

			internal_error_coll = []
			external_error_coll = []
			internal_error_val_coll = []
			external_error_val_coll = []

			dist_coll = []
			dist_val_coll = []
			dist_val_m_coll = []
			dist_test_coll = []

			a_aggr = np.zeros((O,0),dtype=float)
			s_aggr = np.zeros((I,0),dtype=float)
			S_aggr = np.zeros((N,0),dtype=float)

			for n_batch in range(n_examples):

				a_coll = a_coll_load[n_batch][:,0+t0:time_steps+t0]
				a_coll[:,0:offT] = 0
				a_aggr = np.concatenate( (a_aggr.T,a_coll.T) ).T
				S_gen, action = ltts.compute (inputs[n_batch][:,:]);
				S_aggr = np.concatenate( (S_aggr.T,S_gen.T) ).T

			avg_reward_collection = []
			validation_error_collection = []
			training_error_collection = []

		
			ltts.Jout = np.random.normal(0,1,size=(O,N))*.0
		
			plt.figure()

			for kk in trange (num_iterations_out, leave = False, desc = 'Cloning'):

				external_error=0
				external_error_val=0
				internal_error_val=0

				a_out,error = ltts.clone_ro ( a_aggr , S_aggr  , epochs = epochs_out);
				external_error = error#np.mean( np.abs(a_out[:,1:-1] - a_coll[:,0:-2]) ) #np.mean(a_coll[:,] - action)

				#external_error = external_error#/n_examples

				avg_reward = test_cloner.test(10,ltts)

				avg_reward_collection.append(avg_reward)

				for n_batch in range(n_examples_validation):

					S_gen_val, action_val = ltts.compute (inputs_val[n_batch][:,0:time_steps])
					external_error_val += np.std(action_val[:,1:time_steps-1]-a_coll_val_load[n_batch][:,0:time_steps-2])
		
				validation_error_collection.append(external_error_val)
				training_error_collection.append(external_error)

				print("- ext error: " + str( "{: 0.4f}".format(external_error) ))
				print("- ext error val: " + str( "{: 0.4f}".format(external_error_val) ))
				print("- avg reward: " + str( "{: 0.4f}".format(avg_reward) ))

				plt.figure(figsize=(12,6))
				plt.subplot(231)
				plt.plot(DDDS)
				plt.ylabel('DS')
				plt.xlabel('# iter')

				plt.subplot(232)
				plt.plot(DY_APPR)
				plt.ylabel('mse/appr')
				plt.xlabel('# iter')

				plt.plot(DY)
				#plt.ylabel('mse')
				#plt.xlabel('# iter')

				plt.subplot(233)
				plt.plot(stdJ)
				plt.ylabel('std J')
				plt.xlabel('# iter')

				plt.subplot(234)

				plt.plot(avg_reward_collection,color='orange', lw=2)

				plt.xlabel("iterations")
				plt.ylabel('reward')

				plt.subplot(235)

				plt.plot(training_error_collection,color='blue', lw=2)
				plt.plot(validation_error_collection,color='green', lw=2)

				#plt.legend()

				plt.yscale('log')
				plt.xlabel("iterations")
				plt.ylabel('error')

				plt.savefig(os.path.join(folder,"summary" + str(N) + "_si" + str(sigma_input) + "_st" + str(sigma_teach)  + "_" + str(n_rep)  + "_n_ex" +str(n_examples) + "_rank_" +str(rank) + "_tau_" + str(tau_targ)  + ".eps"), format='eps')

				np.save(os.path.join(folder,"training_error_collection" + "_st" + str(sigma_teach) + "_" + str(n_rep) + "_not_clumped_rank" + str(rank) + "_tau_" + str(tau_targ)  + ".npy"), training_error_collection)
				np.save(os.path.join(folder,"validation_error_collection" + "_st" + str(sigma_teach) + "_" + str(n_rep) + "_not_clumped_rank" + str(rank) + "_tau_" + str(tau_targ)  + ".npy"), validation_error_collection)
				np.save(os.path.join(folder,"avg_reward" + "_st" + str(sigma_teach) + "_" + str(n_rep) + "_not_clumped_rank" + str(rank) + "_tau_" + str(tau_targ) + ".npy"), avg_reward_collection)
			np.save(os.path.join(folder,"mse" + "_st" + str(sigma_teach) + "_" + str(n_rep) + "_not_clumped_rank" + str(rank) + "_tau_" + str(tau_targ)  + ".npy"), DDDS)
			np.save(os.path.join(folder,"mse_app" + "_st" + str(sigma_teach) + "_" + str(n_rep) + "_not_clumped_rank" + str(rank) + "_tau_" + str(tau_targ)  + ".npy"), DY_APPR)
			np.save(os.path.join(folder,"ds" + "_st" + str(sigma_teach) + "_" + str(n_rep) + "_not_clumped_rank" + str(rank) + "_tau_" + str(tau_targ) + ".npy"), DY)
			np.save(os.path.join(folder,"stdJ" + "_st" + str(sigma_teach) + "_" + str(n_rep) + "_not_clumped_rank" + str(rank) + "_tau_" + str(tau_targ) + ".npy"), stdJ)

			ltts.save (os.path.join(folder,"model_N" + "_st" + str(sigma_teach) + "_" + str(n_rep) + "_not_clumped_rank" + str(rank) + "_tau_" + str(tau_targ)  + ".npy" ))
