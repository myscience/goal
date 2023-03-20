"""
    © 2022 This work is licensed under a CC-BY-NC-SA license.
    Title: *"Error-based or target-based? A unifying framework for learning in recurrent spiking networks"*
    Authors: Cristiano Capone¹*, Paolo Muratore²* & Pier Stanislao Paolucci¹

    *: These authors equally contributed to the article

    ¹INFN, Sezione di Roma, Italy RM 00185
    ²SISSA - Internation School for Advanced Studies, Trieste, Italy
"""


import matplotlib.pyplot as plt
import numpy as np
#import bipedal_walker_test as bw
from numpy import savetxt,loadtxt
import pylab as py
import os.path
import os


folder = "RewardsErrors_presaved"

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

plt.figure()

n_iter_out = 15
number_of_rpng = 10
sigma_teach =  1.0
smooth_window = 1

#tau_vals = [.01,0.025,.05, .1,.25,.5, 1.,2.,5., 10.]
#tau_vals = np.arange(4,500,50)#[.025, .25, 2.,10.]
#print(rank_vals)

plt.subplot(311)

max_avg = []
max_avg_0 = []
max_avg_1 = []
max_avg_2 = []
max_avg_3 = []

max_std = []

rank = 500#rank_vals[k]

rank_vals = [4,500]
tau_vals = [0.1,3.,10.]

cmap = plt.get_cmap('copper')
colors = [cmap(i) for i in np.linspace(0, 1, len(tau_vals))]

max_rews = np.zeros((3,2))

# Create four polar axes and access them through the returned array
fig, axs = plt.subplots(3, 2)#, subplot_kw=dict(projection="polar"))

dt = 0.001
for n_rank in range(len(rank_vals)):
	for n_tau in range(len(tau_vals)):

		avg_reward = np.zeros(( 0 , n_iter_out - smooth_window+1),dtype=int)
		max_rewards = []
		rank = rank_vals[n_rank]
		tau_targ = tau_vals[n_tau]*dt

		all_rewards = np.zeros((number_of_rpng,n_iter_out))
		max_rew_coll = []

		for n_rep in range(0,number_of_rpng):
			print(n_rep)
			path = "avg_reward" + "_st" + str(sigma_teach) + "_" + str(n_rep) + "_not_clumped_rank" + str(rank) + "_tau_" + str(tau_targ) + ".npy"
			arr = np.load((os.path.join(folder,path) ) )#)
			#arr = np.matrix( moving_average(arr,smooth_window) )
			#avg_reward = np.concatenate((avg_reward ,arr ))
			#max_rewards.append(np.max(arr))
			axs[n_tau , n_rank].plot(arr.T,color=colors[n_tau], lw=.5)#,color=colors[k]
			axs[n_tau , n_rank].set_ylim(-120.,120.)
			all_rewards[n_rep,:] = arr # = np.concatenate((all_rewards ,np.median(avg_reward,0) ))
			max_rew_coll.append(np.max(arr))
		axs[n_tau , n_rank].plot(np.mean(all_rewards.T,1),color=colors[n_tau], lw=2.)
		max_rews[n_tau , n_rank] = np.mean(max_rew_coll)

path = "rank_chunk_tau" + "_st" + str(sigma_teach) + "_9.png"
plt.savefig(os.path.join(folder,path)  , format='png')

plt.figure()
plt.matshow(max_rews)
plt.colorbar()
path = "rank_chunk_tau_imwhow" + "_st" + str(sigma_teach) + "_9.png"
plt.savefig(os.path.join(folder,path)  , format='png')


# Create four polar axes and access them through the returned array
fig, axs = plt.subplots(3, 2)#, subplot_kw=dict(projection="polar"))

n_iter_out = 500

max_rews = np.zeros((3,2))
for n_rank in range(len(rank_vals)):
	for n_tau in range(len(tau_vals)):

		avg_reward = np.zeros(( 0 , n_iter_out - smooth_window+1),dtype=int)
		max_rewards = []
		rank = rank_vals[n_rank]
		tau_targ = tau_vals[n_tau]*dt

		all_rewards = np.zeros((number_of_rpng,n_iter_out))
		max_rew_coll = []

		for n_rep in range(0,number_of_rpng):
			print(n_rep)
			path = "stdJ" + "_st" + str(sigma_teach) + "_" + str(n_rep) + "_not_clumped_rank" + str(rank) + "_tau_" + str(tau_targ) + ".npy"
			arr = np.load((os.path.join(folder,path) ) )#)
			#arr = np.matrix( moving_average(arr,smooth_window) )
			#avg_reward = np.concatenate((avg_reward ,arr ))
			#max_rewards.append(np.max(arr))
			axs[n_tau , n_rank].plot(arr.T,color=colors[n_tau], lw=.5)#,color=colors[k]
			axs[n_tau , n_rank].set_ylim(0.,1.5)
			all_rewards[n_rep,:] = arr # = np.concatenate((all_rewards ,np.median(avg_reward,0) ))
			max_rew_coll.append(np.max(arr))
		axs[n_tau , n_rank].plot(np.mean(all_rewards.T,1),color=colors[n_tau], lw=2.)

		max_rews[n_tau , n_rank] = np.mean(max_rew_coll)

path = "stdJ_tau" + "_st" + str(sigma_teach) + "_9.png"
plt.savefig(os.path.join(folder,path)  , format='png')

# Create four polar axes and access them through the returned array
fig, axs = plt.subplots(3, 2)#, subplot_kw=dict(projection="polar"))

max_rews = np.zeros((3,2))
for n_rank in range(len(rank_vals)):
	for n_tau in range(len(tau_vals)):

		avg_reward = np.zeros(( 0 , n_iter_out - smooth_window+1),dtype=int)
		max_rewards = []
		rank = rank_vals[n_rank]
		tau_targ = tau_vals[n_tau]*dt

		all_rewards = np.zeros((number_of_rpng,n_iter_out))
		max_rew_coll = []

		for n_rep in range(0,number_of_rpng):
			print(n_rep)
			path = "mse_app" + "_st" + str(sigma_teach) + "_" + str(n_rep) + "_not_clumped_rank" + str(rank) + "_tau_" + str(tau_targ) + ".npy"
			arr = np.load((os.path.join(folder,path) ) )#)

			axs[n_tau , n_rank].plot(arr.T,color=colors[n_tau], lw=.5)#,color=colors[k]
			axs[n_tau , n_rank].set_ylim(0.,1.5)
			all_rewards[n_rep,:] = arr # = np.concatenate((all_rewards ,np.median(avg_reward,0) ))
			max_rew_coll.append(np.max(arr))
		#axs[n_tau , n_rank].plot(np.mean(all_rewards.T,1),color=colors[n_tau], lw=2.)
		#max_rews[n_tau , n_rank] = np.mean(max_rew_coll)

path = "mse_app_tau" + "_st" + str(sigma_teach) + "_9.png"
plt.savefig(os.path.join(folder,path)  , format='png')


# Create four polar axes and access them through the returned array
fig, axs = plt.subplots(3, 2)#, subplot_kw=dict(projection="polar"))

max_rews = np.zeros((3,2))
for n_rank in range(len(rank_vals)):
	for n_tau in range(len(tau_vals)):

		avg_reward = np.zeros(( 0 , n_iter_out - smooth_window+1),dtype=int)
		max_rewards = []
		rank = rank_vals[n_rank]
		tau_targ = tau_vals[n_tau]*dt

		all_rewards = np.zeros((number_of_rpng,n_iter_out))
		max_rew_coll = []

		for n_rep in range(0,number_of_rpng):
			print(n_rep)
			path = "ds" + "_st" + str(sigma_teach) + "_" + str(n_rep) + "_not_clumped_rank" + str(rank) + "_tau_" + str(tau_targ) + ".npy"
			arr = np.load((os.path.join(folder,path) ) )#)

			axs[n_tau , n_rank].plot(arr.T,color=colors[n_tau], lw=.5)#,color=colors[k]
			axs[n_tau , n_rank].set_ylim(0.,10000.)
			all_rewards[n_rep,:] = arr # = np.concatenate((all_rewards ,np.median(avg_reward,0) ))
			max_rew_coll.append(np.max(arr))
		#axs[n_tau , n_rank].plot(np.mean(all_rewards.T,1),color=colors[n_tau], lw=2.)
		#max_rews[n_tau , n_rank] = np.mean(max_rew_coll)

path = "ds_.png"
plt.savefig(os.path.join(folder,path)  , format='png')


"""

        plt.plot(np.mean(avg_reward,0).T,color=colors[k],lw=2)
        max_avg.append(np.max(np.median(avg_reward,0).T))
        max_avg_0.append(np.mean(np.median(avg_reward,0).T[7:-1]))
        max_avg_1.append(np.mean(np.mean(avg_reward,0).T[0:5]))
        max_avg_2.append(np.mean(np.mean(avg_reward,0).T[1:2]))

        #max_avg.append( np.mean(avg_reward,0)[1,:] )

        #max_avg.append(np.mean(avg_reward[-1,:]))
        max_avg_3.append(np.median(max_rewards))
        max_std.append(np.std(max_rewards))
        """
"""
print(np.shape(max_avg))
print(np.shape(max_std))

min_val_err = []

plt.subplot(312)

for k in range(len(tau_vals)):
    #sigma_teach = sigma_teach_vals[k]
    validation_error_collection = np.zeros(( 0 , n_iter_out ),dtype=int)
    for n_rep in range(0,number_of_rpng):

        #arr = np.load("validation_error_collection" + "_st" + str(sigma_teach) + "_" + str(n_rep) + "_not_clumped_rank" + str(rank) + ".npy")
        path = "validation_error_collection" + "_st" + str(sigma_teach) + "_" + str(n_rep) + "_not_clumped_tau" + str(tau_vals[k]) + ".npy"
        arr = np.load((os.path.join(folder,path) ) )#)

        arr = np.matrix(arr)
        #print(np.shape(arr))
        validation_error_collection = np.concatenate((validation_error_collection ,arr ))

    min_val_err.append(np.min(np.mean(validation_error_collection,0).T))

    plt.plot(np.mean(validation_error_collection,0).T,color=colors[k],label=str(rank))
#plt.legend()


plt.subplot(313)

min_train_err = []

for k in range(len(tau_vals)):
    #sigma_teach = sigma_teach_vals[k]
    #rank = 500#rank_vals[k]
    training_error_collection = np.zeros(( 0 , n_iter_out ),dtype=int)
    for n_rep in range(0,number_of_rpng):

        #arr = np.load("training_error_collection" + "_st" + str(sigma_teach) + "_" + str(n_rep) + "_not_clumped_rank" + str(rank) + ".npy")
        path = "training_error_collection" + "_st" + str(sigma_teach) + "_" + str(n_rep) + "_not_clumped_tau" + str(tau_vals[k]) + ".npy"
        arr = np.load((os.path.join(folder,path) ) )#)
        arr = np.matrix(arr)
        #print(np.shape(arr))
        training_error_collection = np.concatenate((training_error_collection ,arr ))

    min_train_err.append(np.min(np.mean(training_error_collection,0).T))

    plt.plot(np.mean(training_error_collection,0).T,color=colors[k])


path = "comparison_val_errors_st" + "_st" + str(sigma_teach) + "_tau_9.png"
plt.savefig(os.path.join(folder,path)  , format='png')

plt.figure()
plt.subplot(411)

plt.plot( tau_vals, max_avg  ,'o')
plt.errorbar( tau_vals, max_avg_3, max_std/np.sqrt(number_of_rpng)  )
plt.xscale('log')#)

plt.ylabel('reward')

plt.subplot(413)
plt.plot(tau_vals,min_val_err,'-o')
#plt.xlabel('rank')
plt.ylabel('val err')
plt.xscale('log')#)

plt.subplot(414)
plt.plot(tau_vals,min_train_err,'-o')
plt.xlabel('rank')
plt.ylabel('train err')
plt.xscale('log')#)

plt.xlabel('$ \tau_{targ} (ms)$')

DS_mean = []
DS_std = []


for k in range(len(tau_vals)):
#for n_rep in range(0,number_of_rpng_eb):
    #sigma_teach = sigma_teach_vals[k]
    DS_rep = []
    avg_reward = np.zeros(( 0 , n_iter_out - smooth_window+1),dtype=int)
    max_rewards = []
    for n_rep in range(number_of_rpng):#,number_of_rpng):
        print(n_rep)

        path = "DS" + "_st" + str(sigma_teach) + "_" + str(n_rep) + "_not_clumped_tau" + str(tau_vals[k]) + ".npy"
        DS = np.load((os.path.join(folder,path) ) )#)

        #arr = np.load("avg_reward" + "_st" + str(sigma_teach) + "_" + str(n_rep) + "_not_clumped_rank" + str(rank) + ".npy" )#)
        DS_rep.append(DS)
    DS_mean.append(np.mean(DS_rep))
    DS_std.append(np.std(DS_rep))

plt.subplot(412)

#plt.plot(tau_vals,DS_Coll)
plt.errorbar(tau_vals, DS_mean, DS_std/np.sqrt(number_of_rpng))
plt.xscale('log')#)
plt.yscale('log')#)

plt.ylabel('$\Delta S$')

path = "comparison_max_reward_vs_rank" + "_st" + str(sigma_teach) + "_tau_9.png"
plt.savefig(os.path.join(folder,path)  , format='png')

plt.figure()
plt.matshow(all_rewards)
plt.colorbar()
plt.xlabel('iteration')
plt.ylabel('rank')

path = "all_rewards_vs_rank_" + "_st" + str(sigma_teach) + "_tau_9.png"
plt.savefig(os.path.join(folder,path)  , format='png')

#plt.savefig("DS_rewards_vs_rank_" + "_st" + str(sigma_teach) + "_tau.png", format='png')

max_avg_chunck = []
tau_chunck = []
max_std_chunck = []

chunk_size = 2

for k in range(len(tau_vals)//chunk_size):#np.floor(len(max_avg)/chunk_size)):
    max_avg_chunck.append(np.mean(max_avg_3[k*chunk_size:(k+1)*chunk_size]))
    max_std_chunck.append(np.std(max_avg_3[k*chunk_size:(k+1)*chunk_size]))
    tau_chunck.append(np.mean(tau_vals[k*chunk_size:(k+1)*chunk_size]))

plt.figure()
plt.errorbar(tau_chunck, max_avg_chunck, max_std_chunck, marker='s', mfc='red', mec='green', ms=20, mew=4)
plt.plot([tau_chunck[0], tau_chunck[-1]],[ max_avg_3[0],max_avg_3[0] ])
#plt.errorbar(np.ravel(rank_chunck),np.ravel(max_avg_chunck),np.ravel(max_std_chunck),'-o')
#plt.plot(np.array(rank_chunck),np.array(max_avg_chunck),'-o')
plt.xscale('log')#)
plt.xlabel('$\tau$')
plt.ylabel('reward')

path = "rank_chunk_tau" + "_st" + str(sigma_teach) + "_9.png"
plt.savefig(os.path.join(folder,path)  , format='png')


fig = plt.figure()



plt.subplot(131)

#plt.plot(tau_vals,DS_Coll)
plt.errorbar(tau_vals, DS_mean, DS_std/np.sqrt(number_of_rpng))
plt.xscale('log')
plt.yscale('log')

plt.xlabel('$\u03C4_*$')
plt.ylabel('$\Delta S$')

plt.subplot(132)
#fig.set_size_inches(2.7, 4.5)
fig.set_size_inches(7.5, 2.5)

plt.plot( tau_vals, max_avg_3  ,'o')
plt.errorbar( tau_vals, max_avg_3, max_std/np.sqrt(number_of_rpng)  )
plt.xscale('log')
plt.ylabel('<reward>')
plt.xlabel('$\u03C4_*$')


plt.subplot(133)

plt.plot( DS_mean, max_avg_3  ,'-o')
#plt.errorbar( tau_vals, max_avg_3, max_std/np.sqrt(number_of_rpng)  )

plt.ylabel('<reward>')
plt.xlabel('$\Delta S$')



plt.tight_layout()

path = "FigS3.png"
plt.savefig(os.path.join(folder,path)  , format='png')

#os.chdir('/..')
"""
