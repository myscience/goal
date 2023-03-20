"""
    © 2022 This work is licensed under a CC-BY-NC-SA license.
    Title: *"Error-based or target-based? A unifying framework for learning in recurrent spiking networks"*
    Authors: Cristiano Capone¹*, Paolo Muratore²* & Pier Stanislao Paolucci¹

    *: These authors equally contributed to the article

    ¹INFN, Sezione di Roma, Italy RM 00185
    ²SISSA - Internation School for Advanced Studies, Trieste, Italy
"""


import gym
#from PPO_continuous import PPO, Memory
#from PIL import Image
import torch
import numpy as np
from numpy import savetxt
import goal
import matplotlib.pyplot as plt

N, I, O, T = 500, 24, 4, 100;
shape = (N, I, O, T);

dt = .001# / T;
tau_m = 4. * dt;
tau_s = 2. * dt;
tau_ro = 3. * dt;
beta_s  = np.exp (-dt / tau_s);
beta_ro = np.exp (-dt / tau_ro);
sigma_teach = 5.0;
sigma_input = 50.0;
offT = 1;
dv = 1 / 5.;
alpha = .5;
alpha_rout = .0005;#0.1#.00002;
alpha_pg = 0.0005
Vo = -4;
h = -4;
s_inh = 20;

# Here we build the dictionary of the simulation parameters
par = {'tau_m' : tau_m, 'tau_s' : tau_s, 'tau_ro' : tau_ro, 'beta_ro' : beta_ro,
       'dv' : dv, 'alpha' : alpha, 'Vo' : Vo, 'h' : h, 's_inh' : s_inh,
       'N' : N, 'T' : T, 'dt' : dt, 'offT' : offT, 'alpha_rout' : alpha_rout,
       'sigma_input' : sigma_input, 'sigma_teach' : sigma_teach, 'shape' : shape};

# Here we init our model
ltts = goal.LTTS (par);
#ltts = ltts.load ('model_N500_si3.0_st2.0_1.npy');
#ltts = ltts.load ('model_N_st3.0_2_not_clumped_rank500_0.npy');

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(n_episodes,ltts):
    ############## Hyperparameters ##############
    env_name = "BipedalWalker-v3"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    #n_episodes = 10          # num of episodes to run
    max_timesteps = 1500    # max timesteps in one episode
    render = False           # render the environment
    save_gif = False        # png images are saved in gif folder

    # filename and directory to load model from
    #filename = "PPO_continuous_" +env_name+ ".pth"
    #directory = "./preTrained/"

    action_std = 0.5        # constant std for action distribution (Multivariate Normal)
    K_epochs = 80           # update policy for K epochs
    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr = 0.0003             # parameters for Adam optimizer
    betas = (0.9, 0.999)
    #############################################

    #memory = Memory()
    #ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    #ppo.policy_old.load_state_dict(torch.load(directory+filename))

    action_collection = np.zeros((4,0),dtype=float)
    state_collection = np.zeros((24,0),dtype=float)

    ltts.reset ()

    reward_collection = []

    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state = env.reset()
        #ltts = ltts.load ('model_N_st3.0_0_not_clumped_rank500_0.npy');

        ltts.reset ()

        #S_collection = np.zeros((N,0),dtype=int)
        S_dynamics = np.zeros((N,max_timesteps),dtype=float)# []
        v_dynamics = np.zeros((N,max_timesteps),dtype=float)
        action_dynamics = np.zeros((O,max_timesteps),dtype=float)
        state_dynamics = np.zeros((I,max_timesteps),dtype=float)

        for t in range(max_timesteps):
            #action = ppo.select_action(state, memory)

            action, S, dJ,v = ltts.step (state, 1 )
            state, reward, done, _ = env.step(action)

            #print(S)
            S_dynamics[:,t] = S
            v_dynamics[:,t] = v
            action_dynamics[:,t] = action
            state_dynamics[:,t] = state

            #S_collection = np.concatenate(S_collection,np.array(S))
            #S_collection.append(S)
            #v_collection = np.concatenate(v_collection,np.array(v))
            #ltts.J += dJ*alpha_pg*reward
    #print(ltts.J)
    # = 0#reward*alpha_pg*dJ
            #action_collection = np.concatenate( (action_collection.T,np.matrix(action)) ).T
            #print(np.shape(action_collection))
            #state_collection = np.concatenate( (state_collection.T,np.matrix(state)) ).# TEMP:
            #action, S = ltts.step (state, 1)
            ep_reward += reward
            if render:
                env.render()
            if save_gif:
                 img = env.render(mode = 'rgb_array')
                 img = Image.fromarray(img)
                 img.save('./gif/{}.jpg'.format(t))
            if done:
                break

            #savetxt("action_" + str(ep) + ".csv",action_collection)
            #savetxt("state_" + str(ep) + ".csv",state_collection)
        #np.array(S_collection).reshape(N,int(len(S_collection)/N)


        reward_collection.append(ep_reward)
        #print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        ep_reward = 0
        env.close()

    return np.mean(reward_collection)

def policy_gradient(n_episodes,ltts,avg_rev_pred):
    ############## Hyperparameters ##############
    env_name = "BipedalWalker-v3"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    #n_episodes = 10          # num of episodes to run
    max_timesteps = 1500    # max timesteps in one episode
    render = False           # render the environment
    save_gif = False        # png images are saved in gif folder

    # filename and directory to load model from
    #filename = "PPO_continuous_" +env_name+ ".pth"
    #directory = "./preTrained/"

    action_std = 0.5        # constant std for action distribution (Multivariate Normal)
    K_epochs = 80           # update policy for K epochs
    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr = 0.0003             # parameters for Adam optimizer
    betas = (0.9, 0.999)
    #############################################

    #memory = Memory()
    #ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    #ppo.policy_old.load_state_dict(torch.load(directory+filename))

    action_collection = np.zeros((4,0),dtype=float)
    state_collection = np.zeros((24,0),dtype=float)

    ltts.reset ()

    reward_collection = []

    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state = env.reset()
        ltts.reset ()

        dJ_tot = np.zeros((ltts.N,ltts.N),dtype = float)

        for t in range(max_timesteps):
            #action = ppo.select_action(state, memory)

            action, S, dJ, v = ltts.step (state, 1, probabilistic = True)

            state, reward, done, _ = env.step(action)
            dJ_tot += dJ#*alpha_pg*reward

            #action_collection = np.concatenate( (action_collection.T,np.matrix(action)) ).T
            #print(np.shape(action_collection))
            #state_collection = np.concatenate( (state_collection.T,np.matrix(state)) ).# TEMP:

            #action, S = ltts.step (state, 1)

            ep_reward += reward
            if render:
                env.render()
            if save_gif:
                 img = env.render(mode = 'rgb_array')
                 img = Image.fromarray(img)
                 img.save('./gif/{}.jpg'.format(t))
            if done:
                break

            #savetxt("action_" + str(ep) + ".csv",action_collection)
            #savetxt("state_" + str(ep) + ".csv",state_collection)
        reward_collection.append(ep_reward)
        ltts.J += dJ_tot*alpha_pg*(ep_reward-avg_rev_pred)
        #print(np.std(dJ_tot))

        #print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        ep_reward = 0
        env.close()

    return np.mean(reward_collection),np.std(reward_collection)

if __name__ == '__main__':
    reward = test(10,ltts)
    print(reward)
