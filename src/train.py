import torch
import pickle
import warnings
from timeit import default_timer
from argparse import ArgumentParser

from math import sqrt

from utils import train
from utils import build_clock, build_target

parser = ArgumentParser()

n_msg = 'Number of units [int] in the recurrent network. This parameter is required.'
i_msg = 'Number of input units. This units provide a clock-like input to the network. [Default n_inp = 5]'
o_msg = 'Number of output units. This units collects the network readout. [Default n_out = 3]'

t_msg = 'Length of temporal sequence to learn. [Default T = 150 (dt)]'
dt_msg = 'Discrete time interval (in seconds). [Default dt = 0.001]'

epo_msg = 'Number of epochs spent training the model. [Default epochs = (5000, 10000)]. First is readout training, second is recurrent training.'

tau_m_msg = 'Membrane time constant. [Default tau_m = 8 (dt)]'
tau_s_msg = 'Spike-filter time constant. [Default tau_s = 2 (dt)]'
tau_o_msg = 'Readout-filter time constant. [Default tau_o = 5 (dt)]'

rnk_msg = 'Rank [int] of the feedback matrix. Should be in range [n_out, n_rec]. If not provided, the LTTS algorithm is used'

sgm_i_msg = 'Input Gaussian Random Matrix variance. Use for projecting the input into the network. [Default sigma_inp = 22]'
sgm_t_msg = 'Target Gaussian Random Matrix variance. Use for projecting the target into the network. [Default sigma_inp = 8]'

vo_msg   = 'Initial condition for the unit membrane potential. [Default Vo = -4]' 
rst_msg  = 'Reset current that is injected into a unit after it has produced a spike. [Default reset = -20]'
bias_msg = 'External uniform electric field that every unit experience. Used to bias the network mean firing rate. [Default = -4]'

rlr_msg = 'Learning rate used for recurrent synaptic weights update. [Default rec_lr = 5 / sqrt(N)'
olr_msg = 'Learning rate used for readout synaptic weights update. [Default out_lr = 0.0025]'

gpu_msg = 'Flag that is used to host the PyTorch model on the GPU. Only single-GPU training is supported.'
res_msg = 'Resample flag that indicates whether the optimizer shall resample the feedback matrix at each iteration. [Default False]'
vrb_msg = 'Verbose flag to request model feedbacks during training.'

fbck_msg = 'Feedback target: whether the signal should be computed at the output or spike level. Possible values: ["spk", "out"]. [Default = "spk"]'
save_msg = 'Savepath for storing the training statistics.'

parser.add_argument('-N', '-n_rec', required = True,  type = int, dest = 'n_rec', help = n_msg)
parser.add_argument('-I', '-n_inp', required = False, type = int, dest = 'n_inp', help = i_msg, default = 5)
parser.add_argument('-O', '-n_out', required = False, type = int, dest = 'n_out', help = o_msg, default = 3)

parser.add_argument('-dt', required = False, type = int, dest = 'dt', help = dt_msg, default = 0.001)
parser.add_argument('-T', '-t_seq', required = False, type = int, dest = 't_seq', help = t_msg, default = 150)

parser.add_argument('-E', '-epochs', nargs = 2, required = False, type = int, dest = 'epochs', help = epo_msg, default = (5000, 1000))

parser.add_argument('-Tm', '-tau_m', required = False, type = float, dest = 'tau_m', help = tau_m_msg, default = 8)
parser.add_argument('-Ts', '-tau_s', required = False, type = float, dest = 'tau_s', help = tau_s_msg, default = 2)
parser.add_argument('-To', '-tau_o', required = False, type = float, dest = 'tau_o', help = tau_s_msg, default = 5)

parser.add_argument('-R', '-rank', required = False, type = int, dest = 'rank', help = rnk_msg, default = None)

parser.add_argument('-Si', '-sigma_inp', required = False, type = float, dest = 'sigma_inp', help = sgm_i_msg, default = 22)
parser.add_argument('-St', '-sigma_trg', required = False, type = float, dest = 'sigma_trg', help = sgm_t_msg, default =  8)

parser.add_argument('-Vo', required = False, type = float, dest = 'Vo', help = vo_msg, default = -4)
parser.add_argument('-B', '-bias', required = False, type = float, dest = 'bias', help = bias_msg, default = -4)
parser.add_argument('-rst', '-reset', required = False, type = float, dest = 'reset', help = rst_msg, default = -20)

parser.add_argument('-Rlr', '-rec_lr', required = False, type = float, dest = 'rec_lr', help = rlr_msg, default = None)
parser.add_argument('-Olr', '-out_lr', required = False, type = float, dest = 'out_lr', help = olr_msg, default = 0.0025)


parser.add_argument('-savepath', required = False, type = str, dest = 'savepath', help = save_msg, default = None)
parser.add_argument('-F', '-feedback', required = False, type = str, dest = 'feedback', choices = ['spk', 'out'], help = fbck_msg, default = 'spk')

parser.add_argument('--gpu', action = 'store_true', required = False, dest = 'gpu', help = gpu_msg, default = False)
parser.add_argument('--resample', action = 'store_true', required = False, dest = 'resample', help = res_msg, default = False)
parser.add_argument('--V', '--verbose', action = 'store_true', required = False, dest = 'verbose', help = vrb_msg, default = False)

def main(par : dict):
    # Input sanitization
    par['batch_size'] = 1
    par['rec_lr'] = par['rec_lr'] if par['rec_lr'] else 5 / sqrt(par['n_rec'])

    if par['rank'] and (par['rank'] < par['n_out'] or par['rank'] > par['n_rec']):
        raise ValueError(f'Error matrix rank should be in range [{par["n_out"]}, {par["n_rec"]}], got: {par["rank"]}.')

    if par['feedback'] == 'out' and par['rank'] is None:
        warnings.warn('Cannot use feedback "out" with LTTS training. Falling back to feedback "spk".')
        par['feedback'] = 'spk'

    if par['gpu']:
        if not torch.cuda.is_available():
            warnings.warn('No GPU detected. Falling back to CPU mode.')

    device = 'cuda' if torch.cuda.is_available() and par['gpu'] else 'cpu' 
    par['device'] = torch.device(device)

    par['tau_m'] *= par['dt']
    par['tau_s'] *= par['dt']
    par['tau_o'] *= par['dt']

    if par['verbose']:
        msg = '\t\tModel Recap:\n' + '—' * 50 + '\n'
        for k, v in par.items():
            msg += f'\n{k} : {v}'

        print(msg)

    par['targ'] = build_target(par['n_out'], T = par['t_seq'],     off = 8).to(par['device'])
    par['clck'] = build_clock (par['n_inp'], T = par['t_seq'] - 1, off = 0).to(par['device'])

    # * Start training
    start = default_timer()
    stats = (-1, [-1], [-1])

    try:
        stats = train(par)
    
    finally:
        end = default_timer()
        time = end - start

        # Print summary statistics
        msg = f'\n\nTraining finished after {time:.2f}s.\n' +\
            f'Target readout error (MSE): {stats[0]:.4f}.\n' +\
            f'Model readout error (MSE): {stats[1][-1]:.4f}.\n' +\
            f'Model spike-error (ΔZ): {stats[2][-1]}.\n'

        print(msg)

    if par['savepath']:
        with open(par['savepath'] + '.pkl', 'wb') as f:
            pickle.dump(stats, f)

    return stats

if __name__ == '__main__':
    # Here we parse the arguments
    par = vars(parser.parse_args())

    main(par)

    


