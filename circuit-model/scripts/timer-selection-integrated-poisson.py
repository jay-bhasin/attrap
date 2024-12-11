import numpy as np
import os
import argparse

import sys
sys.path.insert(0, '../../utils')
import te_mp

import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("directory", help="directory to store output files",type=str)
parser.add_argument("sims", help="number of simulations to run",type=int)
parser.add_argument("-init", "--initial", help="location of initial values file", type=str)
parser.add_argument("iters", help="number of iterations per simulation",type=int)
parser.add_argument("-i","--itr_report", help="number of iterations to save per trial",type=int, default=10)


# Simulation time step size (s)
dt_trial = 2e-3

# Total time per block (s)
T_max_full = 0.75
T_min_full = -0.25
T = T_max_full - T_min_full

T_min = 0
t_min_ind = int((T_min-T_min_full)/dt_trial)
T_max = 0.5
t_max_ind = int((T_max - T_min_full)/dt_trial)


# Time vector for each trial (here, 1 s block)
t_trial = np.arange(0, int(T/dt_trial))*dt_trial + T_min_full

# Time vector for eligibility window (here, 200 ms)
dt_mp = dt_trial
T_mp_max = 0.2
tau = np.arange(0, int(T_mp_max/dt_mp))*dt_mp

# Distribution for CF delays is peaked at 120 ms with 5 ms s.d.
def cf_delay_fun(t_trial):
    delay_distr = np.exp(-(t_trial-0.12)**2/0.005**2)
    delay_distr /= np.sum(delay_distr)
    return delay_distr

# CF probability is calculated by normalizing the contraversive retinal slip
# across the 1 s trial and then drawing a spike time
cf_prob_fun = lambda retinal_slip: np.maximum(-retinal_slip, 0)

poisson_model_params = {
    'max_pf_rate': 0.477, # equivalent to average PF spike rake for the basis used during OKR adaptation 
    'cf_delay_fun': cf_delay_fun,
    'exc_current_kernel':np.exp(-(t_trial - t_trial[0])/10e-3),
    'type':'poisson'
}

# These parameters were chosen so that over the course of a simulation, weights would equilibrate near baseline
plasticity_params_untuned = {
    'dw_ltd': 3e-2,
    'dw_ltp': 3e-2*0.023, #  > area under the curve of learning rule 
    'decay_rate': 1/(1000)
}

## Generate timer bank of Gaussians
N_timers = 11
timers = np.zeros((N_timers, len(tau)))
centers = np.linspace(0, 0.2, N_timers+1)[:-1]
width = centers[1]/2
centers += width
for i in range(N_timers):
    timers[i,:] = np.exp(-(tau - centers[i])**2/(2*(width)**2))

# Initial condition
v_0 = np.zeros(N_timers)
v_0[0] = 1

eligibility_params_sel = {
    'timers': timers,
    'v_0': v_0,
    'dv_max':1e-4,
}

def __main__():
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        os.makedirs(args.directory)

    history_samples = np.minimum(args.itr_report, args.iters)
    results_adj = te_mp.simulateLearningAdjustment_TeMP(args.sims, args.iters, t_trial, cf_prob_fun, 
                                   poisson_model_params, plasticity_params_untuned, 'multiple', tau, 
                                   eligibility_params_sel, history_samples = history_samples, 
                                    calculate_hist = True, hist_every = 0,
                                    PF_samples = [0, 60, 119])

    # In the paper, we used 10 simulation blocks of 1,000,000 trials each
    # and saved results every 1,000 trials

    np.save(args.directory+'/w_mean.npy',results_adj['w_mean'])
    np.save(args.directory+'/w_sample.npy',results_adj['w_sample'])
    np.save(args.directory+'/hist_history.npy',results_adj['hist_history_PF'])
    np.save(args.directory+'/eye_history.npy',results_adj['eye_history'])
    np.save(args.directory+'/v.npy',results_adj['v'])
    np.save(args.directory+'/eligibility_window_history.npy', results_adj['temp_history'])
    np.save(args.directory+'/eligibility_window_mean.npy', results_adj['temp_history_mean'])
    np.save(args.directory+'/cf_prob_avg.npy', results_adj['cf_prob_avg'])

if __name__ == "__main__":
    __main__()