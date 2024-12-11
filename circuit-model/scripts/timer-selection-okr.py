import numpy as np
import os
import argparse

import sys
sys.path.insert(0, '../utils')
import te_mp

import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("directory", help="directory to store output files",type=str)
parser.add_argument("sims", help="number of simulations to run",type=int)
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
#
# Here we cube the RS so that spikes are more likely to occur for 
# larger retinal slips than linear
cf_prob_fun = lambda retinal_slip: np.maximum(-retinal_slip, 0)**3

okr_model_params = {
    'max_pf_rate': 5,
    'pf_theta': 9,
    'cf_delay_fun': cf_delay_fun,
    'exc_current_kernel':np.exp(-(t_trial - t_trial[0])/10e-3),
}

# We chose the learning rates to be smaller for the untuned (dark) case,
# to better approximate the experiments, in which this was the first 
# learning experience for animals
plasticity_params_untuned = {
    'dw_ltd': 1.25e-2,
    'dw_ltp': 1.25e-2*0.016,
    'decay_rate': 1/(3600)
}
plasticity_params_tuned = {
    'dw_ltd': 2.5e-2,
    'dw_ltp': 2.5e-2*0.016,
    'decay_rate': 1/(3600)
}

# Shape of eligibility window
sigma2 = 0.01 # s
eligibility_window = lambda t, center: np.exp(-((t-center)/sigma2)**2)

eligibility_params_untuned = {
    'tau_peaks': 0,
    'timer': eligibility_window,
}
eligibility_params_tuned = {
    'tau_peaks': 0.12,
    'timer': eligibility_window,
}

## Visual (optokinetic ) stimulus

peak_stim_vel = 10 # deg/s
f = 1 # Hz
stim_vel_f = lambda t: -peak_stim_vel*np.sin(2*np.pi*f*t)
direct_pathway_f = lambda t: -peak_stim_vel*np.sin(2*np.pi*f*(t-0.04))

## Retinal slip
def retinal_slip_f(t, pc_eye, baseline_gain = 0.35):
    eye = pc_eye + baseline_gain*direct_pathway_f(t)
    return stim_vel_f(t) - eye

def __main__():
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        os.makedirs(args.directory)
    
    retinal_slip_f_= lambda t,e: retinal_slip_f(t, e, baseline_gain=0.35)

    # Results in the paper were generated from 10 simulations of 3600 trials/cycles each
    okr_learning_result_untuned = te_mp.simulateLearningOKR(args.sims, args.iters, t_trial, retinal_slip_f_, cf_prob_fun,
                                                       okr_model_params, plasticity_params_untuned, 'single', tau,
                                                       eligibility_params_untuned, history_samples = args.itr_report, 
                                                       rng_seed = 0, peak_stim_vel=peak_stim_vel, stim_f=stim_f)

    np.save(args.directory+'/eye_history_untuned.npy',okr_learning_result_untuned['eye_history'])
    np.save(args.directory+'/eye_history_no_noise_untuned.npy',okr_learning_result_untuned['eye_history_no_noise'])
    np.save(args.directory+'/w_final_untuned.npy',okr_learning_result_untuned['w_final'] )
    np.save(args.directory+'/w_avg_final_untuned.npy',okr_learning_result_untuned['w_avg_final'] )

    okr_learning_result_tuned = te_mp.simulateLearningOKR(args.sims, args.iters, t_trial, retinal_slip_f_, cf_prob_fun,
                                                       okr_model_params, plasticity_params_tuned, 'single', tau,
                                                       eligibility_params_tuned, history_samples = args.itr_report,
                                                       rng_seed = 42, peak_stim_vel=peak_stim_vel, stim_f=stim_f)

    np.save(args.directory+'/eye_history_tuned.npy',okr_learning_result_tuned['eye_history'])
    np.save(args.directory+'/eye_history_no_noise_tuned.npy',okr_learning_result_tuned['eye_history_no_noise'])
    np.save(args.directory+'/w_final_tuned.npy',okr_learning_result_tuned['w_final'] )
    np.save(args.directory+'/w_avg_final_tuned.npy',okr_learning_result_tuned['w_avg_final'] )


if __name__ == "__main__":
    __main__()