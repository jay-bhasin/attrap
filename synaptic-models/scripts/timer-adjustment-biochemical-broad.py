import numpy as np
import os
import argparse

import sys
sys.path.insert(0, '../utils')

import tqdm

import te_mp

parser = argparse.ArgumentParser()
parser.add_argument("directory", help="directory to store output files")
parser.add_argument("iters", help="number of iterations to run",type=int)
# parser.add_argument("weight", help="weight of peaked distribution (between 0 and 1)", type=float)
parser.add_argument("-i", "--itr_report", help="record output every n iterations", type=int, default=1)
parser.add_argument("-r", "--rate_init", help="Initial value of rate parameter", type=float,default=0.01)
parser.add_argument("-s", "--sims", help="number of simulations to run",type=int, default=1)

def bimodal_distr(t):
    distr = np.exp(-(t-0.05)**2/(2*0.05**2)) + 2*np.exp(-(t-0.14)**2/(2*0.01**2))
    distr /=np.sum(distr)
    return distr

def __main__():
    args = parser.parse_args()
    print(args.directory, args.iters, args.itr_report)

    ## Time vector
    dt_mp = 1e-3
    T_mp_max = 0.2

    tau = np.arange(0, int(T_mp_max/dt_mp))*dt_mp

    drate_scale = 1e-3

    scale_rates_discretized = np.arange(0.01, 0.9, 0.005)
    single_timer_adjustment_discretization = np.load('../utils/timer_adj_discretization.npy')

    ## Setup distribution
    # Use example bimodal distribution (see text)
    distr = bimodal_distr(tau)

    cdf = np.cumsum(distr)
    get_interval = lambda r: np.interp(r, cdf, tau)

    if not os.path.isdir(args.directory):
        os.makedirs(args.directory)

    leave_trange = False if args.sims > 1 else True
    # Run simulation
    peak_times = np.zeros((args.sims, int(np.ceil(args.iters/args.itr_report))))
    scale_rates = np.zeros((args.sims, int(np.ceil(args.iters/args.itr_report))))
    rng_seed = 0
    for i in tqdm.trange(args.sims):
        peak_times[i,:], scale_rates[i,:] = te_mp.simualate_singleTimerBiochemicalDiscretized(single_timer_adjustment_discretization, scale_rates_discretized,
                args.iters, get_interval, False, itr_report = args.itr_report, scale_rate_linear_initial = args.rate_init, drate_scale = drate_scale,
                leave_trange=leave_trange, rng_seed=rng_seed)
        rng_seed += 42

    np.save(args.directory+"/times.npy", peak_times)
    np.save(args.directory+"/rates.npy", scale_rates)

if __name__ == "__main__":
    __main__()
