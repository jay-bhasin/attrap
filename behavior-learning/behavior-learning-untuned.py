import numpy as np
import tqdm
import os

## Generate parallel fiber activity
# Total time per trial (s)
T = 0.5
# Simulation time step size (s)
dt = 2e-3
# Array of time steps
t = np.arange(0, int(T/dt))*dt


# Optokinetic stimulus
peak_stim_vel = 10 # deg/s
stim = peak_stim_vel*np.sin(2*np.pi*t)

N_PFs = 1000
pf_spike_rates = np.zeros((N_PFs, len(t)))
# Array of times of peak response of parallel fibers
delays = np.linspace(-0.25, 0.25, N_PFs)
for n in range(N_PFs):
    a = 5 # Peak spike rate of PFs (Hz)
    theta = 9 # Threshold for optokinetic stimulus
    # PF rates are scaled, thresholded versions of the stimulus, which each PF receives a time-shifted version of
    pf_spike_rates[n,:] = np.maximum(peak_stim*np.sin(2*np.pi*(t - delays[n])) - theta,0)
    pf_spike_rates[n,:] *= a/np.max(pf_spike_rates[n,:])

# PF spikes are convolved with an exponential synaptic kernel
current_kernel = np.exp(-t/10e-3)
pc_eye_sensitivity = 1 # deg/s per unit PC activity

def behaviorLearning(trials, kernel, N_PC = 50, detailed=False,
                     w=None, baseline_gain = 0.3, learning=True, w_min = -1):
    """Simulate OKR learning with a given kernel for PF-CF
    spike-timing-dependent plasticity.

    Arguments:
    trials -- Number of trials (cycles) to simulate
    kernel -- STDP kernel/PF-triggered eligibility for CF-driven associative LTD
    N_PC -- Number of Purkinje cells to simulate (default: 50)
    detailed -- If True, returns individual PC firing rates and CF spike times (default: False)
    w -- Initial weight vector/matrix; If None, defaults to all zeros. (default: None)
    baseline_gain -- Baseline gain value of non-cerebellar contribution to OKR response (default: 0.3)
    learning -- If True, weights will change according to provided plasticity rule after each trial (default: True)
    w_min -- Minimum value for each PF-PC weight (default: -1)

    Returns:
    w -- Final weight vector/matrix (rows correspond to each simulated PC)
    pc_averaged -- Average activity across all PCs for each trial
    errors -- Retinal slip errors on each trial
    sses -- Sum of squared errors on each trial
    """
    if w is None:
        w = np.zeros((N_PC, pf_spike_rates.shape[0]))
    t_delay = int(0.12/dt) # Delay in CF feedback

    # PC activity on each trial
    pc_total = np.zeros((N_PC, trials+1, len(t)))
    cf_total = np.zeros((trials+1, len(t)+t_delay))

    pc_averaged = np.zeros((trials+1, len(t)))
    errors = np.zeros((trials+1, len(t)))
    sses = np.zeros(trials+1)

    for tt in tqdm.trange(trials+1, leave=False):

        pf_spikes = np.zeros((N_PC, pf_spike_rates.shape[0], len(t)))
        pc_current = np.zeros((N_PC, len(t)))
        # Calculate PF spikes as a Poisson process
        for n in range(N_PC):
            for i in range(pf_spike_rates.shape[0]):
                pf_spikes[n, i,:] = np.random.rand(len(t)) < pf_spike_rates[i,:]*dt
                # Convolve spikes with synaptic kernel to generate current
                pc_current[n,:] += np.convolve(pf_spikes[n, i,:], w[n,i]*current_kernel)[:len(t)]
        # Consider PC activity (re baseline) to be linearly related to PC current
        pc_total[:,tt,:] = np.copy(pc_current)
        pc_averaged[tt,:] = np.mean(pc_current, axis=0)
        # Calculate eye velocity
        eye = pc_eye_sensitivity*pc_averaged[tt,:]
        # Calculate retinal slip
        errors[tt,:] = (stim) - (baseline_gain*stim - eye)
        sses[tt] = np.sum((errors[tt,:])**2)

        if learning:
            # On each trial, CF probability is based on the distribution of retinal slip errors
            # experienced on the previous trial
            if(tt > 0):
                pos_errors = np.sort(errors[tt-1,errors[tt-1]>0])
                if len(pos_errors)>0:
                    # Calculate CF prob using CDF of previous trial errors (see text)
                    cdf = lambda x: np.interp(x, pos_errors, np.linspace(0, 1, len(pos_errors)))
                    error_prob = cdf(np.maximum(errors[tt,:], 0))**3
                    # Normalize error probability so that 1 sp per trial on average
                    cf = np.random.rand(len(t)) < error_prob/np.sum(error_prob)*t[-1]
                    # Delay CF errors by 120 ms
                    cf_total[tt,:] = np.concatenate((np.zeros(t_delay), cf))
                    # Calculate change in weights by correlating STDP kernels triggered by PF spikes
                    # with CF spike times
                    for n in range(N_PC):
                        if np.sum(cf_total[tt,:]) > 0:
                            for i in range(len(w[n,:])):
                                if np.sum(pf_spikes[n, i,:]) > 0:
                                    eligibility = np.convolve(pf_spikes[n, i,:], kernel)[:len(t)+t_delay]
                                    dw = np.dot(eligibility, cf_total[tt,:])
                                    if w[n,i] > w_min:
                                        w[n,i] -= dw
    if detailed:
        return w, pc_averaged[1:], errors[1:], sses[1:], pc_total[:, 1:,:], cf_total[:, 1:, :]
    else:
        return w, pc_averaged[1:], errors[1:], sses[1:]

def __main__():
    t_states = np.arange(0, int(0.2/dt))*dt
    std_states = 0.0075
    N_sims = 10
    N_iterations = 3600

    stdp_kernel = 1e-2*np.exp(-(t_states)**2/(2*std_states**2))

    if not os.path.isdir(os.getcwd()+'/results'):
        os.makedirs(os.getcwd()+'/results')
    if not os.path.isdir(os.getcwd()+'/results/untuned'):
        os.makedirs(os.getcwd()+'/results/untuned')

    for n in tqdm.trange(N_sims):
        w, pc_averaged, errors, sses = behaviorLearning(N_iterations, stdp_kernel, baseline_gain=0.41)

        np.save(os.getcwd() + '/results/untuned/%d_pc_untuned.npy'%n, pc_averaged)
        np.save(os.getcwd() +'/results/untuned/%d_sse_untuned.npy'%n, sses)
        np.save(os.getcwd() +'/results/untuned/%d_w_untuned.npy'%n, w)


if __name__ == "__main__":
    __main__()
