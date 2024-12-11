import numpy as np
import tqdm
import tqdm.notebook

import scipy.integrate

### Integrated Simulation
# Simulation parameters
w_max = 5
w_min = 0
w_mli = w_max/2
avg_cf_rate = 1

# Compute a periodic convolution
# Edited from: https://stackoverflow.com/questions/35474078/python-1d-array-circular-convolution
def conv_circ( signal, ker ):
    '''
        signal: real 1D array
        ker: real 1D array
        signal and ker must have same shape
    '''
    if len(ker) < len(signal):
        ker = np.pad(ker, (0, len(signal)-len(ker)),'constant',constant_values = (0))
    return np.real(np.fft.ifft( np.fft.fft(signal)*np.fft.fft(ker) ))

# Generating temporal PF basis for OKR learning
def generatePFratesBasisOKR(t, N_PFs = 120, a=5, theta=9, peak_stim_vel = 10, stim_f=1):
    # N_PFs: Number of PFs
    # a: peak spike rate (Hz)
    # theta: threshold for visual stim velocity

    pf_spike_rates = np.zeros((N_PFs, len(t)))
    # Array of times of peak response of parallel fibers
    delays = np.linspace(0, t[-1]-t[0], N_PFs+1) + t[0]

    for n in range(N_PFs):
        # PF rates are scaled, thresholded versions of the stimulus, 
        # of which each PF receives a time-shifted version
        pf_spike_rates[n,:] = np.maximum(peak_stim_vel*np.cos(2*np.pi*stim_f*(t - delays[n])) - theta,0)
        pf_spike_rates[n,:] *= a/np.max(pf_spike_rates[n,:])
    return pf_spike_rates

# Idealized molecular timer
def idealMolecularTimer(tau, center, sigma2=0.015):
    return np.exp(-((tau-center)/sigma2)**2)

# Generate idealized timer elements for multiple timer selection rule
def multipleTimersGaussian(tau, N_timers, tau_max = 0.2, sigma2=0.015):
    # tau_max = tau[-1] + tau[1]
    timer_centers = np.linspace(0,tau_max,N_timers)
    timers = np.vstack([idealMolecularTimer(tau,timer_centers[i]) for i in range(len(timer_centers))])
    # shape: N_timers x length(tau)
    return timers

# Simulate OKR learning without temporal metaplasticity
# (assuming the idealized molecular timer models)
def simulateLearningOKR(N_simulations, N_trials, t_trial, retinal_slip_fun, cf_prob_fun, model_params,
                    plasticity_params, plasticity_model, tau, eligibility_params,
                    w_0 = None, w_0_avg = None,
                    N_PFs = 120, avg_sensitivity_to_pc = 0.01, pc_eye_sensitivity=5, rng_seed = 0,
                    peak_stim_vel = 10, stim_f = 1,
                    history_samples = 0, calculate_hist = False, PF_samples = [], N_hist_bins=100, notebook_mode = False):
    # N_simulations: number of simulations to perform
    # N_trials: number of cycles (trials) of OKR stimulus to present
    # t_trial: time vector during stimulus cycle
    # retinal_slip_fun: callable function that returns retinal slip magnitude given the PC output
    # cf_prob_fun: callable function that transform retinal slip into CF probability
    # model_params: dictionary of model parameters containing
    #   - 'max_pf_rate': `a` argument for PF basis generator
    #   - 'pf_theta': `theta`` argument for PF basis generator
    #   - 'inh_current_kernel': spike kernel for PFs
    #   - 'exc_current_kernel': spike kernel for interneuron
    #   - 'cf_delay_fun': callable function that returns the CF prob density as function of t_trial
    # plasticity_params: dictionary of plasticity parameters containing:
    #   - 'dw_ltd': max amplitude of LTD
    #   - 'dw_ltp': amplitude of PF-only LTP
    #   - 'decay_rate': rate at which PF weight decays to baseline
    # plasticity_model: 'single' or 'multiple' timers
    # tau: time vector over which to evaluate 
    # eligibility_params: dictionary of parmaeters that for 'single' timer contains
    #       - 'tau_peaks': the time of peak eligibility, either a scalar or a vector with entries for each PF
    #       - 'timer': callable function that takes a time vector (tau) and the center for the peak 
    #   and for 'multiple' timers contains
    #       - 'v_0': coupling strengths (either a single vector or a matrix with rows for each PF)
    # w_0: initial value of parallel fiber weights (which has size = N_PFs; for spiking cell)
    # w_0_avg: initial value of parallel fiber weights (population)
    # N_PFs: number of parallel fibers to simulate
    # avg_sensitivity_to_pc: weight of effect of spike-based PC model on calculating eye movement output
    # rng_seed: seed for random number generator
    # history_samples: how many weight iterations to save (0 = every iteration)
    # calculate_hist: True to calculate spike interval histogram (over time range specified by tau)
    # PF_samples: if calculate_hist is True, specify which PFs to sample in addition to combined histogram; empty -> calculate only combined histogram
    #   if not empty, also store time course of weights of specified PFs for spiking cell
    # N_hist_bins: number of histogram bins
    # notebook_model: True to use jupyter tqdm widget

    # Generate eligibility window for every PF
    if plasticity_model == 'single':
        tau_peaks = eligibility_params['tau_peaks']
        if np.shape(tau_peaks) == ():
            eligibility_window = eligibility_params['timer'](tau, tau_peaks)
            eligibility_windows = np.tile(eligibility_window, (N_PFs, 1))
        elif np.shape(tau_peaks) == (N_PFs,):
            eligibility_windows = np.vstack([eligibility_params['eligibility_window_f'](tau, t_p) for t_p in tau_peaks])
        else:
            raise ValueError("Incompatible parameters for eligibility window.")
    elif plasticity_model == 'multiple':
        v_0 = eligibility_params['v_0']
        N_timers = np.shape(v_0)[-1]
        timers = multipleTimersGaussian(tau,  N_timers)
        # if v_0 is a vector of length N_timers then one eligibility window for all timers
        if np.shape(v_0) == (N_timers,):
            eligibility_window = eligibility_params['v_0']@timers 
            eligibility_windows = np.tile(eligibility_window, (N_PFs, 1))
        elif np.shape(v_0) == (N_PFs, N_timers):
            eligibility_windows = eligibility_params['v_0']@timers
        else:
            raise ValueError("Incompatible parameters for eligibility window.")
        
        state_counts = np.zeros((N_simulations, len(PF_samples)+1, N_timers))
    else:
        raise ValueError("Invalid value for 'plasticity_model'.")

    # Setup how frequently to store results
    if history_samples == 0: # if history_samples is 0, save every iteration
        history_samples = N_trials
    sample_interval = int(np.floor(N_trials/history_samples))

    dt = t_trial[1] - t_trial[0]
    T_min_full = t_trial[0]
    T_max_full = t_trial[-1] + dt

    ## Setup outputs
    # Learned component of eye movement - PC population + effect of spiking cell
    eye_history = np.zeros((N_simulations, len(t_trial), history_samples))
    # Learned component of eye movement - PC population + effect of spiking cell
    eye_history_no_noise = np.zeros((N_simulations, len(t_trial), history_samples))

    # Histogram of PF-CF spike-timing intervals
    hist_history_PF = np.zeros((N_simulations, len(PF_samples) + 1, N_hist_bins,history_samples))
    # edges of histogram for "digitize" function
    hist_edges = np.linspace(0, t_trial[-1] + (t_trial[1]-t_trial[0]), N_hist_bins+1)

    
    
    # Final weights of spiking cell
    w_final = np.zeros((N_simulations, N_PFs))
    # Final weights of cell population
    w_avg_final = np.zeros((N_simulations, N_PFs))

    # Examples - empirical CF probability at each time in trial (cumulative)
    cf_prob_avg = np.zeros((N_simulations, len(t_trial), history_samples))

    # Store time course of PFs specified by PF_samples
    if len(PF_samples) > 0:
        w_sample = np.zeros((N_simulations, len(PF_samples), history_samples))

    if notebook_mode:
        trange = tqdm.notebook.trange
    else:
        trange = tqdm.trange

    # spike kernels:
    exc_current_kernel = model_params['exc_current_kernel']
    # inh_current_kernel = model_params['inh_current_kernel']

    # Plasticity rates
    dw_ltp = plasticity_params['dw_ltp']
    dw_ltd = plasticity_params['dw_ltd']

    # Initialize random number generator
    rng = np.random.default_rng(seed=rng_seed)

    # Generate PF spike rates
    pf_spike_rates = generatePFratesBasisOKR(t_trial, N_PFs=N_PFs, a = model_params['max_pf_rate'], theta=model_params['pf_theta'], 
                                                peak_stim_vel=peak_stim_vel, stim_f = stim_f)

    # Inhibitory interneuron current sums up PF inputs with same weight (defines the baseline)
    mli_current = dt*conv_circ(np.ones(N_PFs)@pf_spike_rates, exc_current_kernel)

    # CF delay distribution
    delay_distr = model_params['cf_delay_fun'](t_trial-t_trial[0])

    for sim in trange(N_simulations):
        # Reset weights
        if w_0 == None:
            w = np.ones(N_PFs)*w_max/2 # spiking cell
        else:
            w = np.copy(w_0)
        if w_0_avg == None:
            w_avg = np.ones(N_PFs)*w_max/2 # population
        else:
            w_avg = np.copy(w_0_avg)
        n_to_divide = 0 # denominator for calculating CF prob
        sample_count = 1 # reset iteration counter for saving results
        
        # histogram for this simulation
        hist_history_sim = np.zeros((len(PF_samples)+1, N_hist_bins))
        # reset average climbing fiber prob
        cf_prob_sim = np.zeros(len(t_trial))
        
        for tt in trange(N_trials, leave=False):  

            # Generate PF spikes for sample PC
            pf_spikes = rng.random(size=pf_spike_rates.shape) < pf_spike_rates*dt
            pf_spikes_weighted = w@pf_spikes
            pc_current_exc = dt*conv_circ(pf_spikes_weighted/dt, exc_current_kernel)

            ## Generate PC firing rate for rest of population
            pc_current_exc_avg_bef = w_avg@pf_spike_rates
            # convolve with spike kernel filter
            # pc_current_exc_avg_no_noise = dt*conv_circ(pc_current_exc_avg_bef, exc_current_kernel)
            pc_current_exc_avg = dt*conv_circ(pc_current_exc_avg_bef, exc_current_kernel)

            # Subtract MLI current
            # mli_current_same = dt*conv_circ(np.sum(pf_spikes/dt, axis=0), inh_current_kernel)

            pc_current = pc_current_exc - w_mli*mli_current
            pc_current_avg = pc_current_exc_avg - w_mli*mli_current
            # pc_current_avg_no_noise = pc_current_exc_avg_no_noise - w_mli*mli_current

            # learned component of eye movement
            eye_out = pc_eye_sensitivity*(pc_current_avg*(1-avg_sensitivity_to_pc) + pc_current*avg_sensitivity_to_pc)

            # calculate retinal slip
            retinal_slip = retinal_slip_fun(t_trial, eye_out)

            # Calculate CF probability (here, contraversive RS)
            cf_prob = cf_prob_fun(retinal_slip)

            # Rescale so that the average rate over the block is ~1 Hz
            if np.sum(cf_prob) > 0:
                cf_prob /= np.sum(cf_prob)
                cf_prob *= avg_cf_rate*(t_trial[-1] - t_trial[0])
            
            # Delay the CF probability distribution
            cf_prob_shifted = conv_circ(cf_prob, delay_distr)

            # Save history
            if tt > 0 and tt % sample_interval == 0:
                eye_history[sim,:,sample_count] = pc_eye_sensitivity*pc_current
                eye_history_no_noise[sim, :,sample_count] = pc_eye_sensitivity*pc_current_avg
                
                hist_history_PF[sim, :, :, sample_count] += hist_history_sim
                cf_prob_avg[sim, :, sample_count] += cf_prob_sim/n_to_divide
                # w_mean[sim, sample_count] = np.mean(w)
                if len(PF_samples) > 0:
                    w_sample[sim, :, sample_count] = np.copy(w[PF_samples])
                sample_count += 1
                
            ## Choose CF spikes
            if np.sum(cf_prob) > 0:
                cf_prob_sim += cf_prob_shifted
                n_to_divide += 1
                cf_prob_cdf = np.cumsum(cf_prob_shifted)
                cf_spike_time = np.interp(rng.random(), cf_prob_cdf, t_trial)

            ## Do plasticity
            for j in range(N_PFs):
                eligibility_window = eligibility_windows[j,:]
                
                ## Firing rates
                pf_rate = pf_spike_rates[j,:]
                if np.sum(cf_prob_shifted) > 0:
                    pf_eligibility = dt*conv_circ(pf_rate,eligibility_window)
                    # LTD for population ~ cross-correlation of PF eligibility trace with CF probability
                    dw_avg_d = np.sum(pf_eligibility*cf_prob_shifted)
                else:
                    dw_avg_d = 0
                dw_avg_p = np.sum(pf_rate*dt) # PF-only LTP for population ~ integral of PF activity

                # Update weights for population
                w_avg[j] += -dw_ltd*dw_avg_d + dw_ltp*dw_avg_p

                # Update weights for spiking cell
                for s in np.where(pf_spikes[j,:])[0]:
                    w[j] += dw_ltp # LTP for every PF

                    if np.sum(cf_prob) > 0:
                        spike_time_1 = cf_spike_time - t_trial[s]
                        # use circular assumption for simplicity
                        spike_time_2 = spike_time_1 + T_max_full - T_min_full

                        if 0 <= spike_time_1 <= 0.2 or 0<= spike_time_2 <= 0.2:
                            spike_time = spike_time_1 if 0<=spike_time_1<=0.2 else spike_time_2
                            cf_spike_ind = np.digitize(spike_time, tau)-1

                            # Histogram
                            if calculate_hist:
                                hist_bin = np.digitize(spike_time, hist_edges)-1
                                hist_history_sim[0, hist_bin] += 1
                                for p in range(len(PF_samples)):
                                    if j == PF_samples[p]:
                                        hist_history_sim[p+1, hist_bin] += 1
                            
                            ## Plasticity
                            dw = eligibility_window[cf_spike_ind]
                            w[j] -= dw*dw_ltd # spike-timing-dependent LTD

                # Keep weights within limits
                if w_avg[j] < w_min: w_avg[j] = w_min
                elif w_avg[j] > w_max: w_avg[j] = w_max
                w_avg[j] += plasticity_params['decay_rate']*(w_max/2 - w_avg[j])

                if w[j] < w_min: w[j] = w_min
                elif w[j] > w_max: w[j] = w_max
                w[j] += plasticity_params['decay_rate']*(w_max/2 - w[j])

        w_final[sim,:] = np.copy(w)
        w_avg_final[sim,:] = np.copy(w_avg)

    return_dict = {
        'w_final': w_final,
        'w_avg_final': w_avg_final,
        'eye_history': eye_history,
        'eye_history_no_noise': eye_history_no_noise,
        'cf_prob_avg': cf_prob_avg,
    }
    if calculate_hist:
        return_dict['hist_history_PF'] = hist_history_PF
    if len(PF_samples) > 0:
        return_dict['w_sample'] = w_sample
        
    return return_dict

### Simulate integrated metaplasticity model
def simulateLearningAdjustment_TeMP(N_simulations, N_trials, t_trial, cf_prob_fun, model_params,
                    plasticity_params, plasticity_model, tau, eligibility_params, metaplasticity=True,
                    N_PFs = 120, avg_sensitivity_to_pc = 0.01, pc_eye_sensitivity = 0.1, rng_seed = 0,
                    history_samples = 0, calculate_hist = False, hist_every = 0, PF_samples = [], N_hist_bins=100, notebook_mode = False):
    # N_simulations: number of simulations to perform
    # N_trials: number of cycles (trials) of OKR stimulus to present
    # t_trial: time vector during stimulus cycle
    # retinal_slip_fun: callable function that returns retinal slip magnitude given the PC output
    # cf_prob_fun: callable function that transform retinal slip into CF probability
    # model_params: dictionary of model parameters containing
    #   - 'max_pf_rate': `a` argument for PF basis generator
    #   - 'pf_theta': `theta`` argument for PF basis generator
    #   - 'inh_current_kernel': spike kernel for PFs
    #   - 'exc_current_kernel': spike kernel for interneuron
    #   - 'cf_delay_fun': callable function that returns the CF prob density as function of t_trial
    # plasticity_params: dictionary of plasticity parameters containing:
    #   - 'dw_ltd': max amplitude of LTD
    #   - 'dw_ltp': amplitude of PF-only LTP
    #   - 'decay_rate': rate at which PF weight decays to baseline
    # plasticity_model: 'single' or 'multiple' timers
    # tau: time vector over which to evaluate 
    # eligibility_params: dictionary of parmaeters that for 'single' timer contains
    #       - 'tau_peaks': the time of peak eligibility, either a scalar or a vector with entries for each PF
    #       - 'timer': callable function that takes a time vector (tau) and the center for the peak 
    #       - 'temp_rule: callable function that takes a time vector (tau) and the current peak (tau_p) 
    #           and returns change in tau_p
    #   and for 'multiple' timers contains
    #       - 'v_0': coupling strengths (either a single vector or a matrix with rows for each PF)
    # w_0: initial value of parallel fiber weights (which has size = N_PFs; for spiking cell)
    # w_0_avg: initial value of parallel fiber weights (population)
    # N_PFs: number of parallel fibers to simulate
    # avg_sensitivity_to_pc: weight of effect of spike-based PC model on calculating eye movement output
    # rng_seed: seed for random number generator
    # history_samples: how many weight iterations to save (0 = every iteration)
    # calculate_hist: True to calculate spike interval histogram (over time range specified by tau)
    # PF_samples: if calculate_hist is True, specify which PFs to sample in addition to combined histogram; empty -> calculate only combined histogram
    #   if not empty, also store time course of weights of specified PFs for spiking cell
    # N_hist_bins: number of histogram bins
    # notebook_model: True to use jupyter tqdm widget

    # Setup how frequently to store results
    if history_samples == 0: # if history_samples is 0, save every iteration
        history_samples = N_trials
    sample_interval = int(np.floor(N_trials/history_samples))

    dt = t_trial[1] - t_trial[0]
    T_min_full = t_trial[0]
    T_max_full = t_trial[-1] + dt
    # tau = t_trial
    if plasticity_model == 'single':
        tau_peaks = eligibility_params['tau_peaks']
        if np.shape(tau_peaks) == ():
            eligibility_window = eligibility_params['timer'](tau, tau_peaks)
            eligibility_windows = np.tile(eligibility_window, (N_PFs, 1))
            tau_peaks = np.ones(N_PFs)*tau_peaks
        elif np.shape(tau_peaks) == (N_PFs,):
            eligibility_windows = np.vstack([eligibility_params['eligibility_window_f'](tau, t_p) for t_p in tau_peaks])
        else:
            raise ValueError("Incompatible parameters for eligibility window.")
        
        temp_rule = eligibility_params['temp_rule']
        if len(PF_samples) > 0:
            temp_history = np.zeros((N_simulations, len(PF_samples), history_samples + 1))
        temp_history_mean = np.zeros((N_simulations,2, history_samples+1))
    elif plasticity_model == 'multiple':
        # raise NotImplementedError("Not yet implemented.")
        v_0 = eligibility_params['v_0']

        timers = eligibility_params['timers']
        N_timers = timers.shape[0]

        # if v_0 is a vector of length N_timers then one eligibility window for all timers
        if np.shape(v_0) == (N_timers,):
            eligibility_window = eligibility_params['v_0']@timers 
            eligibility_windows = np.tile(eligibility_window, (N_PFs, 1))
            v = np.tile(v_0, (N_PFs, 1))
        elif np.shape(v_0) == (N_PFs, N_timers):
            eligibility_windows = eligibility_params['v_0']@timers
            v = np.copy(v_0)
        else:
            raise ValueError("Incompatible parameters for eligibility window.")
        # calculate average eligibility window across population
        temp_history_mean = np.zeros((N_simulations, len(tau), history_samples + 1))
        if len(PF_samples) > 0:
            # store coupling weights for specified sample PFs
            temp_history = np.zeros((N_simulations, len(PF_samples), N_timers, history_samples + 1))
        dv = eligibility_params['dv_max']*np.ones(N_timers)
        # dv[0] =eligibility_params['dv_zero'] # to implement biased selection

        # state_counts = np.zeros((N_simulations, len(PF_samples)+1, N_timers)) # count how many times each timer was most active
    else:
        raise ValueError("Invalid value for 'plasticity_model'.")

    ## Setup outputs
    # Learned component of eye movement - PC population + effect of spiking cell
    eye_history = np.zeros((N_simulations, len(t_trial), history_samples + 1))
    # Learned component of eye movement - PC population + effect of spiking cell
    # eye_history_no_noise = np.zeros((N_simulations, len(t_trial), history_samples))

    # Histogram of PF-CF spike-timing intervals
    # if hist_every == 0: report only at end of all simulations
    if hist_every == 2: # report for every sample break
        hist_history_PF = np.zeros((N_simulations, len(PF_samples) + 1, N_hist_bins,history_samples + 1))
    elif hist_every == 1: # report at end of every simulation
        hist_history_PF = np.zeros((N_simulations, len(PF_samples)+1, N_hist_bins))
    elif hist_every == 0: # report at end of all simulations
        hist_history_PF = np.zeros((len(PF_samples) +1 , N_hist_bins))
    # edges of histogram for "digitize" function
    hist_edges = np.linspace(0, tau[-1] + (tau[1]-tau[0]), N_hist_bins+1)

    
    
    # Final weights of spiking cell
    w_final = np.zeros((N_simulations, N_PFs))

    # Examples - empirical CF probability at each time in trial (cumulative)
    cf_prob_avg = np.zeros((N_simulations, len(t_trial), history_samples+ 1))

    # Store time course of PFs specified by PF_samples
    if len(PF_samples) > 0:
        w_sample = np.zeros((N_simulations, len(PF_samples), history_samples + 1))
    w_mean = np.zeros((N_simulations, history_samples + 1))

    if notebook_mode:
        trange = tqdm.notebook.trange
    else:
        trange = tqdm.trange

    # spike kernels:
    exc_current_kernel = model_params['exc_current_kernel']
    # inh_current_kernel = model_params['inh_current_kernel']

    # Plasticity rates
    dw_ltp = plasticity_params['dw_ltp']
    dw_ltd = plasticity_params['dw_ltd']

    # Initialize random number generator
    rng = np.random.default_rng(seed=rng_seed)

    dt = t_trial[1] - t_trial[0]

    # Generate PF spike rates
    if model_params['type'] == 'poisson':
        pf_spike_rates = model_params['max_pf_rate']*np.ones((N_PFs, len(t_trial)))
    elif model_params['type'] == 'okr':
        pf_spike_rates = generatePFratesBasisOKR(t_trial, N_PFs= N_PFs, a=model_params['max_pf_rate'], theta= model_params['pf_theta'],
                                                peak_stim_vel = model_params['peak_stim_vel'], stim_f = model_params['stim_f'])

    # Inhibitory interneuron current sums up PF inputs with same weight (defines the baseline)
    mli_current = dt*conv_circ(np.ones(N_PFs)@pf_spike_rates, exc_current_kernel)

    # CF delay distribution
    delay_distr = model_params['cf_delay_fun'](t_trial-t_trial[0])

    for sim in trange(N_simulations):
        # Reset weights
        w = np.ones(N_PFs)*w_max/2 # spiking cell
        w_mean[sim,0] = w_max/2
        if len(PF_samples) > 0:
            w_sample[sim,:,0] = w_max/2
            if metaplasticity:
                if plasticity_model == 'single':
                    temp_history[sim, :, 0] = np.copy(tau_peaks[PF_samples])
                elif plasticity_model == 'multiple':
                    temp_history[sim, :, :, 0] = np.copy(v[PF_samples,:])
        if metaplasticity:
            if plasticity_model == 'single':
                temp_history_mean[sim,:,0] = (np.mean(tau_peaks),np.std(tau_peaks))
            elif plasticity_model == 'multiple':
                temp_history_mean[sim,:,0] = np.mean(v@timers, axis=0)
        n_to_divide = 0 # denominator for calculating CF prob
        sample_count = 1 # reset iteration counter for saving results
        
        # histogram for this simulation
        hist_history_sim = np.zeros((len(PF_samples)+1, N_hist_bins))
        # reset average climbing fiber prob
        cf_prob_sim = np.zeros(len(t_trial))
        
        for tt in trange(N_trials, leave=False):  

            # Generate PF spikes for sample PC
            pf_spikes = rng.random(size=pf_spike_rates.shape) < pf_spike_rates*dt
            pf_spikes_weighted = w@pf_spikes
            pc_current_exc = dt*conv_circ(pf_spikes_weighted/dt, exc_current_kernel)

            ## Generate PC firing rate for rest of population
            # pc_current_exc_avg_bef = w_avg@pf_spike_rates
            # convolve with spike kernel filter
            # pc_current_exc_avg_no_noise = dt*conv_circ(pc_current_exc_avg_bef, exc_current_kernel)
            # pc_current_exc_avg = dt*conv_circ(pc_current_exc_avg_bef, exc_current_kernel)

            # Subtract MLI current
            # mli_current_same = dt*conv_circ(np.sum(pf_spikes/dt, axis=0), inh_current_kernel)

            pc_current = pc_current_exc - w_mli*mli_current
            # pc_current_avg = pc_current_exc_avg - w_mli*mli_current
            # pc_current_avg_no_noise = pc_current_exc_avg_no_noise - w_mli*mli_current

            # Generate eye movement
            if model_params['type'] == 'poisson':
                target_rate = model_params['max_pf_rate']
            else:
                target_rate = model_params['poisson_rate_equiv'] # spikes expected across population per PF per bin
            random_target = pc_eye_sensitivity*conv_circ(target_rate*rng.standard_normal(size=(len(t_trial),)), exc_current_kernel)
            
            # learned component of eye movement
            # eye_out = pc_eye_sensitivity*(pc_current_avg*(1-avg_sensitivity_to_pc) + pc_current*avg_sensitivity_to_pc)
            eye_out = pc_eye_sensitivity*pc_current

            # calculate retinal slip
            retinal_slip = (random_target*np.sqrt(avg_sensitivity_to_pc*(1-avg_sensitivity_to_pc))
                            - eye_out*avg_sensitivity_to_pc)

            # Calculate CF probability (here, contraversive RS)
            cf_prob = cf_prob_fun(retinal_slip)

            # Rescale so that the average rate over the block is ~1 Hz
            if np.sum(cf_prob) > 0:
                cf_prob /= np.sum(cf_prob)
                cf_prob *= avg_cf_rate*(t_trial[-1] - t_trial[0])
            
            # Delay the CF probability distribution
            cf_prob_shifted = conv_circ(cf_prob, delay_distr)

            # Save history
            if tt > 0 and tt % sample_interval == 0:
                eye_history[sim,:,sample_count] = pc_eye_sensitivity*pc_current
                # eye_history_no_noise[sim, :,sample_count] = pc_eye_sensitivity*pc_current_avg
                if hist_every == 2:
                    hist_history_PF[sim, :, :, sample_count] += hist_history_sim
                cf_prob_avg[sim, :, sample_count] += cf_prob_sim/n_to_divide
                w_mean[sim, sample_count] = np.mean(w)
                if len(PF_samples) > 0:
                    w_sample[sim, :, sample_count] = np.copy(w[PF_samples])
                if metaplasticity:
                    if plasticity_model == 'single':
                        if len(PF_samples) > 0:
                            temp_history[sim, :, sample_count] = np.copy(tau_peaks[PF_samples])
                        temp_history_mean[sim, :, sample_count] = (np.mean(tau_peaks), np.std(tau_peaks))
                    elif plasticity_model == 'multiple':
                        if len(PF_samples) > 0:
                            temp_history[sim, :, :, sample_count] = np.copy(v[PF_samples, :])
                        temp_history_mean[sim, :, sample_count] = np.mean(v@timers,axis=0)
                sample_count += 1
                
            ## Choose CF spikes
            if np.sum(cf_prob) > 0:
                cf_prob_sim += cf_prob_shifted
                n_to_divide += 1
                cf_prob_cdf = np.cumsum(cf_prob_shifted)
                cf_spike_time = np.interp(rng.random(), cf_prob_cdf, t_trial)

            ## Do plasticity
            for j in range(N_PFs):
                eligibility_window = eligibility_windows[j,:]

                # Update weights for spiking cell
                for s in np.where(pf_spikes[j,:])[0]:
                    w[j] += dw_ltp # LTP for every PF

                    if np.sum(cf_prob) > 0:
                        spike_time_1 = cf_spike_time - t_trial[s]
                        # use circular assumption for simplicity
                        spike_time_2 = spike_time_1 + T_max_full - T_min_full

                        if 0 <= spike_time_1 <= 0.2 or 0<= spike_time_2 <= 0.2:
                            spike_time = spike_time_1 if 0<=spike_time_1<=0.2 else spike_time_2
                            cf_spike_ind = np.digitize(spike_time, tau)-1

                            # Histogram
                            if calculate_hist:
                                hist_bin = np.digitize(spike_time, hist_edges)-1
                                hist_history_sim[0, hist_bin] += 1
                                for p in range(len(PF_samples)):
                                    if j == PF_samples[p]:
                                        hist_history_sim[p+1, hist_bin] += 1
                            
                            ## Plasticity
                            dw = eligibility_window[cf_spike_ind]
                            w[j] -= dw*dw_ltd # spike-timing-dependent LTD

                            ## Metaplasticity
                            if metaplasticity:
                                if plasticity_model == 'single':
                                    delta_tau_p = temp_rule(spike_time, tau_peaks[j])
                                    tau_peaks[j] += delta_tau_p
                                    if tau_peaks[j] > 0.2: tau_peaks[j] = 0.2
                                    elif tau_peaks[j] < 0: tau_peaks[j] = 0
                                    eligibility_window = eligibility_params['timer'](tau, tau_peaks[j])
                                elif plasticity_model == 'multiple':
                                    timer_vals = timers[:,cf_spike_ind]
                                    # Get index of most active timer
                                    active_timer = np.argmax(timer_vals)
                                    # Shouldn't end up with negative weights
                                    timers_to_update = (v[j,:]>0)
                                    timers_to_update[active_timer] = False
                                    num_active_timers = np.sum(timers_to_update)
                                    if num_active_timers > 0:
                                        dv_minus = np.zeros(N_timers)
                                        # Decrease coupling weights of all timers except the most active (ind_to_increase)
                                        dv_minus[timers_to_update] = np.minimum(v[j,timers_to_update], dv[timers_to_update])
                                        dv_minus[active_timer] = 0

                                        # Add weight changes
                                        v[j,:] -= dv_minus
                                        
                                        v[j,active_timer] += np.sum(dv_minus)
                                        eligibility_window = v[j,:]@timers

                # Keep weights within limits
                if w[j] < w_min: w[j] = w_min
                elif w[j] > w_max: w[j] = w_max
                w[j] += plasticity_params['decay_rate']*(w_max/2 - w[j])

        # Final state
        w_final[sim,:] = np.copy(w)

        eye_history[sim,:,sample_count] = pc_eye_sensitivity*pc_current
        # eye_history_no_noise[sim, :,sample_count] = pc_eye_sensitivity*pc_current_avg
        
        if hist_every == 2:
            hist_history_PF[sim, :, :, sample_count] += np.copy(hist_history_sim)
        elif hist_every == 1:
            hist_history_PF[sim, :, :] += np.copy(hist_history_sim)
        elif hist_every == 0:
            hist_history_PF += np.copy(hist_history_sim)
        cf_prob_avg[sim, :, sample_count] += cf_prob_sim/n_to_divide
        w_mean[sim, sample_count] = np.mean(w)
        if len(PF_samples) > 0:
            w_sample[sim, :, sample_count] = np.copy(w[PF_samples])
        if metaplasticity:
            if plasticity_model == 'single':
                if len(PF_samples) > 0:
                    temp_history[sim, :, sample_count] = np.copy(tau_peaks[PF_samples])
                temp_history_mean[sim, :, sample_count] = (np.mean(tau_peaks), np.std(tau_peaks))
            elif plasticity_model == 'multiple':
                if len(PF_samples) > 0:
                    temp_history[sim, :, :, sample_count] = np.copy(v[PF_samples, :])
                temp_history_mean[sim, :, sample_count] = np.mean(v@timers,axis=0)

    return_dict = {
        'w_final': w_final,
        'w_mean': w_mean,
        'eye_history': eye_history,
        'cf_prob_avg': cf_prob_avg,
    }
    if calculate_hist:
        return_dict['hist_history_PF'] = hist_history_PF
    if len(PF_samples) > 0:
        return_dict['w_sample'] = w_sample

    if metaplasticity:
        if plasticity_model == 'single':
            return_dict['tau_peaks'] = tau_peaks
        elif plasticity_model == 'multiple':
            return_dict['v'] = v
        if len(PF_samples) > 0:
            return_dict['temp_history_mean'] = temp_history_mean
            return_dict['temp_history'] = temp_history

    
    return return_dict


### TeMP using spike-timing distributions
def selectionFixed(tau, timers, v_initial, get_interval, its=5e6, dv = 1e-4, report_its = 1e4,notebook_mode=False,rng_seed=0):
    """Simulate ATTRAP with fixed update method of timer selection mechanism.
    
    Arguments:
    tau -- time vector
    timer_bank -- matrix containing timer activations
    v_initial -- initial vector of eligibility coupling weights
    distr -- distribution of PF/CF intervals
    its -- total number of PF/CF presentations (iterations) to simulate
    dv -- "Delta", the amount by which to decrease coupling weights for inactive timers
    report_its -- Frequency (in iterations) of reporting output weight vector
    
    Returns:
    v -- final weight vector
    v_all -- weight vector reported at intervals specified by report_its
    """
    
    # cdf = np.cumsum(distr/np.sum(distr))

    # Use CDF to draw intervals randomly from distribution given uniform random variables (see Methods)
    # get_interval = lambda r: np.interp(r, cdf, t)
    # get_interval_NN = lambda r: np.array(np.round(get_interval(r)*10000)/10000/dt, dtype=int)
    
    # Used for reporting weight values periodically
    report_counter = 0
    report_index = 0

    num_states = timers.shape[0]
    active_v_threshold = dv/num_states

    # num_updates = np.zeros(num_states)

    v_all = np.zeros((num_states, int(its/report_its)))
    v = np.copy(v_initial)

    trange = tqdm.notebook.trange if notebook_mode else tqdm.trange

    rng = np.random.default_rng(seed=rng_seed)

    for it in trange(its, leave=False):
        # Calculate values of most active basis function
        # given random draw of PF/CF interval
        spike_ind = np.digitize(get_interval(rng.random()), tau)-1
        timer_vals = timers[:,spike_ind]

        # Get index of most active timer
        active_timer = np.argmax(timer_vals)

        # Shouldn't end up with negative weights
        timers_to_update = (v>0)
        timers_to_update[active_timer] = False
        num_active_timers = np.sum(timers_to_update)

        if num_active_timers > 0:
            dv_minus = np.zeros(num_states)
            # Decrease coupling weights of all timers except the most active (ind_to_increase)
            dv_minus[timers_to_update] = np.minimum(v[timers_to_update], dv)
            dv_minus[active_timer] = 0

            # Add weight changes
            v -= dv_minus
            
            v[active_timer] += np.sum(dv_minus)

        report_counter += 1
        # Save current weight vector periodically
        if report_counter == report_its:
            v_all[:, report_index] = np.copy(v)
            report_counter = 0
            report_index += 1
    return v, v_all

def selectionProportional(tau, timers, v_initial, get_interval, its=5e6, dv = 1e-4, report_its = 1e4, notebook_mode=False, rng_seed=0):
    """Simulate ATTRAP with fixed update method of timer selection mechanism.
    
    Arguments:
    v_initial -- initial vector of eligibility coupling weights
    distr -- distribution of PF/CF intervals
    its -- total number of PF/CF presentations (iterations) to simulate
    dv -- "Delta", the amount by which to decrease coupling weights for inactive timers
    report_its -- Frequency (in iterations) of reporting output weight vector
    
    Returns:
    v -- final weight vector
    v_all -- weight vector reported at intervals specified by report_its
    """

    # Used for reporting weight values periodically
    report_counter = 0
    report_index = 0

    num_states = timers.shape[0]
    active_v_threshold = dv/num_states

    v_all = np.zeros((num_states, int(its/report_its)))
    v = np.copy(v_initial)

    trange = tqdm.notebook.trange if notebook_mode else tqdm.trange

    rng = np.random.default_rng(seed=rng_seed)

    for it in trange(its, leave=False):
        # Calculate values of most active basis function
        # given random draw of PF/CF interval
        spike_ind = np.digitize(get_interval(rng.random()), tau)-1
        timer_vals = timers[:,spike_ind]

        # Get index of most active timer
        active_timer = np.argmax(timer_vals)
        
        # Update weights
        v_ = v*(1-dv)
        v_[active_timer] = v[active_timer] + dv*(1-v[active_timer])
        v = np.copy(v_)
        
        report_counter += 1
        # Save current weight vector periodically
        if report_counter == report_its:
            v_all[:, report_index] = np.copy(v)
            report_counter = 0
            report_index += 1
    return v, v_all

### Biochemical implementations

timer_rates = np.array([[10000,2000],[100,40], [5,7.5], [2000, 750], [2000, 500]])
timer_km = np.array([[0.01,0.01], [0,0.1], [0.01,0.01], [0.01,0.01], [0.01, 0.01]])
k_switch_off = 100

def GKSwitchDynamicIntegrator(t,y, scale_rate):
    dydt = np.zeros(4)

    dydt[0] = timer_rates[0,0]*y[1]*(1-y[0])/(timer_km[0,0] + 1 - y[0]) - timer_rates[0,1]*y[0]/(timer_km[0,1] + y[0])
    dydt[1] = -(timer_rates[1,0] + k_switch_off*y[3])*y[1] + (timer_rates[1,1]*y[0])*(1-y[1])/(1-y[1]+timer_km[1,1])

    dydt[2] = timer_rates[2,0]*y[0]*(1 - y[2])/(1 - y[2] + timer_km[2,0]) - timer_rates[2,1]*y[2]/(y[2]+timer_km[2,1])*(1-y[0])

    dydt[3] = scale_rate[0]*y[2]*(1-y[3])/(timer_km[3,0] + 1 - y[3]) - scale_rate[1]*y[3]/(timer_km[3,1]+y[3])
    return dydt

def GKSwitchDynamicIntegratorJac(t,y, scale_rate):
    jac = np.zeros((4,4))

    jac[0,0] = timer_rates[0,0]*y[1]*(-timer_km[0,0]/(timer_km[0,0]+1-y[0])**2) - timer_rates[0,1]*timer_km[0,1]/(timer_km[0,1]+y[0])**2
    jac[0,1] = timer_rates[0,0]*(1-y[0])/(timer_km[0,0]+1-y[0])

    jac[1,0] = timer_rates[1,1]*(1-y[1])/(1-y[1]+timer_km[1,1])
    jac[1,1] = -(timer_rates[1,0]+k_switch_off*y[3]) + (timer_rates[1,1]*y[0])*(-timer_km[1,1])/(1-y[1]+timer_km[1,1])**2
    jac[1,3] = -k_switch_off*y[1]

    jac[2,0] = timer_rates[2,1]*y[2]/(y[2]+timer_km[2,1])
    # jac[2,1] = k_pr*scale_rate*(1-y[2])/(1-y[2]+timer_km[2,0])
    jac[2,2] = timer_rates[2,0]*y[0]*(-1)/(1-y[2]+timer_km[2,0])**2 - timer_rates[2,1]*(1-y[0])/(y[2]+timer_km[2,1])**2

    jac[3,2] = scale_rate[0]*(1-y[3])/(timer_km[3,0] + 1 - y[3])
    jac[3,3] = scale_rate[0]*y[2]*(-timer_km[3,0])/(timer_km[3,0]+1-y[3])**2 - scale_rate[1]/(timer_km[3,1]+y[3])**2

    return jac

def singleTimerBiochemical(scale_rate_linear, t=[], dephos_rate=5000, max_step=1e-4, initial_a2 = 0.3, T_max = 0.2):
    
    dense_output = False if len(t) == 0 else True
    T_max = t[-1] + t[1] if len(t) == 0 else T_max

    ## Initialize simulation
    y0 = np.array([0, initial_a2,0,0])

    scale_rate_linear_2 = np.maximum(scale_rate_linear - 0.05, scale_rate_linear)

    scale_rate = np.array([dephos_rate/scale_rate_linear, dephos_rate])
    scale_rate_2 = np.array([dephos_rate/scale_rate_linear_2, dephos_rate])
    
    sol = scipy.integrate.solve_ivp(lambda t,y: GKSwitchDynamicIntegrator(t,y, scale_rate), (0, T_max), y0,
                                jac=lambda t,y: GKSwitchDynamicIntegratorJac(t,y,scale_rate),
                                method='BDF',max_step=max_step, dense_output=dense_output)
    if dense_output:
        return sol.sol(t)
    else:
        return sol.t, sol.y

def TeMP_rule_singleTimer(spike_times, tau, timer, use_narrow, left_scale=1,right_scale=1):
    if use_narrow:
        drate_plus = np.interp(spike_times,  tau, (1-timer[0,:])*(timer[3,:]))
        drate_minus = np.interp(spike_times,  tau, timer[0,:]*(timer[3,:]))
    else:
        drate_plus = np.interp(spike_times,  tau, (1-timer[0,:]))
        drate_minus = np.interp(spike_times,  tau, timer[0,:])
    return right_scale*drate_plus - left_scale*drate_minus

def simualate_singleTimerBiochemicalDiscretized(timers, scale_rates_discretized,
                                        itrs, get_interval, use_narrow, itr_report = 1, scale_rate_linear_initial = 0.01,
                                        drate_scale = 0.01, dephos_rate = 5000,notebook_mode=False,leave_trange=True, rng_seed = 0):
    if notebook_mode:
        trange = tqdm.notebook.trange
    else:
        trange = tqdm.trange

    scale_rate_linear = scale_rate_linear_initial

    # Store peak times
    times = np.zeros(int(np.ceil(itrs/itr_report)))
    # Store scale rate at that time
    scale_rates = np.zeros(int(np.ceil(itrs/itr_report)))

    # Load discretized timers
    dt_mp = 1e-3
    T_mp_max = 0.2

    tau = np.arange(0, int(T_mp_max/dt_mp))*dt_mp
    rng = np.random.default_rng(rng_seed)

    for i in trange(itrs, dynamic_ncols=True,leave=leave_trange):
        timer_ind = np.digitize(scale_rate_linear, scale_rates_discretized)-1
        timer = timers[timer_ind,:,:]

        r = rng.random()
        spike_time =   get_interval(r)

        drate = drate_scale*TeMP_rule_singleTimer(spike_time, tau, timer, use_narrow, right_scale = 1/0.8)
        if drate > drate_scale: drate = drate_scale # *0.001
        if drate < -drate_scale: drate = -drate_scale # *0.001
        scale_rate_linear += drate
        if i%itr_report == 0:
            j = int(i/itr_report)
            times[j] = tau[np.argmax(timer[2,:])] # sol.t[np.where(sol.y[3,:]>0)[0][0]]
            scale_rates[j] = scale_rate_linear

        if scale_rate_linear < scale_rate_linear_initial: scale_rate_linear = scale_rate_linear_initial

    return times, scale_rates