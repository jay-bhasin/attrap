import numpy as np
import scipy.integrate
import tqdm

rates = np.array([[10000,2000],[100,40], [5,7.5], [2000, 750], [2000, 500]])
km = np.array([[0.01,0.01], [0,0.1], [0.01,0.01], [0.01,0.01], [0.01, 0.01]])
k_switch_off = 100

def GKSwitchDynamicIntegrator2(t,y, scale_rate, scale_rate_2):
    dydt = np.zeros(5)

    dydt[0] = rates[0,0]*y[1]*(1-y[0])/(km[0,0] + 1 - y[0]) - rates[0,1]*y[0]/(km[0,1] + y[0])
    dydt[1] = -(rates[1,0] + k_switch_off*y[3])*y[1] + (rates[1,1]*y[0])*(1-y[1])/(1-y[1]+km[1,1])

    dydt[2] = rates[2,0]*y[0]*(1 - y[2])/(1 - y[2] + km[2,0]) - rates[2,1]*y[2]/(y[2]+km[2,1])*(1-y[0])

    dydt[3] = scale_rate[0]*y[2]*(1-y[3])/(km[3,0] + 1 - y[3]) - scale_rate[1]*y[3]/(km[3,1]+y[3])
    dydt[4] = scale_rate_2[0]*y[2]*(1-y[4])/(km[4,0] + 1 - y[4]) - scale_rate_2[1]*y[4]/(km[4,1]+y[4])
    return dydt

def GKSwitchDynamicIntegratorJac2(t,y, scale_rate, scale_rate_2):
    jac = np.zeros((5,5))

    jac[0,0] = rates[0,0]*y[1]*(-km[0,0]/(km[0,0]+1-y[0])**2) - rates[0,1]*km[0,1]/(km[0,1]+y[0])**2
    jac[0,1] = rates[0,0]*(1-y[0])/(km[0,0]+1-y[0])

    jac[1,0] = rates[1,1]*(1-y[1])/(1-y[1]+km[1,1])
    jac[1,1] = -(rates[1,0]+k_switch_off*y[3]) + (rates[1,1]*y[0])*(-km[1,1])/(1-y[1]+km[1,1])**2
    jac[1,3] = -k_switch_off*y[1]

    jac[2,0] = rates[2,1]*y[2]/(y[2]+km[2,1])
    # jac[2,1] = k_pr*scale_rate*(1-y[2])/(1-y[2]+km[2,0])
    jac[2,2] = rates[2,0]*y[0]*(-1)/(1-y[2]+km[2,0])**2 - rates[2,1]*(1-y[0])/(y[2]+km[2,1])**2

    jac[3,2] = scale_rate[0]*(1-y[3])/(km[3,0] + 1 - y[3])
    jac[3,3] = scale_rate[0]*y[2]*(-km[3,0])/(km[3,0]+1-y[3])**2 - scale_rate[1]/(km[3,1]+y[3])**2

    jac[4,2] = scale_rate_2[0]*(1-y[4])/(km[4,0] + 1 - y[4])
    jac[4,4] = scale_rate_2[0]*y[2]*(-km[4,0])/(km[4,0]+1-y[4])**2 - scale_rate_2[1]/(km[4,1]+y[4])**2

    return jac

def simulateTimerAdjustment(itrs, distr_weight, itr_report = 1, scale_rate_linear_initial = 0.01, drate_scale = 0.01, median_d = 0.01, dephos_rate = 5000 ):
    ## Define distribution
    dt = 1e-4 # Time step size (s)
    T = 0.2 # Total time of simulation
    t = np.arange(0, int(T/dt))*dt

    distr_ = np.exp(-((t - 120e-3)/(2*5e-3))**2)
    distr_flat = np.ones(len(t))/len(t)

    # distr_weight = 0.1
    distr = distr_/np.sum(distr_)*distr_weight + (1-distr_weight)*distr_flat
    cdf = np.cumsum(distr)
    get_interval = lambda r: np.interp(r, cdf, t)

    ## Initialize simulation
    scale_rate_linear = scale_rate_linear_initial

    times = np.zeros(int(np.ceil(itrs/itr_report)))
    scale_rates = np.zeros(int(np.ceil(itrs/itr_report)))
    y0 = np.array([0, 0.3,0,0, 0])

    for i in tqdm.trange(itrs, dynamic_ncols=True):

        scale_rate_linear_2 = np.maximum(scale_rate_linear - 0.05, scale_rate_linear)


        scale_rate = np.array([dephos_rate/scale_rate_linear, dephos_rate])
        scale_rate_2 = np.array([dephos_rate/scale_rate_linear_2, dephos_rate])
        sol = scipy.integrate.solve_ivp(lambda t,y: GKSwitchDynamicIntegrator2(t,y, scale_rate, scale_rate_2), (0, 0.2), y0,
                                    jac=lambda t,y: GKSwitchDynamicIntegratorJac2(t,y,scale_rate, scale_rate_2),
                                    method='BDF',max_step=1e-4)

        r = np.random.rand()
        spike_time =   get_interval(r) # get_interval_flat(r) #

        attrap_plus = np.interp(spike_time,  sol.t, (1-sol.y[0,:])*(sol.y[4,:]+median_d))
        attrap_minus = np.interp(spike_time,  sol.t, sol.y[0,:]*(sol.y[4,:]+median_d))

        drate = drate_scale*(attrap_plus - attrap_minus)
        if drate > drate_scale: drate = drate_scale*0.001
        if drate < -drate_scale: drate = -drate_scale*0.001
        scale_rate_linear += drate
        if i%itr_report == 0:
            j = int(i/itr_report)
            times[j] = sol.t[np.argmax(sol.y[2,:])] # sol.t[np.where(sol.y[3,:]>0)[0][0]]
            scale_rates[j] = scale_rate_linear

        if scale_rate_linear < scale_rate_linear_initial: scale_rate_linear = scale_rate_linear_initial

    return times, scale_rates
