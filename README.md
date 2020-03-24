# Experience Adaptively Tunes the Timing Rules for Associative Plasticity
This repository contains code for simulation results found in our paper. It includes the following directories:

## behavior-learning

Contains files for the cerebellar learning model used to make behavioral predictions, in figure 2 and extended data figure 2:

- **behavior-learning-tuned.py**: simulates learning in the cerebellar model using a plasticity rule tuned to the 120 ms feedback delay present in the circuit.
- **behavior-learning-untuned.py**: simulates learning in the cerebellar model using a plasticity rule that is untuned to this delay, i.e., induces plasticity for coincident parallel fiber and climbing fiber spikes. 
- **behavior-learning.ipynb**: Jupyter notebook for analysis of the simulation results, and generates plots for **_figure 2c,d_** and **_extended data figure 2_**. 

The **results** subdirectory contains raw simulation results for 10 runs, analyzed in **behavior-learning.ipynb**.

## synaptic-models

Contains files for our implementations of ATTRAP for climbing-fiber induced associative LTD at parallel fiber-Purkinje cell synapses. We hypothesize two classes of mechanism, timer adjustment and timer selection:

- **timer-adjustment.ipynb**: contains the biochemical implementation of an accumulation-to-bound timer and simulates ATTRAP by adjusting this timer, generating plots for **_figure 3c,e_** and **_extended data figure 3b,e_**. Also contains simulation of an idealized model, showing that this mechanism picks out the median of the parallel fiber/climbing fiber (PF/CF) interval distribution, generating plots for **_extended data figure 3f_**.
- **timer-selection.ipynb**: contains the biochemical implementation of a cascade of molecular timers and simulates ATTRAP by selecting from these timers, generating plots for **_figure 3d,f_** and **_extended data figure 3c_**. Also contains simulation of idealized models, proposing that the fixed update method of this mechanism picks out the mode of the PF/CF interval distribution  (**_extended data fig. 3g_, left**), and that the proportional update method results in an eligibility window whose shape approximates the shape of the distribution  (**_extended data fig. 3g_, right**). We also generate the ATTRAP window, analogous to the STDP window but translating spike times into changes in the eligibility window (**_fig. 3d_**, **_extended data fig. 3h_**).
- **timer-selection-biochemical-updates.ipynb**: contains implementations of mechanisms to perform the fixed and proportional updates biochemically, and simulates ATTRAP using these implementations (**_extended data fig. 4_**). 

Additionally, this folder contains **cascade_timers.npy**, which contains a Numpy matrix of the time courses of timer activations from the cascade implementation, and **thresholds.npy** which contains a Numpy array of threshold parameters for generating these timers.