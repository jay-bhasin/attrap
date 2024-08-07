# Temporal Metaplasticity
This repository contains code for producing simulation results found in our paper. Because of size constraints, we do not include the outputs of our simulations, but similar results can be obtained by running the code found within. 

## Required dependencies
Code was developed in Python 3.10.9, using NumPy 1.23.5, Scipy 1.10.0 and Matplotlib 3.7.0. Many scripts rely on tqdm (using version 4.64.1). Notebooks were developed in ipython 8.10.0.

## Installation guide
Simply clone this repository.

## Instructions for use
This repository contains code that simulates the temporal metaplasticity mechanisms at the synaptic level described in the Methods and Supplementary Methods sections of our paper, as well as integrated simulations of plasticity and metaplasticity in the cerebellar circuit underlying oculomotor learning. Instructions for reproducing the results shown in Fig. 4 and Extended Data Figs. 6–9 are given below. 

### Temporal metaplasticity mechanisms
Code for reproducing panels in Fig. 4 and Extended Data Figs. 6, 8 and 9 is found in the following files in the `synaptic-models` directory:
<!-- Contains files for idealized and biochemical implementations of temporal metaplasticity mechanisms. We hypothesize two classes of mechanism, single timer adjustment and multiple timer selection: -->

- `ideal-models.ipynb`: contains code for simulating idealized versions of all 4 implementations of temporal metaplasticity described in the Methods (_Modeling of temporal metaplasticity_) given a parallel fiber-climbing fiber spike time interval distribution, and for producing the final plasticity rules
- `ideal-models-dark.ipynb`: contains code for simulating modified versions of the narrow windowed single timer adjustment rule and the winner-take-all multiple timer selection rule that tune plasticity to a close-to-coincident timing preference given a uniform spike interval distribution in the dark, while correctly selecting the peak timing distribution in the light (see Methods, _Returning to a default timing in the dark_); generates panels shown in **_Extended Data Fig. 8**
- `timer-adjustment-biochemical.py`: contains code for simulating the biochemical implementation of accumulation-to-bound timer (described in Methods, _Single timer adjustment mechanism_)
- `timer-adjustment-biochemical.ipynb`: contains analysis code for biochemical implementation of accumulation-to-bound timer and generates plots for **_Extended Data Fig. 6b–d_**
- `timer-selection.ipynb`: contains the biochemical implementation of a cascade of molecular timers (Methods, _Biochemical implementation of molecular timer bank_) and simulates temporal metaplasticity by selecting from these timers, generating plots for **_Extended Data Fig. 6e,g–h_**
- `timer-selection-biochemical-updates.ipynb`: contains implementations of mechanisms to perform the winner-take-all and proportional updates biochemically, and simulates temporal metaplasticity using these implementations (**_Extended Data Fig. 9_**), as described in the Supplementary Methods

Additionally, this folder contains `cascade_timers.npy`, which contains a NumPy matrix of the time courses of timer activations from the cascade implementation, and `thresholds.npy` which contains a NumPy array of threshold parameters for generating these timers. The file `timer_adjustment_utils.py` is required for running the simulation code in `timer-adjustment-biochemical.py`.


### Integrated simulation of plasticity and metaplasticity in a cerebellar circuit model

Code for simulating the cerebellar learning model used to generate results shown in Extended Data Fig. 7 can be found in the following files in the `behavior-learning` directory:

- `poisson.py`: script for running integrated simulations of plasticity and metaplasticity assuming Poisson firing in parallel fibers
- `poisson.ipynb`: notebook for analyzing integrated simulation with Poisson firing, and for generating panels **c** and **d** in Extended Data Fig. 7
- `learning.ipynb`: notebook for running and analyzing simulations of oculomotor (OKR) learning, and for generating panels **b** and **e**, top, in Extended Data Fig. 7
- `learning-basis.py`: script for running integrated simulation of plasticity and metaplasticity assuming parallel fiber firing is same as for OKR learning
- `learning-basis.ipynb`: notebook for analyzing integrated simulations of plasticity and metaplasticity assuming PF firing is same as for OKR learning, and for generating panel **e**, bottom, in Extended Data Fig. 7

