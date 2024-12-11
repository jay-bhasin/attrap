# Experience Alters the Timing Rules Governing Synaptic Plasticity and Learning
This repository contains code for producing simulation results found in our paper. Because of size constraints, we do not include all the outputs of our simulations, but similar results can be obtained by running the code found within. 

## Required dependencies
Code was developed in Python 3.13.0, using NumPy 2.1.2, Scipy 1.14.1 and Matplotlib 3.9.2. Many scripts rely on tqdm (using version 4.66.5). Notebooks were developed in ipython 8.28.0.

## Installation guide
Simply clone this repository.

## Instructions for use
This repository contains code that simulates the temporal metaplasticity mechanisms at the synaptic level described in the Methods and Supplemental Methods sections of our paper, as well as integrated simulations of plasticity and metaplasticity in the cerebellar circuit underlying oculomotor learning. Instructions for reproducing the results shown in **Figures 4** and **5** and **S6–S9** are given below. 

Much of the simulation and analysis files depend on code in the library defined in `utils/te_mp.py`.

### Temporal metaplasticity mechanisms
Code for simulating our proposed temporal metaplasticity mechanisms given PF-CF pairings from a distribution, and for producing the corresponding figure panels in **Figures 4** and **5** and **S6-S9** is found in the following files in the `synaptic-models` directory:
<!-- Contains files for idealized and biochemical implementations of temporal metaplasticity mechanisms. We hypothesize two classes of mechanism, single timer adjustment and multiple timer selection: -->

- `timers-biochemical-accumulator.ipynb`: contains analysis code for all 4 implementations of temporal metaplasticity described in the Methods (_Modeling of temporal metaplasticity_) using a biochemical implementation of an accumulation-to-threshold timer, given the bimodal parallel fiber-climbing fiber spike time interval distribution (as seen in **Figs. 4H-K**), and for producing the final plasticity rules
    - `scripts/timer-adjustment-biochemical-narrow.py`: code to simulate the single timer adjustment mechanism with a narrow TeMP window
    - `scripts/timer-adjustment-biochemical-broad.py`: code to simulate the single timer adjustment mechanism with a broad TeMP window
    - `scripts/timer-selection-biochemical-fixed.py`: code to simulate the multiple timer adjustment mechanism with fixed updates (winner-take-all)
    - `scripts/timer-selection-biochemical-proportional.py`: code to simulate the multiple timer adjustment mechanism with proportional updates
- `ideal-models-dark.ipynb`: contains code for simulating modified versions of the narrow window single timer adjustment rule and the fixed updates multiple timer selection rule that tune plasticity to a close-to-coincident timing preference given a uniform spike interval distribution in the dark, while correctly selecting the peak timing distribution in the light (see Methods, _Returning to a default timing in the dark_); generates panels shown in **Fig. S7**
- `timer-selection-biochemical-updates.ipynb`: contains biochemical implementations of binding reactions for the fixed and proportional updates, and simulates temporal metaplasticity using these implementations with a simple distribution (**Fig. S9**), as described in the Supplemental Methods.
- `timers-cascade.ipynb`: contains code for generating a timer bank constructed from a cascade of sequentially active molecular molecular timers

### Integrated simulation of plasticity and metaplasticity in a cerebellar circuit model

Code for simulating the cerebellar learning model used to generate results shown in **Figure 4** and **S6** can be found in the following files in the `circuit-model` directory:

- `single-timer-adjustment-analysis.ipynb`: contains analysis code for the integrated simulation using the single-timer-adjustment rule of **Fig. 4C** (including the histogram in **Fig. 4I** and the dynamics in **Fig. 4J**)
    - `scripts/timer-adjustment-integrated.py`: script to run simulation of a long period of temporal metaplasticity with parallel fiber firing drawn from an uncorrelated basis, using an idealized eligibility window
- `multiple-timer-selection-analysis.ipynb`: contains analysis code for the integrated simulation using the multiple-timer-adjustment rule of **Fig. 4D** (including the historgram in **Fig. 4I**)
    - `scripts/timer-selection-integrated-poisson.py`: script to run simulation of a long period of temporal metaplasticity with parallel fiber firing drawn from an uncorrelated basis, using an idealized eligibility window
    - `scripts/timer-selection-integrated-structured.py`: script to run simulation of a long period of temporal metaplasticity with parallel fiber firing drawn from a structured basis (same as during the OKR adaptation experiment), using an idealized eligibility window
- `okr-adaptation.py`: analysis code for simulation of OKR adaptation (before/after temporal metaplasticity; **Fig. 4H**)
    - `scripts/timer-selection-okr.py`: script to run simulation of OKR adaptation