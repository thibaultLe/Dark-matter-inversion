# Dark-matter-inversion
Welcome to the repo for the dark matter inversion project of [ESA's Advanced Concepts Team](https://www.esa.int/gsp/ACT/).



# Installation:
Make a new conda environment, activate it, then do the following steps:

Install Heyoka from https://bluescarni.github.io/heyoka.py/install.html

Run 'conda install pygmo matplotlib pandas'

Run 'python buildDataset.py' to build the taylor integrators


# Usage:
The code is built around a main module (orbitModule.py), containing the most important functions.
You should use the provided scripts to interface with this module (e.g. reconstructDistribution.py, simulateOrbits.py, ...)
Most scripts are structured as follows: functions are defined at the top, which are called at the bottom. Comment/uncomment the desired functions and run the files to run the experiments.

The /Datasets folder contains files with observational data, which is used by some of the scripts.


# Citation:
      @Article{lechien2023dark,
      title={Dark Matter reconstruction from stellar orbits in the Galactic Centre}, 
      author={Thibault Lechien and Gernot Heißel and Jai Grover and Dario Izzo},
      year={2023},
      eprint={2308.09170},
      archivePrefix={arXiv},
      primaryClass={astro-ph.GA}}
