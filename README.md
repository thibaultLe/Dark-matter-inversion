# Dark-matter-inversion
Welcome to the repo for the dark matter inversion project of ESA's Advanced Concepts Team.

The code is built around a main module (orbitModule.py), containing the most important functions.
All other scripts use this module and you can run our experiments from there (e.g. reconstructDistribution.py, simulateOrbits.py, ...).

The datasets used can be found under /Datasets.


# Installation:
Make a new conda environment.

Install Heyoka from https://bluescarni.github.io/heyoka.py/install.html

Run 'conda install pygmo matplotlib pandas'

Run 'python buildDataset.py' to build the taylor integrators

