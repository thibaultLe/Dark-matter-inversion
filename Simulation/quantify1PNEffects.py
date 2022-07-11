# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 17:33:26 2022

@author: Thibault
"""

import numpy as np
from matplotlib.pylab import plt
from simulateOrbits import simulateOrbits

def AU_to_arcseconds(dist):
    D_0 = 149597870700
    R = 2.5540153e+20
    return 2 * np.arctan(dist*D_0/(2*R)) * 206264.8


comparedData = np.loadtxt('Kepler.txt')
timegrid = comparedData[:,0]

#Dark matter:
n = 5
mis = n*[0] #-> 0 dark matter, has no effect (confirmed)
ris = np.linspace(0,1000,n)


[rx,ry,rz] , [vx,vy,vz] = simulateOrbits(False, mis, ris)
[rxPN,ryPN,rzPN] , [vxPN,vyPN,vzPN] = simulateOrbits(True, mis, ris)

#Plot effects of 1PN:
plt.figure()
plt.xlabel('Time (years)')
plt.ylabel('Difference')
plt.scatter(timegrid,AU_to_arcseconds(rx-rxPN),label='X newton - X PN , [as]',s=10)
plt.legend()

#Plot effects of 1PN:
plt.figure()
plt.xlabel('Time (years)')
plt.ylabel('Value')
plt.scatter(timegrid,AU_to_arcseconds(rx),label='X newton , [as]',s=10)
plt.scatter(timegrid,AU_to_arcseconds(rxPN),label='X PN , [as]',s=10)
plt.legend()


