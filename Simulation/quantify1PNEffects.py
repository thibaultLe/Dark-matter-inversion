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

#Plot difference of 1PN:
plt.figure()
plt.xlabel('Time (years)')
plt.ylabel('X Kepler - X PN  [µas]')
plt.scatter(timegrid,AU_to_arcseconds(rx-rxPN)*1e6,s=10,label='Difference')
plt.plot(timegrid,len(timegrid)*[50],'--',label='Precision',color='red')
plt.plot(timegrid,len(timegrid)*[-50],'--',color='red')
plt.legend()

#Plot difference of 1PN:
plt.figure()
plt.xlabel('Time (years)')
plt.ylabel('Y Kepler - Y PN  [µas]')
plt.scatter(timegrid,AU_to_arcseconds(ry-ryPN)*1e6,s=10,label='Difference')
plt.plot(timegrid,len(timegrid)*[50],'--',label='Precision',color='red')
plt.plot(timegrid,len(timegrid)*[-50],'--',color='red')
plt.legend()


#Plot difference of 1PN:
plt.figure()
plt.xlabel('Time (years)')
plt.ylabel('VZ Kepler - VZ PN  [km/s]')
plt.scatter(timegrid,(vz-vzPN)/1000,s=10,label='Difference')
plt.plot(timegrid,len(timegrid)*[10],'--',label='Precision',color='red')
plt.plot(timegrid,len(timegrid)*[-10],'--',color='red')
plt.legend()



#Plot effects of 1PN:
plt.figure()
plt.xlabel('Time (years)')
plt.ylabel('Value')
plt.scatter(timegrid,AU_to_arcseconds(rx)*1e6,label='X Kepler , [µas]',s=10)
plt.scatter(timegrid,AU_to_arcseconds(rxPN)*1e6,label='X PN , [µas]',s=10)
plt.legend()


#Plot effects of 1PN:
plt.figure()
plt.xlabel('Time (years)')
plt.ylabel('Value')
plt.scatter(timegrid,AU_to_arcseconds(ry)*1e6,label='Y Kepler , [µas]',s=10)
plt.scatter(timegrid,AU_to_arcseconds(ryPN)*1e6,label='Y PN , [µas]',s=10)
plt.legend()


#Plot effects of 1PN:
plt.figure()
plt.xlabel('Time (years)')
plt.ylabel('Value')
plt.scatter(timegrid,vz/1000,label='VZ Kepler , [km/s]',s=10)
plt.scatter(timegrid,vzPN/1000,label='VZ PN , [km/s]',s=10)
plt.legend()


