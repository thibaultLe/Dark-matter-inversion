# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 17:33:26 2022

@author: Thibault
"""

import numpy as np
from matplotlib.pylab import plt
import orbitModule



comparedData = np.loadtxt('Datasets/Kepler.txt')
timegrid = comparedData[:,0]

t_grid = orbitModule.convertYearsTimegridToOurFormat(timegrid)

IC = orbitModule.get_S2_IC()

#Dark matter:
N = 5
mis = N*[0] #-> 0 dark matter, has no effect
ris = np.linspace(0,1000,N)

rx,ry,rz, vx,vy,vz = orbitModule.simulateOrbitsCartesian(False,IC, mis, ris,t_grid)
rxPN,ryPN,rzPN, vxPN,vyPN,vzPN = orbitModule.simulateOrbitsCartesian(True,IC, mis, ris,t_grid)


#Plot effects of 1PN:
plt.figure()
plt.xlabel('Time (years)')
plt.ylabel('Value')
plt.scatter(timegrid,orbitModule.AU_to_arcseconds(rx)*1e6,label='X Kepler , [µas]',s=10)
plt.scatter(timegrid,orbitModule.AU_to_arcseconds(rxPN)*1e6,label='X PN , [µas]',s=10)
plt.legend()
plt.title('Comparison of X solutions')


#Plot effects of 1PN:
plt.figure()
plt.xlabel('Time (years)')
plt.ylabel('Value')
plt.scatter(timegrid,orbitModule.AU_to_arcseconds(ry)*1e6,label='Y Kepler , [µas]',s=10)
plt.scatter(timegrid,orbitModule.AU_to_arcseconds(ryPN)*1e6,label='Y PN , [µas]',s=10)
plt.legend()
plt.title('Comparison of Y solutions')


#Plot effects of 1PN:
plt.figure()
plt.xlabel('Time (years)')
plt.ylabel('Value')
plt.scatter(timegrid,vz/1000,label='VZ Kepler , [km/s]',s=10)
plt.scatter(timegrid,vzPN/1000,label='VZ PN , [km/s]',s=10)
plt.legend()
plt.title('Comparison of VZ solutions')



#Plot difference of 1PN:
plt.figure()
plt.xlabel('Time (years)')
plt.ylabel('X Kepler - X PN  [µas]')
plt.scatter(timegrid,orbitModule.AU_to_arcseconds(rx-rxPN)*1e6,s=10,label='Difference')
plt.plot(timegrid,len(timegrid)*[50],'--',label='Precision',color='red')
plt.plot(timegrid,len(timegrid)*[-50],'--',color='red')
plt.legend()
plt.title('Difference of X solutions')

#Plot difference of 1PN:
plt.figure()
plt.xlabel('Time (years)')
plt.ylabel('Y Kepler - Y PN  [µas]')
plt.scatter(timegrid,orbitModule.AU_to_arcseconds(ry-ryPN)*1e6,s=10,label='Difference')
plt.plot(timegrid,len(timegrid)*[50],'--',label='Precision',color='red')
plt.plot(timegrid,len(timegrid)*[-50],'--',color='red')
plt.legend()
plt.title('Difference of Y solutions')


#Plot difference of 1PN:
plt.figure()
plt.xlabel('Time (years)')
plt.ylabel('VZ Kepler - VZ PN  [km/s]')
plt.scatter(timegrid,(vz-vzPN)/1000,s=10,label='Difference')
plt.plot(timegrid,len(timegrid)*[10],'--',label='Precision',color='red')
plt.plot(timegrid,len(timegrid)*[-10],'--',color='red')
plt.legend()
plt.title('Difference of VZ solutions')


