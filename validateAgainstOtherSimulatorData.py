# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 17:08:53 2022

@author: Thibault
"""
import numpy as np
from matplotlib.pylab import plt
import orbitModule


def getDataFromText(filename):
    return np.loadtxt(filename)

"""
Set parameters:
"""
PNCORRECTION = False

#Amount of dark matter shells
n = 5
# mis = n*[10**(-3)]
mis = n*[0] #-> 0 dark matter, has no effect (confirmed)
ris = np.linspace(0,1000,n)



"""
Read data from files
"""
if PNCORRECTION:
    txtfile = 'Datasets/1PN.txt'
else:
    txtfile = 'Datasets/Kepler.txt'
    
comparedData = getDataFromText(txtfile)
timegrid = comparedData[:,0]
comparedYs = comparedData[:,1]
comparedXs = comparedData[:,2]
comparedVZs = comparedData[:,3]

t_grid = orbitModule.convertYearsTimegridToOurFormat(timegrid)

IC = orbitModule.get_S2_IC()

rx,ry,rz, vx,vy,vz = orbitModule.simulateOrbitsCartesian(PNCORRECTION,IC, mis, ris,t_grid)


ydifs = []
xdifs = []
vzdifs = []
test = []
for i in range(len(timegrid)):
    ydifs.append(orbitModule.AU_to_arcseconds(ry[i]) - comparedYs[i])
    xdifs.append(orbitModule.AU_to_arcseconds(rx[i]) - comparedXs[i])
    vzdifs.append(-vz[i] / (1000) - comparedVZs[i])



# Plot difference with Gernot's solutions

plt.figure()
plt.plot(timegrid,orbitModule.AU_to_arcseconds(rx),label='DEC (x), [as]')
plt.plot(timegrid,comparedXs,label='Truth')
plt.xlabel('Time (years)')
plt.ylabel('Value')
plt.title('Comparison of X solutions')
plt.legend()

plt.figure()
plt.scatter(timegrid,orbitModule.AU_to_arcseconds(ry),label='RA (y), [as]',s=10)
plt.scatter(timegrid,comparedYs,label='Truth',s=10)
plt.xlabel('Time (years)')
plt.ylabel('Value')
plt.title('Comparison of Y solutions')
plt.legend()

plt.figure()
plt.plot(timegrid,-vz / 1000 ,label='RV (vz), [km/s]')
plt.plot(timegrid,comparedVZs,label='Truth')
plt.xlabel('Time (years)')
plt.ylabel('Value')
plt.title('Comparison of VZ solutions')
plt.legend()


plt.figure()
plt.xlabel('Time (years)')
plt.ylabel('Difference')
plt.scatter(timegrid,xdifs,label='X_thib - X_gernot , [as]',s=10)
plt.title('Difference of X solutions')
plt.legend()

plt.figure()
plt.xlabel('Time (years)')
plt.ylabel('Difference')
plt.scatter(timegrid,ydifs,label='Y_thib - Y_gernot , [as]',s=10)
plt.title('Difference of Y solutions')
plt.legend()

plt.figure()
plt.xlabel('Time (years)')
plt.ylabel('Difference')
plt.scatter(timegrid,vzdifs,label='VZ_thib - vZ_gernot , [km/s]',s=10)
plt.title('Difference of VZ solutions')
plt.legend()
