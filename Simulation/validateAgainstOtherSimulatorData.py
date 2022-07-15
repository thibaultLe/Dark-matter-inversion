# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 17:08:53 2022

@author: Thibault
"""
import numpy as np
from matplotlib.pylab import plt
from simulateOrbits import simulateOrbits

def AU_to_arcseconds(dist):
    D_0 = 149597870700
    R = 2.5540153e+20
    return 2 * np.arctan(dist*D_0/(2*R)) * 206264.8

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
    txtfile = '1PN.txt'
else:
    txtfile = 'Kepler.txt'
    
comparedData = getDataFromText(txtfile)
comparedYs = comparedData[:,1]
comparedXs = comparedData[:,2]
comparedVZs = comparedData[:,3]

timegrid = comparedData[:,0]
#Use an offset so that t=0 corresponds to the first observation
timeoffset = timegrid[0]
timegrid = timegrid - timeoffset

[rx,ry,rz] , [vx,vy,vz] ,lf = simulateOrbits(PNCORRECTION, mis, ris)


ydifs = []
xdifs = []
vzdifs = []
test = []
for i in range(len(timegrid)):
    ydifs.append(AU_to_arcseconds(ry[i]) - comparedYs[i])
    xdifs.append(AU_to_arcseconds(rx[i]) - comparedXs[i])
    vzdifs.append(-vz[i] / (1000) - comparedVZs[i])


#Convert back from t=0 of first observation to t = timegrid[0]
timegrid = timegrid + timeoffset

# Plot difference with Gernot's solutions
plt.figure()
plt.scatter(timegrid,AU_to_arcseconds(ry),label='RA (y), [as]',s=10)
plt.scatter(timegrid,comparedYs,label='Truth',s=10)
plt.xlabel('Time (years)')
plt.ylabel('Value')
plt.legend()

plt.figure()
plt.plot(timegrid,AU_to_arcseconds(rx),label='DEC (x), [as]')
plt.plot(timegrid,comparedXs,label='Truth')
plt.xlabel('Time (years)')
plt.ylabel('Value')
plt.legend()

plt.figure()
plt.plot(timegrid,-vz / 1000 ,label='RV (vz), [km/s]')
plt.plot(timegrid,comparedVZs,label='Truth')
plt.xlabel('Time (years)')
plt.ylabel('Value')
plt.legend()


plt.figure()
plt.xlabel('Time (years)')
plt.ylabel('Difference')
plt.scatter(timegrid,ydifs,label='Y_thib - Y_gernot , [as]',s=10)
plt.legend()

plt.figure()
plt.xlabel('Time (years)')
plt.ylabel('Difference')
plt.scatter(timegrid,xdifs,label='X_thib - X_gernot , [as]',s=10)
plt.legend()

plt.figure()
plt.xlabel('Time (years)')
plt.ylabel('Difference')
plt.scatter(timegrid,vzdifs,label='VZ_thib - vZ_gernot , [km/s]',s=10)
plt.legend()
