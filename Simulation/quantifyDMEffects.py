# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:25:26 2022

@author: Thibault
"""

import numpy as np
from matplotlib.pylab import plt
from simulateOrbits import simulateOrbits

#1AU in meters
D_0 = 149597870700
#Solar mass
M_sol = 1.98841 * 10**30 
# mass m' = m/M_0 -> MBH = 1
M_0 = 4.2970174 * 10**6 * M_sol

def AU_to_arcseconds(dist):
    R = 2.5540153e+20
    return 2 * np.arctan(dist*D_0/(2*R)) * 206264.8



comparedData = np.loadtxt('Kepler.txt')
timegrid = comparedData[:,0]

#Dark matter:
#Amount of mascons:
N = 20
mis0 = N*[0] #-> 0 dark matter, has no effect
ris0 = np.linspace(0,1000,N)

[rxPN,ryPN,rzPN] , [vxPN,vyPN,vzPN] = simulateOrbits(True, mis0, ris0)



#Plot dark matter distribution
#AU limit
xlim = 4000
#Amount of points in linspace
n = 1000
#Bahcall-Wolf cusp model:
rDM = np.linspace(0,xlim,n)

rho0plum = 1.69*10**(-10) * (D_0**3) / M_0
# rho0cusp = 2.24*10**(-11) * (D_0**3) / M_0
r0 = 2474.01

#Plummer model:
rhoPlum = rho0plum *( 1. + ((rDM**2) / (r0**2)))**(-5/2)
#Cusp model:
# rhoCusp = rho0cusp * (rDM / r0)**(-7/4)

def enclosedMass(a,rho0):
    return (4 * a**3 * np.pi * r0**3 * rho0) / ( 3 * (a**2 + r0**2)**(3/2))


x_right = rDM[::round(n/(N+1))] # ri's of mascon shells
ris = x_right[1:]
y_right = enclosedMass(rDM,rho0plum)[::round(n/(N+1))] # enclosed mi's of mascon shells
#mascon masses = difference in enclosed mass:
mis = [t - s for s, t in zip(y_right, y_right[1:])]

print('ris:',ris)
print('mis:',mis)

#Check numerical stability by setting most mascons to zero:
# for i in range(len(mis)):
#     if i != 5:
#         mis[i] = 0
# print('mis:',mis)


#Plot enclosed mass
# plt.figure()
# plt.xlabel('Distance from MBH [AU]')
# plt.plot(rDM,enclosedMass(rDM,rho0plum),label='Plum model')
# plt.ylabel('enclosed mass [MBH masses]')
# plt.axvline(rp,linestyle='--',label='rp and ra',color='black')
# plt.axvline(ra,linestyle='--',color='black')
# plt.scatter(x_right[1:],y_right[1:],color='orange',label='Mascon enclosed mass')
# plt.bar(x_right,y_right,width=(xlim)/(N+1),alpha=0.2,align='edge',edgecolor='orange',color='orange')
# plt.legend()

#Plot mass of mascons
plt.figure()
#Plot rp, ra for ellipses:
a = 1034.2454074981154
e = 0.884429099282 
rp = a*(1-e)
ra = a*(1+e)
plt.scatter(ris,mis,label='mascons',color='orange')
plt.axvline(rp,linestyle='--',label='rp and ra',color='black')
plt.axvline(ra,linestyle='--',color='black')
plt.ylabel('Mass [MBH masses]',color='orange')
plt.xlabel('Distance from MBH [AU]')
plt.legend()



#Plot theoretical plum model in kg/m^3 and mpc
# rDMmpc = rDM*0.004848
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.plot(rDMmpc,rhoPlum*M_0/(D_0**3)*10**10,color='b')
# # ax1.plot(rDMmpc,rhoCusp*M_0/(D_0**3)*10**10,'--',color='b')
# ax1.set_xlabel('r [mpc]')
# ax1.set_ylabel('rho [10^(-10) kg/m³]',color='b')
# ax1.set_ylim(0,4)
# ax2.plot(rDMmpc,enclosedMass(rDM,rho0plum)*M_0/(M_sol*1e3),color='orange')
# ax2.set_ylabel('enclosed mass [10³ solar masses]',color='orange')
# plt.show()




[rxDM,ryDM,rzDM] , [vxDM,vyDM,vzDM] = simulateOrbits(True, mis, ris)

#Plot difference of DM:
plt.figure()
plt.xlabel('Time (years)')
plt.ylabel('X PN - X DM  [µas]')
plt.scatter(timegrid,AU_to_arcseconds(rxPN-rxDM)*1e6,s=10,label='Difference')
plt.plot(timegrid,len(timegrid)*[25],'--',label='Precision',color='red')
plt.plot(timegrid,len(timegrid)*[-25],'--',color='red')
plt.legend()

#Plot difference of DM:
plt.figure()
plt.xlabel('Time (years)')
plt.ylabel('Y PN - Y DM  [µas]')
plt.scatter(timegrid,AU_to_arcseconds(ryPN-ryDM)*1e6,s=10,label='Difference')
plt.plot(timegrid,len(timegrid)*[25],'--',label='Precision',color='red')
plt.plot(timegrid,len(timegrid)*[-25],'--',color='red')
plt.legend()


#Plot difference of DM:
plt.figure()
plt.xlabel('Time (years)')
plt.ylabel('VZ PN - VZ DM  [km/s]')
plt.scatter(timegrid,-(vzPN-vzDM)/1000,s=10,label='Difference')
plt.plot(timegrid,len(timegrid)*[5],'--',label='Precision',color='red')
plt.plot(timegrid,len(timegrid)*[-5],'--',color='red')
plt.legend()



#Plot effects of DM:
plt.figure()
plt.xlabel('Time (years)')
plt.ylabel('Value')
plt.scatter(timegrid,AU_to_arcseconds(rxDM)*1e6,label='X DM , [µas]',s=10)
plt.scatter(timegrid,AU_to_arcseconds(rxPN)*1e6,label='X PN , [µas]',s=10)
plt.legend()


#Plot effects of DM:
plt.figure()
plt.xlabel('Time (years)')
plt.ylabel('Value')
plt.scatter(timegrid,AU_to_arcseconds(ryDM)*1e6,label='Y DM , [µas]',s=10)
plt.scatter(timegrid,AU_to_arcseconds(ryPN)*1e6,label='Y PN , [µas]',s=10)
plt.legend()


#Plot effects of DM:
plt.figure()
plt.xlabel('Time (years)')
plt.ylabel('Value')
plt.scatter(timegrid,-vzDM/1000,label='VZ DM , [km/s]',s=10)
plt.scatter(timegrid,-vzPN/1000,label='VZ PN , [km/s]',s=10)
plt.legend()


