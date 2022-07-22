# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 14:56:35 2022

@author: Thibault
"""

import numpy as np
from matplotlib.pylab import plt
import orbitModule


M_0, D_0, T_0 = orbitModule.getBaseUnitConversions()

def enclosedMass(a,rho0):
    return (4 * a**3 * np.pi * r0**3 * rho0) / ( 3 * (a**2 + r0**2)**(3/2))


IC = orbitModule.get_S2_IC()
t_grid =  np.append(0,(np.linspace(0,16.056,228) * 365.25 * 24 * 60**2 /T_0 ) + 84187.772)
    

#Dark matter:
#Amount of mascons:
N = 20

#Set dark matter distribution
#AU limit
xlim = 3000
#Amount of points in linspace
n = 1000
#Bahcall-Wolf cusp model:
rDM = np.linspace(0,xlim,n)

r0 = 2474.01
rho0plum = 1.69*10**(-10) * (D_0**3) / M_0
# rho0cusp = 2.24*10**(-11) * (D_0**3) / M_0

#Plummer model:
rhoPlum = rho0plum *( 1. + ((rDM**2) / (r0**2)))**(-5/2)
#Cusp model:
# rhoCusp = rho0cusp * (rDM / r0)**(-7/4)

#Convert enclosed mass to mascons
x_right = rDM[::round(n/(N+1))] # ri's of mascon shells
y_right = enclosedMass(rDM,rho0plum)[::round(n/(N+1))] # enclosed mi's of mascon shells

#mascon masses = difference in enclosed mass:
ris = x_right[1:]
mis = [t - s for s, t in zip(y_right, y_right[1:])]


#Plot enclosed mass
rp = 119.52867
ra = 1948.96214
plt.figure()
plt.xlabel('Distance from MBH [AU]')
plt.plot(rDM,enclosedMass(rDM,rho0plum),label='Plum model')
plt.ylabel('enclosed mass [MBH masses]')
plt.axvline(ra,linestyle='--',label='rp and ra',color='black')
plt.axvline(rp,linestyle='--',color='black')
plt.scatter(x_right[1:],y_right[1:],color='orange',label='Mascon enclosed mass')
plt.bar(x_right,y_right,width=(xlim)/(N+1),alpha=0.2,align='edge',edgecolor='orange',color='orange')
plt.legend()

#Plot mass of mascons
plt.figure()
plt.scatter(ris,mis,label='mascons',color='orange')
plt.axvline(rp,linestyle='--',label='rp and ra',color='black')
plt.axvline(ra,linestyle='--',color='black')
plt.ylabel('Mass [MBH masses]',color='orange')
plt.xlabel('Distance from MBH [AU]')
plt.legend()


#TODO: sinusoidal, uniform, etc distributions

[rxDM,ryDM,rzDM] , [vxDM,vyDM,vzDM] = orbitModule.simulateOrbitsCartesian(True, IC, mis, ris, t_grid)


#Plot position and MBH
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(rxDM[1:], ryDM[1:], rzDM[1:], label='Position')
# ax.scatter(rxDM[0], ryDM[0], rzDM[0], label='Start',color='lawngreen')
# ax.scatter(rxDM[-1], ryDM[-1], rzDM[-1], label='End',color="red")
# ax.scatter(0,0,0,color='black',label="MBH")
# ax.set_xlabel('rX')
# ax.set_ylabel('rY')
# ax.set_zlabel('rZ')

# #Plot DM shells (3D spheres)
# u = np.linspace(0, 2 * np.pi, 20)
# v = np.linspace(0, np.pi, 20)

# x_sphere = 1 * np.outer(np.cos(u), np.sin(v))
# y_sphere = 1 * np.outer(np.sin(u), np.sin(v))
# z_sphere = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

# for i in range(N):
#     #Only plot if dark matter mass is not zero
#     if mis[i] != 0:
#         surf =ax.plot_surface(ris[i]*x_sphere, ris[i]*y_sphere, ris[i]*z_sphere,  \
#             rstride=1, cstride=1, color='black', linewidth=0, alpha=0.02,label='DM shell(s)')
#         surf._facecolors2d = surf._facecolor3d
#         surf._edgecolors2d = surf._edgecolor3d
        
# plotlim = (max(max(abs(rxDM)),max(abs(ryDM)),max(abs(rzDM))))
# ax.set_xlim(-plotlim,plotlim)
# ax.set_ylim(-plotlim,plotlim)
# ax.set_zlim(-plotlim,plotlim)
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys())
# plt.show()





