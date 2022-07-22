# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 14:56:35 2022

@author: Thibault
"""

import numpy as np
from matplotlib.pylab import plt
import orbitModule


M_0, D_0, T_0 = orbitModule.getBaseUnitConversions()

IC = orbitModule.get_S2_IC()
t_grid =  np.append(0,(np.linspace(0,16.056,228) * 365.25 * 24 * 60**2 /T_0 ) + 84187.772)
    

#Dark matter:
#Amount of mascons:
N = 20

#Set dark matter distribution
#AU limit
xlim = 3000

mis,ris = orbitModule.get_Plummer_DM(N, xlim)

#Plot mass of mascons
rp = 119.52867
ra = 1948.96214
plt.figure()
plt.scatter(ris,mis,label='mascons',color='orange')
plt.axvline(rp,linestyle='--',label='rp and ra',color='black')
plt.axvline(ra,linestyle='--',color='black')
plt.ylabel('Mass [MBH masses]',color='orange')
plt.xlabel('Distance from MBH [AU]')
plt.legend()


#TODO: sinusoidal, uniform, etc distributions and save them

rxDM,ryDM,rzDM ,vxDM,vyDM,vzDM = orbitModule.simulateOrbitsCartesian(True, IC, mis, ris, t_grid)







