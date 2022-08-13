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
    
#Convert time to years
_, _, T_0 = orbitModule.getBaseUnitConversions()
timegrid = 2.010356112597776246e+03 + t_grid * T_0 / (365.25 * 24 * 60**2 )

#Dark matter:
#Amount of mascons:
N = 20

#Set dark matter distribution
#AU limit
xlim = 3000


#TODO: sinusoidal, uniform, etc distributions and save them
#TODO: add noise (perhaps after loading the data)




mis,ris = orbitModule.get_Plummer_DM(N, xlim)

rx,ry,rz,vx,vy,vz = orbitModule.simulateOrbitsCartesian(True, IC, mis, ris, t_grid)
# Units of distance: [AU], units of speed: [m/s]
rx, ry, vz = orbitModule.convertXYVZtoArcsec(rx, ry, vz)

#format: [[t0, y0, x0, vz0],
#         [t1, y1, x1, vz1],...]
#Save dataset
data = np.column_stack((timegrid,ry,rx,vz))
np.savetxt('Datasets/Plummer_N={}.txt'.format(N),data)



mis,ris = orbitModule.get_BahcallWolf_DM(N, xlim)

rx,ry,rz,vx,vy,vz = orbitModule.simulateOrbitsCartesian(True, IC, mis, ris, t_grid)
rx, ry, vz = orbitModule.convertXYVZtoArcsec(rx, ry, vz)

#Save dataset
data = np.column_stack((timegrid,ry,rx,vz))
np.savetxt('Datasets/BahcallWolf_N={}.txt'.format(N),data)






