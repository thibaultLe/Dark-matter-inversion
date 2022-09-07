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
xlim = 2500


#Plummer
mis,ris = orbitModule.get_Plummer_DM(N, xlim)

rx,ry,rz,vx,vy,vz = orbitModule.simulateOrbitsCartesian(True, IC, mis, ris, t_grid)
rx, ry, vz = orbitModule.convertXYVZtoArcsec(rx, ry, vz)
data = np.column_stack((timegrid,ry,rx,vz))
np.savetxt('Datasets/Plummer_N={}.txt'.format(N),data)
#format: [[t0, y0, x0, vz0],
#         [t1, y1, x1, vz1],...]


#Bahcall-Wolf
mis,ris = orbitModule.get_BahcallWolf_DM(N, xlim)

rx,ry,rz,vx,vy,vz = orbitModule.simulateOrbitsCartesian(True, IC, mis, ris, t_grid)
rx, ry, vz = orbitModule.convertXYVZtoArcsec(rx, ry, vz)
data = np.column_stack((timegrid,ry,rx,vz))
np.savetxt('Datasets/BahcallWolf_N={}.txt'.format(N),data)

#Uniform
mis,ris = orbitModule.get_Uniform_DM(N, xlim)

rx,ry,rz,vx,vy,vz = orbitModule.simulateOrbitsCartesian(True, IC, mis, ris, t_grid)
rx, ry, vz = orbitModule.convertXYVZtoArcsec(rx, ry, vz)
data = np.column_stack((timegrid,ry,rx,vz))
np.savetxt('Datasets/Uniform_N={}.txt'.format(N),data)


#Uniform
mis,ris = orbitModule.get_Sinusoidal_DM(N, xlim)

rx,ry,rz,vx,vy,vz = orbitModule.simulateOrbitsCartesian(True, IC, mis, ris, t_grid)
rx, ry, vz = orbitModule.convertXYVZtoArcsec(rx, ry, vz)
data = np.column_stack((timegrid,ry,rx,vz))
np.savetxt('Datasets/Sinusoidal_N={}.txt'.format(N),data)






