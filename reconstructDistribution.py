# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 16:21:03 2022

@author: Thibault
"""

import numpy as np
from matplotlib.pylab import plt
import orbitModule




def reconstructDistribution(PNCORRECTION,mis,ris, ic_guess, dm_guess, CARTESIANOBS = True,OBS3 = True):
    """
    

    Parameters
    ----------
    PNCORRECTION : boolean
        True if using 1PN correction
    mis : list of floats
        initial guesses of masses of dark matter shells in MBH masses
    ris : list of floats
        distances of dark matter shells in AU.
    ic_guess : list of floats
        the initial guess for the initial conditions
    dm_guess : list of floats
        the initial guess for the dark matter masses
    CARTESIANOBS : boolean, optional
        True if using cartesian observations instead of orbital parameter observations. The default is True.
    OBS3 : boolean, optional
        True if only using 3 observed parameters (first, second and last = x,y and vz for cartesian). The default is True.

    Raises
    ------
    RuntimeError
        If the length of dark matter masses and distances do not match, an error is raised.

    Returns
    -------
    list of floats
        list of reconstructed dark matter masses.

    Reconstructs dark matter distribution starting from an initial guess
    """
    
    if len(mis) != len(ris):
        raise RuntimeError("Lengths of DM masses and distances does not match")
        
    N = len(mis)
    
    ta = orbitModule.buildTaylorIntegrator(PNCORRECTION, N)
    
    
    np.set_printoptions(precision=5)
    
    M_0, D_0, T_0 = orbitModule.getBaseUnitConversions()
    
    #Time of observation
    # last_time = 293097.9510676383
    # t_obslist = [last_time,last_time*1.1,last_time*1.2]
    
    t_obslist =  np.append(0,(np.linspace(0,16.056,228) * 365.25 * 24 * 60**2 /T_0 ) + 84187.772)
    # t_obslist = np.linspace(0,16.056,228) * 365.25 * 24 * 60**2 /T_0 
    
        
    IC = orbitModule.get_S2_IC()
        
    #Setup for fake reconstruction:
    ta.state[:6] = IC
    ta.time = 0
    ta.pars[:N] = mis
    ta.pars[N:] = ris
    out = ta.propagate_grid(t_obslist)
    
    
    observationlist = np.asarray(out[4][:,[0,1,2,3,4,5]]).copy()
    
    
    if CARTESIANOBS:
        observationlist = orbitModule.convertToCartesian(observationlist[:,0], observationlist[:,1], observationlist[:,2],\
                observationlist[:,3], observationlist[:,4], observationlist[:,5])
        
        if OBS3:
            observationlist = np.array(observationlist)
            observationlist =  observationlist[[0,1,-1],:]
    
        observationlist = np.transpose(observationlist)
    
    #Convert time to years
    _, _, T_0 = orbitModule.getBaseUnitConversions()
    timegrid = 2.010356112597776246e+03 + t_obslist * T_0 / (365.25 * 24 * 60**2 )
    
    observationlist = np.column_stack((timegrid, observationlist))
    
    
    #observationlist =[[t1 x1 y1 ... vz1], [t2 x2 y2 ... vz2],...[]]
    return orbitModule.reconstructDistribution(observationlist, ic_guess, dm_guess,CARTESIANOBS,OBS3)
    

def reconstructFromFile(filename,ic_guess,dm_guess):
    M_0, D_0, T_0 = orbitModule.getBaseUnitConversions()
    observations = np.loadtxt(filename)
    #Observations are given in time [yr] Y [arcsec] X [arcsec] VZ [km/s]
    
    timegrid = observations[:,0]
    rYs = orbitModule.arcseconds_to_AU(observations[:,1])
    rXs = orbitModule.arcseconds_to_AU(observations[:,2])
    vZs = observations[:,3] * 1000 * T_0 / D_0 

    
    observationlist = np.column_stack((timegrid, rXs, rYs, vZs))
    
    return orbitModule.reconstructDistribution(observationlist, ic_guess, dm_guess,CARTESIANOBS=True,OBS3=True)
    
    


#20 mascons plummer
mis, ris = orbitModule.get_Plummer_DM(20,3000)

mis = 1*np.array(mis)

IC = orbitModule.get_S2_IC()
ic_guess = IC
# ic_guess = np.multiply(IC, len(IC)*[1.000001])

# dm_guess = len(mis)*[0]
# dm_guess = len(mis)*[0.00008]
# dm_guess = mis
dm_guess = 1.001*np.array(mis)

# reconic, reconmis = reconstructDistribution(True,mis,ris,ic_guess,dm_guess, \
#                                     CARTESIANOBS = True,OBS3 = True)
    
reconic, reconmis = reconstructFromFile('Datasets/Plummer_N=20.txt',ic_guess,dm_guess)


rp = 119.52867
ra = 1948.96214
plt.figure()

plt.scatter(ris,mis,label='Plummer model')
plt.scatter(ris,dm_guess,label='Initial guess',color='grey')
plt.scatter(ris,reconmis,label='Reconstructed')
plt.axvline(-rp,linestyle='--',label='rp and ra',color='black')
plt.axvline(ra,linestyle='--',color='black')
plt.xlabel("Distance from MBH [AU]")
plt.ylabel("Mass [MBH masses]")
plt.title('Reconstructed dark matter distribution')
plt.legend()




M_0, D_0, T_0 = orbitModule.getBaseUnitConversions()
xlim = 3000
#Amount of points in linspace
n = 1000
#Bahcall-Wolf cusp model:
rDM = np.linspace(0,xlim,n)

rho0plum = 1.69*10**(-10) * (D_0**3) / M_0
# rho0cusp = 2.24*10**(-11) * (D_0**3) / M_0
r0 = 2474.01


N = len(mis)
k = 0.1

    
# Mascon model (mi, ri), sigmoid approximation of step function
listOfSigs = [0.5 + 0.5 * np.tanh( k * (rDM - ris[i])) for i in range(N)]

listOfRis = [reconmis[i]* listOfSigs[i] for i in range(N)]

suml = listOfSigs[0]
for i in range(1,len(listOfSigs)):
    suml = suml + listOfSigs[i]
    
sumRis = listOfRis[0]
for i in range(1,len(listOfRis)):
    sumRis = sumRis + listOfRis[i]
    

listOfRis = [mis[i]* listOfSigs[i] for i in range(N)]
suml = listOfSigs[0]
for i in range(1,len(listOfSigs)):
    suml = suml + listOfSigs[i]
    
sumRisTrue = listOfRis[0]
for i in range(1,len(listOfRis)):
    sumRisTrue = sumRisTrue + listOfRis[i]
    
    
listOfRis = [dm_guess[i]* listOfSigs[i] for i in range(N)]
suml = listOfSigs[0]
for i in range(1,len(listOfSigs)):
    suml = suml + listOfSigs[i]
    
sumRisInit = listOfRis[0]
for i in range(1,len(listOfRis)):
    sumRisInit = sumRisInit + listOfRis[i]

# def enclosedMass(a,rho0):
#     return (4 * a**3 * np.pi * r0**3 * rho0) / ( 3 * (a**2 + r0**2)**(3/2))

#Plot enclosed mass
plt.figure()
plt.xlabel('Distance from MBH [AU]')
plt.ylabel('Enclosed mass [MBH masses]')
# plt.plot(rDM,enclosedMass(rDM,rho0plum),label='Plum model')
plt.plot(rDM,sumRisInit,label='Initial guess',color='grey')
plt.plot(rDM,sumRisTrue,label='True')
plt.plot(rDM,sumRis,label='Reconstructed')
plt.axvline(rp,linestyle='--',label='rp and ra',color='black')
plt.axvline(ra,linestyle='--',color='black')
# plt.scatter(ris,np.cumsum(reconmis),label='Mascon enclosed mass',color='orange')
# plt.bar(ris,np.cumsum(reconmis),width=(xlim)/(N),alpha=0.2,align='edge',edgecolor='orange',color='orange')
plt.legend()
plt.title('Enclosed mass')


#Plot difference in enclosed mass
plt.figure()
plt.xlabel('Distance from MBH [AU]')
plt.ylabel('Enclosed mass [MBH masses]')
plt.plot(rDM,sumRis - sumRisTrue,label='Reconstructed - True')
plt.axvline(rp,linestyle='--',label='rp and ra',color='black')
plt.axvline(ra,linestyle='--',color='black')
# plt.scatter(ris,np.cumsum(reconmis),label='Mascon enclosed mass',color='orange')
# plt.bar(ris,np.cumsum(reconmis),width=(xlim)/(N),alpha=0.2,align='edge',edgecolor='orange',color='orange')
plt.legend()
plt.title('Difference in enclosed mass')






