# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 16:21:03 2022

@author: Thibault
"""

import os
import numpy as np
from matplotlib.pylab import plt
import orbitModule

#Max x limit in [AU]
xlim = 3000
#Amount of points in linspace
n = 1000
#X points:
rDM = np.linspace(0,xlim,n)
#Sigmoid steepness factor
k = 0.1
#Amount of dark matter shells
N = 20

ris = orbitModule.get_DM_distances(N, xlim)


def getEnclosedMass(mis):
    # Mascon model (mi, ri), sigmoid approximation of step function
    listOfSigs = [0.5 + 0.5 * np.tanh( k * (rDM - ris[i])) for i in range(N)]

    listOfRis = [mis[i]* listOfSigs[i] for i in range(N)]

    suml = listOfSigs[0]
    for i in range(1,N):
        suml = suml + listOfSigs[i]
        
    sumRis = listOfRis[0]
    for i in range(1,N):
        sumRis = sumRis + listOfRis[i]
    
    return sumRis

def plotInitReconTrueMasses(dm_guess,reconmis,mis):
    rp = 119.52867
    ra = 1948.96214
    
    fig, ((ax11,ax12,ax13)) = plt.subplots(1,3)
    fig.set_size_inches(19,4)
    fig.set_tight_layout(True)
    
    #Plot masses:
    # plt.figure()
    ax11.scatter(ris,mis,label='True')
    ax11.scatter(ris,dm_guess,label='Initial guess',color='grey',alpha=0.5)
    ax11.scatter(ris,reconmis,label='Reconstructed')
    ax11.axvline(rp,linestyle='--',label='rp and ra',color='black')
    ax11.axvline(ra,linestyle='--',color='black')
    ax11.set_xlabel("Distance from MBH [AU]")
    ax11.set_ylabel("Mass [MBH masses]")
    ax11.set_title('Reconstructed dark matter distribution')
    ax11.legend()
    
    #Plot enclosed mass:
    sumRis = getEnclosedMass(reconmis)
    sumRisTrue = getEnclosedMass(mis)
    sumRisInit = getEnclosedMass(dm_guess)
    
    
    def enclosedMassPlum(a):
        M_0, D_0, T_0 = orbitModule.getBaseUnitConversions()
        rho0plum = 1.69*10**(-10) * (D_0**3) / M_0
        r0 = 2474.01
        return (4 * a**3 * np.pi * r0**3 * rho0plum) / ( 3 * (a**2 + r0**2)**(3/2))
    
    def enclosedMassCusp(a):
        M_0, D_0, T_0 = orbitModule.getBaseUnitConversions()
        r0 = 2474.01
        rho0cusp = 2.24*10**(-11) * (D_0**3) / M_0
        return (4 * a**3 * np.pi * (a/r0)**(-7/4) * rho0cusp) / (3 - (7/4))
    
    # plt.figure()
    ax12.set_xlabel('Distance from MBH [AU]')
    ax12.set_ylabel('Enclosed mass [MBH masses]')
    # plt.plot(rDM,enclosedMassPlum(rDM),label='Plum model')
    ax12.plot(rDM,sumRisInit,label='Initial guess',color='grey')
    ax12.plot(rDM,sumRisTrue,label='True')
    ax12.plot(rDM,sumRis,label='Reconstructed')
    ax12.axvline(rp,linestyle='--',label='rp and ra',color='black')
    ax12.axvline(ra,linestyle='--',color='black')
    ax12.legend()
    ax12.set_title('Enclosed mass')
    
    
    #Plot difference in enclosed mass
    # plt.figure()
    ax13.set_xlabel('Distance from MBH [AU]')
    ax13.set_ylabel('Enclosed mass [MBH masses]')
    ax13.plot(rDM,sumRis - sumRisTrue,label='Reconstructed - True')
    ax13.axvline(rp,linestyle='--',label='rp and ra',color='black')
    ax13.axvline(ra,linestyle='--',color='black')
    ax13.legend()
    ax13.set_title('Difference in enclosed mass')
    



def comparePlummer_BahcallWolfReconstruction(noisefactor):
    #Plummer observations, Bahcall initial guess
    ic_guess = orbitModule.get_S2_IC()
    dm_guess, _ = orbitModule.get_BahcallWolf_DM(N,xlim)
    filename = 'Datasets/Plummer_N={}.txt'.format(N)
    reconic, reconmis = orbitModule.reconstructFromFile(filename,ic_guess,dm_guess, \
                                                        ADD_NOISE = True, noisefactor = noisefactor)
    
    mis, ris = orbitModule.get_Plummer_DM(N,xlim)
    plotInitReconTrueMasses(dm_guess,reconmis,mis)
    
    #Bahcall observations, Plummer initial guess
    ic_guess = orbitModule.get_S2_IC()
    dm_guess, _ = orbitModule.get_Plummer_DM(N,xlim)
    filename = 'Datasets/BahcallWolf_N={}.txt'.format(N)
    reconic, reconmis = orbitModule.reconstructFromFile(filename,ic_guess,dm_guess, \
                                                        ADD_NOISE = True, noisefactor = noisefactor)
    
    mis, ris = orbitModule.get_BahcallWolf_DM(N,xlim)
    plotInitReconTrueMasses(dm_guess,reconmis,mis)

def reconstructAllDatasets():
    for entry in os.scandir("./Datasets/"):       
        # print(entry.path)      
        # print(entry.name)
        filepath = entry.path
        filename = entry.name
        
        if filename.endswith(".txt") and "_" in filename: 
            N = int(filename.split('.')[0].split('=')[1])
            name = filename.split('_')[0]
            print('Reconstructing',filename)
            ic_guess = orbitModule.get_S2_IC()
            dm_guess = N*[0]
            reconic, reconmis = orbitModule.reconstructFromFile(filepath,ic_guess,dm_guess, \
                                                            ADD_NOISE = False, noisefactor = 1)
            getTrueDM = getattr(orbitModule,'get_'+name+'_DM')
            mis, ris = getTrueDM(N,xlim)
            plotInitReconTrueMasses(dm_guess,reconmis,mis)
            continue
        else:
            continue

def reconstructFromTrueMasses():
    # mis, ris = orbitModule.get_Plummer_DM(N,xlim)
    # mis, ris = orbitModule.get_BahcallWolf_DM(N,xlim)
    mis,ris = orbitModule.get_Uniform_DM(N, xlim)
    # mis,ris = orbitModule.get_Sinusoidal_DM(N, xlim)
    # mis = 1*np.array(mis)
    
    IC = orbitModule.get_S2_IC()
    ic_guess = IC
    # ic_guess = np.multiply(IC, len(IC)*[1.000001])
    
    dm_guess = N*[0]
    # dm_guess,_ = orbitModule.get_Uniform_DM(N, xlim)
    # dm_guess = 0.5*np.array(mis)
    
    #Times of observation in [seconds/T_0]
    _, _, T_0 = orbitModule.getBaseUnitConversions()
    
    obstimes =  np.append(0,(np.linspace(0,16.056,228) * 365.25 * 24 * 60**2 /T_0 ) + 84187.772)
    # obstimes = np.linspace(0,16.056,228) * 365.25 * 24 * 60**2 /T_0 
    
    reconic, reconmis = orbitModule.reconstructDistributionFromTrueMasses(True,mis,ris,obstimes, \
                                                  ic_guess,dm_guess, CARTESIANOBS = True,OBS3 = True)
        
    
    plotInitReconTrueMasses(dm_guess,reconmis,mis)



"""
#Uncomment functions here to use them:
"""

if __name__ == "__main__":
    # reconstructAllDatasets()
    reconstructFromTrueMasses()
    # comparePlummer_BahcallWolfReconstruction(noisefactor=1e-1)
    # comparePlummer_BahcallWolfReconstruction(noisefactor=5e-1)
    # comparePlummer_BahcallWolfReconstruction(noisefactor=1)