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
xlim = 2100
#Sigmoid steepness factor
k = 0.01
#Amount of dark matter shells
N = 10

#Don't change these:
#X points: 
rDM = np.linspace(0,xlim,1000)
#DM shell distances
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

def plotInitReconTrueMasses(dm_guess,reconmis,mis,stddevs=[]):
    rp = 119.52867
    ra = 1948.96214
    
    fig, ((ax11,ax12,ax13)) = plt.subplots(1,3)
    fig.set_size_inches(19,4)
    fig.set_tight_layout(True)
    
    #Plot masses:
    # plt.figure()
    ax11.scatter(ris,mis,label='True')
    if len(stddevs) > 0:
        ax11.scatter(ris,reconmis,label='Mean reconstruction') 
        for i in range(N):
            if i == 0:
                ax11.errorbar(ris[i],reconmis[i],stddevs[i],capsize=5,color='orange',label='[Standard deviation]')
                
            else:
                ax11.errorbar(ris[i],reconmis[i],stddevs[i],capsize=5,color='orange')
                
        ax11.set_ylim(0,max(1.2*max(reconmis),1.2*max(mis)))
        
    else:
        ax11.scatter(ris,reconmis,label='Reconstructed')
    
    ax11.scatter(ris,dm_guess,label='Initial guess',color='lightgrey',alpha=0.5)
    
    # misPlum,_ = orbitModule.get_Plummer_DM(N, xlim)
    # ax11.scatter(ris,misPlum,label='Plummer')
    
    ax11.axvline(rp,linestyle='--',label='rp and ra',color='black')
    ax11.axvline(ra,linestyle='--',color='black')
    ax11.set_xlabel("Distance from MBH [AU]")
    ax11.set_ylabel("Mass [MBH masses]")
    ax11.set_title('Mass')
    # ax11.legend()

    
   
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
    ax12.plot(rDM,sumRisTrue,label='True')
    ax12.plot(rDM,sumRis,label='Reconstructed')
    ax12.plot(rDM,sumRisInit,label='Initial guess',color='grey')
    ax12.axvline(rp,linestyle='--',label='rp and ra',color='black')
    ax12.axvline(ra,linestyle='--',color='black')
    ax12.legend()
    ax12.set_title('Enclosed mass')
    
    
    #Plot difference in enclosed mass
    # plt.figure()
    # ax13.set_xlabel('Distance from MBH [AU]')
    # ax13.set_ylabel('Enclosed mass [MBH masses]')
    # ax13.plot(rDM,sumRis - sumRisTrue,label='Reconstructed - True')
    # ax13.axvline(rp,linestyle='--',label='rp and ra',color='black')
    # ax13.axvline(ra,linestyle='--',color='black')
    # ax13.legend()
    # ax13.set_title('Difference in enclosed mass')
    
    
    #Plot density:
    vols = 4*np.pi*(np.array(ris)**3)/3
    for i in range(1,len(vols)):
        vols[i] = vols[i] - vols[i-1]
    
    dens = mis/vols
    recondens = reconmis/vols
    initdens = dm_guess/vols
    
    ax13.scatter(ris,dens,label='True')
    ax13.scatter(ris,recondens,label='Reconstructed')
    ax13.scatter(ris,initdens,label='Initial guess',color='grey')
    ax13.axvline(rp,linestyle='--',label='rp and ra',color='black')
    ax13.axvline(ra,linestyle='--',color='black')
    ax13.set_ylabel('Density [MBH masses/(AU³)')
    ax13.set_xlabel('Distance from MBH [AU]')
    ax13.set_title('Density')
    ax13.set_yscale('log')
    ax13.set_ylim(1e-15)
    
    ax13.legend()
    



def comparePlummer_BahcallWolfReconstruction(noisefactor):
    #Plummer observations, Bahcall initial guess
    ic_guess = orbitModule.get_S2_IC()
    dm_guess, _ = orbitModule.get_BahcallWolf_DM(N,xlim)
    filename = 'Datasets/Plummer_N={}.txt'.format(N)
    reconic, reconmis = orbitModule.reconstructFromFile(filename,ic_guess,dm_guess, \
                                        noisefactor = noisefactor)
    
    mis, ris = orbitModule.get_Plummer_DM(N,xlim)
    plotInitReconTrueMasses(dm_guess,reconmis,mis)
    
    #Bahcall observations, Plummer initial guess
    ic_guess = orbitModule.get_S2_IC()
    dm_guess, _ = orbitModule.get_Plummer_DM(N,xlim)
    filename = 'Datasets/BahcallWolf_N={}.txt'.format(N)
    reconic, reconmis = orbitModule.reconstructFromFile(filename,ic_guess,dm_guess, \
                                        noisefactor = noisefactor)
    
    mis, ris = orbitModule.get_BahcallWolf_DM(N,xlim)
    plotInitReconTrueMasses(dm_guess,reconmis,mis)

def reconstructAllDatasets(noisefactor=1):
    for entry in os.scandir("./Datasets/"):    
        filepath = entry.path
        filename = entry.name
        
        if filename.endswith(".txt") and "_" in filename: 
            Nf = int(filename.split('.')[0].split('=')[1])
            if Nf == N:
                name = filename.split('_')[0]
                # if name == 'Sinusoidal':
                #     continue
                print('\nReconstructing',filename)
                
                getTrueDM = getattr(orbitModule,'get_'+name+'_DM')
                mis, ris = getTrueDM(N,xlim)
                
                ic_guess = orbitModule.get_S2_IC()
                
                
                print('Starting from 0')
                dm_guess = Nf*[0]
                # print('Starting from true masses')
                # dm_guess = mis.copy()
                # print('Starting from Plummer')
                # dm_guess, _ = orbitModule.get_Plummer_DM(N, xlim)
                # print('Starting from Random IG')
                # np.random.seed(0)
                # noiseLevel = 0.5*max(mis)
                # noise = np.random.normal(0,noiseLevel,len(mis))
                # dm_guess = mis.copy() + noise
                
                
                reconic, reconmisInit = orbitModule.reconstructFromFile(filepath,ic_guess,dm_guess, \
                            noisefactor = noisefactor,seed=0)
               
                
                plotInitReconTrueMasses(dm_guess,reconmisInit,mis)
                
                # print(list(reconmisInit))
                
                #0 noise -> 5000+ iterations/loss threshold
                #1e-2 -> 1200 iterations
                #1 -> 200 iterations?
                
            
                
                
                # break
        else:
            continue


def reconstructFromTrueMasses(noisefactor = 0,name='Plummer'):
    getTrueDM = getattr(orbitModule,'get_'+name+'_DM')
    mis, ris = getTrueDM(N,xlim)
    
    ic_guess = orbitModule.get_S2_IC()
    
    dm_guess = N*[0]
    
    #Times of observation in [seconds/T_0]
    M_0, D_0, T_0 = orbitModule.getBaseUnitConversions()
    
    # obstimes =  np.append(0,(np.linspace(0,16.056740695411154,228) * 365.25 * 24 * 60**2 /T_0 ) + 84187.772)
   
    obstimes =  orbitModule.getObservationTimes()
    
    # obstimes =  np.linspace(0,16.056740695411154,300) * 365.25 * 24 * 60**2 /T_0  + 84187.772
    # print(obstimes)
    
    reconic, reconmis = orbitModule.reconstructDistributionFromTrueMasses(True,mis,ris,obstimes, \
                              ic_guess,dm_guess, CARTESIANOBS = True,OBS3 = True, \
                              noisefactor=noisefactor,seed=0)
        
    
    plotInitReconTrueMasses(dm_guess,reconmis,mis)
    
    
def checkRobustnessToNoise(noisefactor,amountOfRecons,name):
    getTrueDM = getattr(orbitModule,'get_'+name+'_DM')
    mis, ris = getTrueDM(N,xlim)
    
    IC = orbitModule.get_S2_IC()
    ic_guess = IC
    
    dm_guess = N*[0]
    
    #Times of observation in [seconds/T_0]
    M_0, D_0, T_0 = orbitModule.getBaseUnitConversions()
    
    obstimes =  orbitModule.getObservationTimes()
    
    reconmises = []
    for i in range(amountOfRecons):
        reconic, reconmis = orbitModule.reconstructDistributionFromTrueMasses(True,mis,ris,obstimes, \
                              ic_guess,dm_guess, CARTESIANOBS = True,OBS3 = True, \
                              noisefactor = noisefactor,seed=i)
        reconmises.append(reconmis)
    
    mean = np.mean(reconmises,axis=0)
    stddevs = np.std(reconmises,axis=0)
    
    plotInitReconTrueMasses(dm_guess,mean,mis,stddevs)


def compareDifferentTimeGrids(noisefactor = 0,name='Plummer'):
    getTrueDM = getattr(orbitModule,'get_'+name+'_DM')
    mis, ris = getTrueDM(N,xlim)
    
    IC = orbitModule.get_S2_IC()
    ic_guess = IC
    
    dm_guess = N*[0]
    
    #Times of observation in [seconds/T_0]
    M_0, D_0, T_0 = orbitModule.getBaseUnitConversions()
    
    txtfile = 'Datasets/1PN.txt'
    comparedData = np.loadtxt(txtfile)
    obstimes = comparedData[:,0]
    
    reconic, reconmis = orbitModule.reconstructDistributionFromTrueMasses(True,mis,ris,obstimes, \
                              ic_guess,dm_guess, CARTESIANOBS = True,OBS3 = True, \
                              noisefactor=noisefactor,seed=0)
        
    plotInitReconTrueMasses(dm_guess,reconmis,mis)
    
    
    obstimes =  orbitModule.getObservationTimes()
    
    reconic, reconmis = orbitModule.reconstructDistributionFromTrueMasses(True,mis,ris,obstimes, \
                              ic_guess,dm_guess, CARTESIANOBS = True,OBS3 = True, \
                              noisefactor=noisefactor,seed=0)
        
    plotInitReconTrueMasses(dm_guess,reconmis,mis)
    

def checkLossVariance(noisefactor=1e-1):
    mis, ris = orbitModule.get_Plummer_DM(N,xlim)
    
    truelosses = orbitModule.lossesForDifferentNoiseProfiles(mis,noisefactor)
    
    means = np.mean(truelosses,axis=0)
    stddevs = np.std(truelosses,axis=0)

    plt.figure()
    plt.ylabel('Loss')
    plt.errorbar([1], means,stddevs,fmt='o',capsize=5,label='Mean and standard deviation')
    plt.legend()
    plt.title('Loss of true masses')

"""
Set your desired parameters at the top,
 then uncomment functions here to use them:
"""

if __name__ == "__main__":
    
    #Reconstruct all datasets
    # reconstructAllDatasets(noisefactor=0)
    
    #Possible names: Plummer, BahcallWolf, Sinusoidal, Uniform,ConstantDensity,
    # ReversedPlummer
    reconstructFromTrueMasses(noisefactor = 0,name='Sinusoidal')
    
    #For different noise samples, check the robustness of the reconstruction:
    # checkRobustnessToNoise(noisefactor=1e-1,amountOfRecons=5,name='Plummer')
    
    #Compare equi-temporal spacing vs equi-spatial spacing
    # compareDifferentTimeGrids(noisefactor = 1e-1,name='Plummer')
    
    #Start from the Plummer distribution, reconstruct on Bahcall observations and vice versa
    # comparePlummer_BahcallWolfReconstruction(noisefactor=1e-5)
    
    #Look at the loss landscape for different distributions
    # orbitModule.lossLandscape(N=5,noisefactor=1e-1,nbrOfDistributions=10000)
    
    #Calculate the variance of the true loss wrt different noise samples
    # checkLossVariance(noisefactor=1e-1)