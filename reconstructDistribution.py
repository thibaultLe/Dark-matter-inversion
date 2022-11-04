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
xlim = orbitModule.get_xlim()
#Sigmoid steepness factor
k = 0.01
#Amount of dark matter shells
N = 10

#Don't change these:
#X points: 
rDM = np.linspace(0,xlim,1000)
#DM shell distances
ris = orbitModule.get_DM_distances(N)


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
    #Multiply by 100 to get percentages
    dm_guess = 100*np.array(dm_guess)
    reconmis = 100*np.array(reconmis)
    mis = 100*np.array(mis)
    
    rp = 119.52867
    ra = 1948.96214
    
    fig, ((ax11,ax12,ax13)) = plt.subplots(1,3)
    # fig.set_size_inches(19,4)
    fig.set_size_inches(14,3)
    fig.set_tight_layout(True)
    
    #Plot masses:
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
    
    ax11.scatter(ris,dm_guess,label='Initial guess',color='grey',alpha=0.5)
    
    # ax11.axvline(rp,linestyle='--',label='$r_a$ and $r_p$',color='black')
    ax11.axvline(ra,linestyle='--',color='black')
    ax11.axvline(rp,linestyle='--',color='black')
    ax11.set_xlabel("$r$ [AU]")
    ax11.set_ylabel(r'Shell mass [% of $M_\bullet$]')
    ax11.set_xlim(0,2100)
    ax11.legend(loc='best')

    
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
    ax12.set_xlabel('$r$ [AU]')
    ax12.set_ylabel(r'Enclosed mass [% of $M_\bullet$]')
    # plt.plot(rDM,enclosedMassPlum(rDM),label='Plum model')
    ax12.plot(rDM,sumRisTrue,label='True')
    ax12.plot(rDM,sumRis,label='Reconstructed')
    ax12.plot(rDM,sumRisInit,label='Initial guess',color='grey',alpha=0.5)
    # ax12.axvline(rp,linestyle='--',label='$r_a$ and $r_p$',color='black')
    ax12.axvline(ra,linestyle='--',color='black')
    ax12.axvline(rp,linestyle='--',color='black')
    ax12.legend()
    ax12.set_xlim(0,2100)
    
    
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
    
    dens = mis/(100*vols)
    recondens = reconmis/(100*vols)
    initdens = dm_guess/(100*vols)
    
    
    M_0, D_0, T_0 = orbitModule.getBaseUnitConversions()
    dens = dens * M_0 / (D_0**3)
    recondens = recondens * M_0 / (D_0**3)
    initdens = initdens * M_0 / (D_0**3)
    
    ax13.scatter(ris,dens,label='True')
    ax13.scatter(ris,recondens,label='Reconstructed')
    
    # if 0 not in initdens:
    ax13.scatter(ris,initdens,label='Initial guess',color='grey',alpha=0.5)
    # ax13.axvline(rp,linestyle='--',label='$r_a$ and $r_p$',color='black')
    ax13.axvline(ra,linestyle='--',color='black')
    ax13.axvline(rp,linestyle='--',color='black')
    # ax13.set_ylabel('Density [MBH masses/(AU³)')
    ax13.set_ylabel('Density [kg/$m^3$]')
    ax13.set_xlabel('$r$ [AU]')
    ax13.set_yscale('log')
    # ax13.set_ylim(1e-15)
    ax13.set_xlim(0,2100)
    ax13.legend()
    
    
    # ax11.set_title('Mass')
    # ax12.set_title('Enclosed mass')
    # ax13.set_title('Density')
    



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
    
    # noiseLevel = 0.00005
    # np.random.seed(42)
    # noise = np.random.normal(0,noiseLevel,len(mis))
    # dm_guess = mis.copy() + noise
    # dm_guess = [0 if i < 0 else i for i in dm_guess]
    
    #Times of observation in [seconds/T_0]
    M_0, D_0, T_0 = orbitModule.getBaseUnitConversions()
    
    
    obstimes =  orbitModule.getObservationTimes(nbrOfOrbits=1)
    
    
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
    # reconstructAllDatasets(noisefactor=1e-1)
    
    #Possible names: Plummer,BahcallWolf,Sinusoidal,Uniform,ReversedPlummer,ConstantDensity
    reconstructFromTrueMasses(noisefactor = 1,name='Plummer')
    
    #For different noise samples, check the robustness of the reconstruction:
    # checkRobustnessToNoise(noisefactor=1e-1,amountOfRecons=5,name='Plummer')
    
    #Compare equi-temporal spacing vs equi-spatial spacing
    # compareDifferentTimeGrids(noisefactor = 1e-1,name='Plummer')
    
    #Start from the Plummer distribution, reconstruct on Bahcall observations and vice versa
    # comparePlummer_BahcallWolfReconstruction(noisefactor=1e-5)
    
    #Look at the loss landscape for different distributions
    # orbitModule.lossLandscape(N=10,noisefactor=1e-1,nbrOfDistributions=500000)
    
    #Calculate the variance of the true loss wrt different noise samples
    # checkLossVariance(noisefactor=1e-2)