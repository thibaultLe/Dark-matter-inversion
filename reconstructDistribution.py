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
xlim = 2500
#Amount of points in linspace
n = 1000
#X points:
rDM = np.linspace(0,xlim,n)
#Sigmoid steepness factor
k = 0.01
#Amount of dark matter shells
N = 10

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

def plotInitReconTrueMasses(dm_guess,reconmis,mis,STD=False):
    rp = 119.52867
    ra = 1948.96214
    
    fig, ((ax11,ax12,ax13)) = plt.subplots(1,3)
    fig.set_size_inches(19,4)
    fig.set_tight_layout(True)
    
    #Plot masses:
    # plt.figure()
    ax11.scatter(ris,mis,label='True')
    ax11.scatter(ris,reconmis,label='Reconstructed')
    ax11.scatter(ris,dm_guess,label='Initial guess',color='lightgrey',alpha=1)
    
    # misPlum,_ = orbitModule.get_Plummer_DM(N, xlim)
    # ax11.scatter(ris,misPlum,label='Plummer')
    
    ax11.axvline(rp,linestyle='--',label='rp and ra',color='black')
    ax11.axvline(ra,linestyle='--',color='black')
    ax11.set_xlabel("Distance from MBH [AU]")
    ax11.set_ylabel("Mass [MBH masses]")
    ax11.set_title('Mass')
    # ax11.legend()
    

    observations = np.loadtxt('Datasets/BahcallWolf_N={}.txt'.format(N))
    timegrid = observations[:,0]
    t_grid = orbitModule.convertYearsTimegridToOurFormat(timegrid)
    
    if STD:
        variance_x0, variance_DM = orbitModule.getModelUncertainty(  \
                orbitModule.get_S2_IC(), reconmis, t_grid, noisefactor=1e-2)
        
        M_0, D_0, T_0 = orbitModule.getBaseUnitConversions()
        
        # variance_DM = 20*variance_DM
        # print(variance_DM)
        for i in range(N):
            if i == 0:
                ax11.errorbar(ris[i],reconmis[i],variance_DM[i],capsize=5,color='orange',label='[Standard deviation]')
                # plt.errorbar(ris[i],0,variance_DM[i],capsize=5,color='orange',label='Variance')
       
            else:
                ax11.errorbar(ris[i],reconmis[i],variance_DM[i],capsize=5,color='orange')
                # plt.errorbar(ris[i],0,variance_DM[i],capsize=5,color='orange')
        ax11.set_ylim(0,max(1.2*max(reconmis),1.2*max(mis)))
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
    rp = 119.52867
    ra = 1948.96214
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
        filepath = entry.path
        filename = entry.name
        
        if filename.endswith(".txt") and "_" in filename: 
            Nf = int(filename.split('.')[0].split('=')[1])
            if Nf == N:
                name = filename.split('_')[0]
                print('Reconstructing',filename)
                
                getTrueDM = getattr(orbitModule,'get_'+name+'_DM')
                mis, ris = getTrueDM(N,xlim)
                
                
                ic_guess = orbitModule.get_S2_IC()
                
                
                
                print('Starting from 0')
                dm_guess = Nf*[0]
                # dm_guess = [9.492757471095918e-05, 9.906663990242069e-05, 0.00010494936656500701, 0.00011156525554012521, 0.00011916811948511612, 0.00012831685318671443, 0.00014053512764023823, 0.00016180965436519293, 0.0001984523234489342, 2.4308125394118965e-10]
                # dm_guess = [8.444633654919243e-05, 9.872153866761411e-05, 0.00011193335857871409, 0.00011968076505676556, 0.00012578257548928034, 0.0001324663808173125, 0.00013724169718901135, 0.00014207480882013533, 0.000211996380321655, 2.4325580272583567e-10]
                
                # print('Starting from true masses')
                # dm_guess = mis.copy()
                # print('Starting from Plummer')
                # dm_guess, _ = orbitModule.get_Plummer_DM(N, xlim)
                # print('Starting from Random IG')
                # np.random.seed(0)
                # noiseLevel = 0.5*max(mis)
                # noise = np.random.normal(0,noiseLevel,len(mis))
                # dm_guess = mis.copy() + noise
                
                
                reconic, reconmisInit, initloss = orbitModule.reconstructFromFile(filepath,ic_guess,dm_guess, \
                            ADD_NOISE = False, noisefactor = 1e-1,seed=2)
               
                
                # # reconmisInit = [0.00016456404871297104, 0.00021621004966190728, 0.0002881053212509268, 0.00020701015669467748, 0.00028355321722857375]
                
                plotInitReconTrueMasses(dm_guess,reconmisInit,mis,STD=False)
                
                print(list(reconmisInit))
                print(initloss)
                
                # observations = np.loadtxt('Datasets/BahcallWolf_N={}.txt'.format(N))
                # timegrid = observations[:,0]
                # t_grid = orbitModule.convertYearsTimegridToOurFormat(timegrid)
                # variance_x0, variance_DM = orbitModule.getModelUncertainty(  \
                #         orbitModule.get_S2_IC(), dm_guess, t_grid, noisefactor=1e-1)
                # variance_DM = 10*variance_DM
                # noiseLevel = 20*variance_DM
                # for i in range(7):
                #     noiseLevel[13+i] = 0
                # noiseLevel[4] = 0
                    
                
                #0 noise -> 5000 iterations
                #1e-2 -> 1200 iterations
                #1 -> 200 iterations?
                
                # reconMises = []
                # losses = []
                # for i in range(10):
                #     # dm_guess_new = Nf*[0]
                #     np.random.seed(i)
                #     noise = np.random.normal(0,0.0001,len(dm_guess))
                #     # dm_guess_noise = mis.copy() + noise
                #     # dm_guess_new = reconmisInit.copy() * [0.5 + (i+1)/7]
                #     dm_guess_new = reconmisInit.copy() + noise
                #     # dm_guess_new = [0.00016456404871297104, 0.00021621004966190728, 0.0002881053212509268, 0.00020701015669467748, 0.00028355321722857375]
                    
                    
                #     # dm_guess_new = [0.00017661782060660752, 0.00021817973039456827, 0.00023547195019420783, 0.00025362165691468854, 0.0002564173694075912]
                    
                    
                #     reconic, reconmis,loss = orbitModule.reconstructFromFile(filepath,ic_guess,dm_guess_new, \
                #                                 ADD_NOISE = True, noisefactor = 1e-1,seed=2)
                    
                #     # if i < 2:
                #     plotInitReconTrueMasses(dm_guess_new,reconmis,mis,STD = False)
                    
                #     # print(list(reconmis))
                #     reconMises.append(list(reconmis))
                #     losses.append(loss)
                # # print(noiseLevel)
                # # print(list(reconmisInit))
                # print(reconMises)
                # print(losses)
                # continue
            
                
                
                break
        else:
            continue

def reconstructFromTrueMasses():
    # mis, ris = orbitModule.get_Plummer_DM(N,xlim)
    # mis, ris = orbitModule.get_BahcallWolf_DM(N,xlim)
    # mis,ris = orbitModule.get_Uniform_DM(N, xlim)
    mis,ris = orbitModule.get_Sinusoidal_DM(N, xlim)
    # mis = 1*np.array(mis)
    
    IC = orbitModule.get_S2_IC()
    ic_guess = IC
    # ic_guess = np.multiply(IC, len(IC)*[1.000001])
    
    dm_guess = N*[0]
    # dm_guess,_ = orbitModule.get_Uniform_DM(N, xlim)
    # dm_guess,_ = orbitModule.get_Plummer_DM(N,xlim)
    # dm_guess = 0.5*np.array(mis)
    
    #Times of observation in [seconds/T_0]
    M_0, D_0, T_0 = orbitModule.getBaseUnitConversions()
    
    obstimes =  np.append(0,(np.linspace(0,16.056,228) * 365.25 * 24 * 60**2 /T_0 ) + 84187.772)
    # obstimes = np.linspace(0,16.056,228) * 365.25 * 24 * 60**2 /T_0 
    
    reconic, reconmis = orbitModule.reconstructDistributionFromTrueMasses(True,mis,ris,obstimes, \
                                      ic_guess,dm_guess, CARTESIANOBS = True,OBS3 = True)
        
    
    plotInitReconTrueMasses(dm_guess,reconmis,mis)
    
    
    # obstimes = obstimes[:2]
    # variance_x0, variance_DM = orbitModule.getModelUncertainty(reconic, reconmis, obstimes)
    
    # rp = 119.52867
    # ra = 1948.96214
    # plt.figure()
    # plt.scatter(ris,mis,label='True')
    # # plt.scatter(ris,dm_guess,label='Initial guess',color='grey',alpha=0.5)
    # plt.scatter(ris,reconmis,label='Reconstructed')
    # plt.axvline(rp,linestyle='--',label='rp and ra',color='black')
    # plt.axvline(ra,linestyle='--',color='black')
    # plt.xlabel("Distance from MBH [AU]")
    # plt.ylabel("Mass [MBH masses]")
    # plt.title('Reconstructed dark matter distribution')
    
    # variance_DM = variance_DM
    # # print(variance_DM)
    # for i in range(N):
    #     if i == 0:
    #         plt.errorbar(ris[i],reconmis[i],variance_DM[i],capsize=5,color='blue',label='Variance')
    #     else:
    #         plt.errorbar(ris[i],reconmis[i],variance_DM[i],capsize=5,color='blue')

    # plt.legend()



"""
#Uncomment functions here to use them:
"""

if __name__ == "__main__":
    # reconstructAllDatasets()
    reconstructFromTrueMasses()
    # comparePlummer_BahcallWolfReconstruction(noisefactor=1e-5)
    # comparePlummer_BahcallWolfReconstruction(noisefactor=5e-1)
    # comparePlummer_BahcallWolfReconstruction(noisefactor=1)