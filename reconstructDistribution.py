# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 16:21:03 2022

@author: Thibault
"""

import os
import numpy as np
from matplotlib.pylab import plt
import orbitModule
import pandas as pd

#Max x limit in [AU]
xlim = orbitModule.get_xlim()
#Amount of dark matter shells
N = 5
#Sigmoid steepness factor
k = orbitModule.get_k_SteepnessFactor(N)

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
    
    
def plotICRobustness(biglist,names,noisefactor=1):
    colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple"]
    alpha = 0.2
    
    rp = 119.52867
    ra = 1948.96214
    
    fig, ax12 = plt.subplots(1,1)
    # fig.set_size_inches(19,4)
    fig.set_size_inches(6,4)
    fig.set_tight_layout(True)
    
    mergedICs = []
    
    
    for i in range(len(names)):
        reconICs = biglist[i][0]
        reconmis = biglist[i][1]
        mis = biglist[i][2]
        
        nbrRecons = len(reconICs)
        #Multiply by 100 to get percentages
        reconmis = 100*np.array(reconmis)
        mis = 100*np.array(mis)
        
        for ic in reconICs:
            mergedICs.append(ic)
    
        #Plot masses:
        # ax11.scatter(ris,mis,label=names[i],color=colors[i])
        # for j in range(1,nbrRecons):
        #     ax11.scatter(ris,reconmis[j],alpha=alpha,color=colors[i]) 
        # #Again to get the label;
        # ax11.scatter(ris,reconmis[0],label='Reconstructed ' + names[i],alpha=alpha,color=colors[i]) 
        # # ax11.set_ylim(0,1.2*max(mis))
            
        
        
        # # ax11.axvline(rp,linestyle='--',label='$r_a$ and $r_p$',color='black')
        # ax11.axvline(ra,linestyle='--',color='black')
        # ax11.axvline(rp,linestyle='--',color='black')
        # ax11.set_xlabel("$r$ [AU]")
        # ax11.set_ylabel(r'Shell mass [% of $M_\bullet$]')
        # ax11.set_xlim(0,2100)
        # ax11.legend(loc='best')
    
        
        #Plot enclosed mass:
        allSumRis = []
        for j in range(nbrRecons):
            sumRis = getEnclosedMass(reconmis[j])
            allSumRis.append(sumRis)
        sumRisTrue = getEnclosedMass(mis)
        # sumRisInit = getEnclosedMass(dm_guess)
        
        
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
        ax12.set_xlabel('r [AU]')
        ax12.set_ylabel(r'enclosed mass [% of $M_\bullet$]')
        # plt.plot(rDM,enclosedMassPlum(rDM),label='Plum model')
        
        ax12.plot(rDM,sumRisTrue,label=names[i] + ' GT',color=colors[i])
        
        
        if noisefactor != 0:
            #Means:
            means = np.mean(allSumRis,axis=0)
            stddevs = np.std(allSumRis,axis=0)
            ax12.plot(rDM,means,"--",label=names[i]+" mean reconstruction",color=colors[i])
            ax12.fill_between(rDM,means-stddevs,means+stddevs,alpha=alpha, facecolor=colors[i],
                            label=names[i]+' standard deviation')
        else:
            ax12.plot(rDM,sumRisTrue,"--",label=names[i]+" reconstruction",color=colors[i])
            
        
        
        #All:
        # for j in range(1,nbrRecons):
        #     ax12.plot(rDM,allSumRis[j],color=colors[i],alpha=alpha)
        # #Again to get the label once
        # ax12.plot(rDM,allSumRis[0],label=names[i] + ' reconstruction',color=colors[i],alpha=alpha)
        
        
        # ax12.plot(rDM,sumRisInit,label='Initial guess',color='grey',alpha=0.5)
        # ax12.axvline(rp,linestyle='--',label='$r_a$ and $r_p$',color='black')
        ax12.axvline(ra,linestyle='--',color='black')
        ax12.axvline(rp,linestyle='--',color='black')
        ax12.legend()
        ax12.set_xlim(0,2100)
        
    # print(mergedICs)
    
    # plt.figure()
    df = pd.DataFrame(mergedICs, columns = ['p', 'e','i','Om','w','f'])
    # df.boxplot()
    print(df.describe())
    
    #Convert to degrees:
    df.i *= 180/np.pi
    df.Om *= 180/np.pi
    df.w *= 180/np.pi
    df.f *= 180/np.pi
    
    
    IC = orbitModule.get_S2_IC()
    IC[2:] = [i*180/np.pi for i in IC[2:]]
    
    fig, ((ax1,ax2,ax3)) = plt.subplots(1,3)
    # fig.set_size_inches(19,4)
    fig.set_size_inches(13,3)
    fig.set_tight_layout(True)
    
    
    plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
    
    df.iloc[0:10].plot.scatter(x='p',
                        y='e',color=colors[0],ax=ax1,label='Plummer')
    df.iloc[10:20].plot.scatter(x='p',
                        y='e',color=colors[1],ax=ax1,label='Bahcall-Wolf')
    df.iloc[20:30].plot.scatter(x='p',
                        y='e',color=colors[2],ax=ax1,label='Alpha')
    # ax1.scatter(IC[0],IC[1],label="True")
    ax1.errorbar(IC[0],IC[1], xerr=noisefactor*0.131, yerr=noisefactor*0.00006, fmt='none')
    ax1.legend()
    ax1.set_xlabel(r"$p$ [AU]")
    ax1.set_ylabel(r"$e$")
    
    # ax1 = df.plot.scatter(x='i',
    #                     y='Om')
    df.iloc[0:10].plot.scatter(x='i',
                        y='Om',color=colors[0],ax=ax2,label='Plummer')
    df.iloc[10:20].plot.scatter(x='i',
                        y='Om',color=colors[1],ax=ax2,label='Bahcall-Wolf')
    df.iloc[20:30].plot.scatter(x='i',
                        y='Om',color=colors[2],ax=ax2,label='Alpha')
    # ax1.scatter(IC[2],IC[3],label="True")
    ax2.errorbar(IC[2],IC[3], xerr=noisefactor*0.03, yerr=noisefactor*0.03, fmt='none')
    ax2.legend()
    ax2.set_xlabel(r"$\iota$ [°]")
    ax2.set_ylabel(r"$\Omega$ [°]")
    
    # ax1 = df.plot.scatter(x='w',
    #                     y='f')
    df.iloc[0:10].plot.scatter(x='w',
                        y='f',color=colors[0],ax=ax3,label='Plummer')
    df.iloc[10:20].plot.scatter(x='w',
                        y='f',color=colors[1],ax=ax3,label='Bahcall-Wolf')
    df.iloc[20:30].plot.scatter(x='w',
                        y='f',color=colors[2],ax=ax3,label='Alpha')
    # ax1.scatter(IC[4],IC[5],label="True")
    ax3.errorbar(IC[4],IC[5], xerr=noisefactor*0.03, yerr=0, fmt='none')
    ax3.legend()
    ax3.set_xlabel(r"$\omega$ [°]")
    ax3.set_ylabel(r"$f$ [°]")
    
    
            # ic_noisy[0] = ic_noisy[0] + np.random.normal(0,noisefactor*0.131) #p
            # ic_noisy[1] = ic_noisy[1] + np.random.normal(0,noisefactor*0.00006) #e
            # ic_noisy[2] = ic_noisy[2] + np.random.normal(0,noisefactor*0.03 / 180 * np.pi) #i
            # ic_noisy[3] = ic_noisy[3] + np.random.normal(0,noisefactor*0.03 / 180 * np.pi) #om
            # ic_noisy[4] = ic_noisy[4] + np.random.normal(0,noisefactor*0.03 / 180 * np.pi) #w
            # ic_noisy[5] = ic_noisy[5] + np.random.normal(0,noisefactor*0) #f
    
    
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
    # vols = 4*np.pi*(np.array(ris)**3)/3
    # for i in range(1,len(vols)):
    #     vols[i] = vols[i] - vols[i-1]
    
    # dens = mis/(100*vols)
    # recondens = reconmis/(100*vols)
    # # initdens = dm_guess/(100*vols)
    
    
    # M_0, D_0, T_0 = orbitModule.getBaseUnitConversions()
    # dens = dens * M_0 / (D_0**3)
    # recondens = recondens * M_0 / (D_0**3)
    # initdens = initdens * M_0 / (D_0**3)
    
    # ax13.scatter(ris,dens,label='True')
    # ax13.scatter(ris,recondens,label='Reconstructed')
    
    # # if 0 not in initdens:
    # ax13.scatter(ris,initdens,label='Initial guess',color='grey',alpha=0.5)
    # # ax13.axvline(rp,linestyle='--',label='$r_a$ and $r_p$',color='black')
    # ax13.axvline(ra,linestyle='--',color='black')
    # ax13.axvline(rp,linestyle='--',color='black')
    # # ax13.set_ylabel('Density [MBH masses/(AU³)')
    # ax13.set_ylabel('Density [kg/$m^3$]')
    # ax13.set_xlabel('$r$ [AU]')
    # ax13.set_yscale('log')
    # # ax13.set_ylim(1e-15)
    # ax13.set_xlim(0,2100)
    # ax13.legend()
    
    
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
                ic_noisy = ic_guess.copy()
                
                
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
                
                
                reconic, reconmisInit = orbitModule.reconstructFromFile(filepath,ic_noisy,dm_guess, \
                            noisefactor = noisefactor,seed=5)
               
                
                plotInitReconTrueMasses(dm_guess,reconmisInit,mis)
                
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
    
    
    obstimes =  orbitModule.getObservationTimes(nbrOfOrbits=1)
    
    
    reconic, reconmis = orbitModule.reconstructDistributionFromTrueMasses(True,mis,ris,obstimes, \
                              ic_guess,dm_guess, CARTESIANOBS = True,OBS3 = True, \
                              noisefactor=noisefactor,seed=0)
        
    
    plotInitReconTrueMasses(dm_guess,reconmis,mis)
    
    
def checkRobustnessToNoise(noisefactor,amountOfRecons,name):
    getTrueDM = getattr(orbitModule,'get_'+name+'_DM')
    mis, ris = getTrueDM(N,xlim)
    
    IC = orbitModule.get_S2_IC()
    
    obstimes =  orbitModule.getObservationTimes()
    
    reconmises = []
    #Reconstructs with different noise profiles (random seed changes every iteration)
    for i in range(amountOfRecons):
        ic_guess = IC.copy()
        dm_guess = N*[0]
        reconic, reconmis = orbitModule.reconstructDistributionFromTrueMasses(True,mis,ris,obstimes, \
                              ic_guess,dm_guess, CARTESIANOBS = True,OBS3 = True, \
                              noisefactor = noisefactor,seed=i)
        reconmises.append(reconmis)
    
    mean = np.mean(reconmises,axis=0)
    stddevs = np.std(reconmises,axis=0)
    
    dm_guess = N*[0]
    plotInitReconTrueMasses(dm_guess,mean,mis,stddevs)
    
    
def checkRobustnessToInitialConditions(noisefactor,amountOfRecons):
    names = ['Plummer','BahcallWolf','Alpha']
    
    #Offset is changed every experiment in order to not have the same seeds
    seedoffset = 300000
    
    #100000 for 1e-1 noise
    #200000 for 1 noise, 10x orbits
    #300000 for 1 noise, 10x observations
    
    biglist = []
    for name in names:
        getTrueDM = getattr(orbitModule,'get_'+name+'_DM')
        mis, ris = getTrueDM(N,xlim)
        
        IC = orbitModule.get_S2_IC()
        
        #TODO: set amount of orbits here
        obstimes =  orbitModule.getObservationTimes(10)
        
        
        reconmises = []
        reconICs = []
        #Reconstructs with different initial conditions
        for i in range(amountOfRecons):
            #Need to set the seed, otherwise following iterations have the same seed
            np.random.seed(i+20*len(biglist)+seedoffset)
            dm_guess = N*[0]
            
            ic_noisy = IC.copy()
            ic_noisy[0] = ic_noisy[0] + np.random.normal(0,0.1*noisefactor*0.131) #p
            ic_noisy[1] = ic_noisy[1] + np.random.normal(0,0.1*noisefactor*0.00006) #e
            ic_noisy[2] = ic_noisy[2] + np.random.normal(0,0.1*noisefactor*0.03 / 180 * np.pi) #i
            ic_noisy[3] = ic_noisy[3] + np.random.normal(0,0.1*noisefactor*0.03 / 180 * np.pi) #om
            ic_noisy[4] = ic_noisy[4] + np.random.normal(0,0.1*noisefactor*0.03 / 180 * np.pi) #w
            ic_noisy[5] = ic_noisy[5] + np.random.normal(0,0.1*noisefactor*0) #f
            
            
            reconic, reconmis = orbitModule.reconstructDistributionFromTrueMasses(True,mis,ris,obstimes, \
                                  ic_noisy,dm_guess, CARTESIANOBS = True,OBS3 = True, \
                                  noisefactor = noisefactor,seed=i+20*len(biglist)+seedoffset+1000)
                
            reconICs.append(reconic)
            reconmises.append(reconmis)
        
        biglist.append([reconICs, reconmises, mis])
        # break
    
    print(biglist)
    plotICRobustness(biglist,names)
    


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
    # reconstructAllDatasets(noisefactor=1)
    
    #Reconstruct specific profiles
    #Possible names: Plummer,BahcallWolf,Sinusoidal,Uniform,ReversedPlummer,ConstantDensity
    # reconstructFromTrueMasses(noisefactor = 1e-1,name='BahcallWolf')
    
    #For different noise samples, check the robustness of the reconstruction:
    # checkRobustnessToNoise(noisefactor=1e-1,amountOfRecons=3,name='Plummer')
    
    #For different initial conditions, check the robustness of the reconstruction:
    checkRobustnessToInitialConditions(noisefactor=1,amountOfRecons=10)
    
    #Compare equi-temporal spacing vs equi-spatial spacing
    # compareDifferentTimeGrids(noisefactor = 1e-1,name='Plummer')
    
    #Start from the Plummer distribution, reconstruct on Bahcall observations and vice versa
    # comparePlummer_BahcallWolfReconstruction(noisefactor=1e-5)
    
    #Look at the loss landscape for different distributions
    # orbitModule.lossLandscape(N=5,noisefactor=1,nbrOfDistributions=250000)
    
    #Calculate the variance of the true loss wrt different noise samples
    # checkLossVariance(noisefactor=1e-2)
    
    