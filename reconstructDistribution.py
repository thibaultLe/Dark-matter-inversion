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
    alpha = 0.15
    
    rp = 119.52867
    ra = 1948.96214
    
    fig, ax12 = plt.subplots(1,1)
    # fig.set_size_inches(19,4)
    fig.set_size_inches(6.7,4*6.7/6)
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
            ax12.plot(rDM,means-stddevs,linestyle=':',color=colors[i])
            ax12.plot(rDM,means+stddevs,linestyle=':',color=colors[i])
        else:
            # ax12.plot(rDM,sumRisTrue,"--",label=names[i]+" reconstruction",color=colors[i])
            
            # ax12.plot(rDM,sumRisTrue,":",label=names[i]+" reconstruction",color=colors[i],linewidth=4)
            ax12.plot(rDM,sumRisTrue,"--",label=names[i]+" reconstruction",color=colors[i],linewidth=3.5)
            # ax12.plot(rDM,sumRisTrue,":",label=names[i]+" reconstruction",color='black',linewidth=3)
            
        
        
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
        ax12.set_ylim(-0.005,0.11)
        
        
    #Set custom legend:
    if noisefactor != 0:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from matplotlib.colors import to_rgba
        custom_lines = [Line2D([0], [0], color='tab:blue'),
                        Line2D([0], [0], ls='--', color='tab:blue'),
                        Patch(facecolor=to_rgba('tab:blue',0.2), edgecolor='tab:blue',ls=':',lw=1.5),
                        Line2D([0], [0], color='tab:orange'),
                        Line2D([0], [0], ls='--',color='tab:orange'),
                        Patch(facecolor=to_rgba('tab:orange',0.2), edgecolor='tab:orange',ls=':',lw=1.5),
                        Line2D([0], [0], color='tab:green'),
                        Line2D([0], [0], ls='--', color='tab:green'),
                        Patch(facecolor=to_rgba('tab:green',0.2), edgecolor='tab:green',ls=':',lw=1.5)]
        
        mean = 'mean reconstruction'
        std = 'standard deviation'
        ax12.legend(custom_lines, ['Plummer GT', 'Plummer '+mean,'Plummer '+std,
                                    'Bahcall-Wolf GT', 'Bahcall-Wolf '+mean,'Bahcall-Wolf '+std,
                                    'Zhao GT', 'Zhao '+mean,'Zhao '+std])
        
        
        
        
        
        
        
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
                        y='e',color=colors[0],ax=ax1,label=names[0])
    df.iloc[10:20].plot.scatter(x='p',
                        y='e',color=colors[1],ax=ax1,label=names[1])
    df.iloc[20:30].plot.scatter(x='p',
                        y='e',color=colors[2],ax=ax1,label=names[2])
    # ax1.scatter(IC[0],IC[1],label="True")
    ax1.errorbar(IC[0],IC[1], xerr=noisefactor*0.131, yerr=noisefactor*0.00006, fmt='none')
    ax1.legend()
    ax1.set_xlabel(r"$p$ [AU]")
    ax1.set_ylabel(r"$e$")
    
    # ax1 = df.plot.scatter(x='i',
    #                     y='Om')
    df.iloc[0:10].plot.scatter(x='i',
                        y='Om',color=colors[0],ax=ax2,label=names[0])
    df.iloc[10:20].plot.scatter(x='i',
                        y='Om',color=colors[1],ax=ax2,label=names[1])
    df.iloc[20:30].plot.scatter(x='i',
                        y='Om',color=colors[2],ax=ax2,label=names[2])
    # ax1.scatter(IC[2],IC[3],label="True")
    ax2.errorbar(IC[2],IC[3], xerr=noisefactor*0.03, yerr=noisefactor*0.03, fmt='none')
    ax2.legend()
    ax2.set_xlabel(r"$\iota$ [°]")
    ax2.set_ylabel(r"$\Omega$ [°]")
    
    # ax1 = df.plot.scatter(x='w',
    #                     y='f')
    df.iloc[0:10].plot.scatter(x='w',
                        y='f',color=colors[0],ax=ax3,label=names[0])
    df.iloc[10:20].plot.scatter(x='w',
                        y='f',color=colors[1],ax=ax3,label=names[1])
    df.iloc[20:30].plot.scatter(x='w',
                        y='f',color=colors[2],ax=ax3,label=names[2])
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

    plt.show()


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
    seedoffset = 200005
    
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
            seed = i+20*len(biglist)+seedoffset
            np.random.seed(seed)
            print(seed)
            
            dm_guess = N*[0]
            ic_noisy = IC.copy()
            ic_noisy[0] = ic_noisy[0] + np.random.normal(0,noisefactor*0.131) #p
            ic_noisy[1] = ic_noisy[1] + np.random.normal(0,noisefactor*0.00006) #e
            ic_noisy[2] = ic_noisy[2] + np.random.normal(0,noisefactor*0.03 / 180 * np.pi) #i
            ic_noisy[3] = ic_noisy[3] + np.random.normal(0,noisefactor*0.03 / 180 * np.pi) #om
            ic_noisy[4] = ic_noisy[4] + np.random.normal(0,noisefactor*0.03 / 180 * np.pi) #w
            ic_noisy[5] = ic_noisy[5] + np.random.normal(0,noisefactor*0) #f
            
            
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
    # checkRobustnessToInitialConditions(noisefactor=1,amountOfRecons=5)
    
    #Compare equi-temporal spacing vs equi-spatial spacing
    # compareDifferentTimeGrids(noisefactor = 1e-1,name='Plummer')
    
    #Start from the Plummer distribution, reconstruct on Bahcall observations and vice versa
    # comparePlummer_BahcallWolfReconstruction(noisefactor=1e-5)
    
    #Look at the loss landscape for different distributions
    # orbitModule.lossLandscape(N=5,noisefactor=1,nbrOfDistributions=250000)
    
    #Calculate the variance of the true loss wrt different noise samples
    # checkLossVariance(noisefactor=1)
    
    
    
    
    
    
    names = ['Plummer','Bahcall-Wolf','Zhao']

    #10 orbits IG3 (figure f)
    biglist = [[[[225.14450316,   0.88448013,  -2.35096032,   3.9827952 ,
          1.15684886,  -3.14160597], [225.29794799,   0.88439777,  -2.35095092,   3.98249256,
          1.15631011,  -3.141585  ], [225.38546742,   0.88435221,  -2.35095718,   3.98242158,
          1.1561018 ,  -3.14157458], [225.12122934,   0.88449669,  -2.35096016,   3.98293329,
          1.15706965,  -3.14160476], [225.12624455,   0.88448663,  -2.35098087,   3.98282319,
          1.15690976,  -3.14160589],[225.12447655,   0.88449038,  -2.35097893,   3.98287905,
          1.15697183,  -3.1416048 ], [225.29052286,   0.88440142,  -2.3509613 ,   3.98258906,
          1.1564431 ,  -3.14158619], [225.46700931,   0.88430648,  -2.3509728 ,   3.98228809,
          1.15573305,  -3.14156028], [225.2295861 ,   0.88443699,  -2.350959  ,   3.98270598,
          1.15659445,  -3.1415927 ], [225.34192066,   0.88437422,  -2.35096733,   3.98247031,
          1.15628201,  -3.14158469]],
[[0.00010786, 0.00014195, 0.00018279, 0.00020784, 0.00018416], [8.03909573e-05, 9.95305335e-05, 1.45070836e-04, 2.25969920e-04,
        3.94457886e-04], [2.42837930e-05, 6.89897562e-05, 1.65154358e-04, 3.00458373e-04,
        5.10241835e-04], [5.81771118e-05, 1.22776568e-04, 2.26504921e-04, 3.09797377e-04,
        2.50004096e-04], [1.60443542e-04, 1.68039964e-04, 1.66782808e-04, 1.42260692e-04,
        1.31467607e-05],[1.11952736e-04, 1.51755348e-04, 2.07250295e-04, 2.17367852e-04,
        7.34887349e-05], [6.54923207e-05, 9.39486278e-05, 1.68197811e-04, 2.71412071e-04,
        3.88613864e-04], [1.66892397e-05, 5.61289158e-05, 1.41851032e-04, 2.84912273e-04,
        5.53501187e-04], [5.87012746e-05, 9.50071734e-05, 1.68569515e-04, 2.82804403e-04,
        4.09564729e-04], [6.83470970e-05, 9.63381668e-05, 1.53669008e-04, 2.43172428e-04,
        4.12154235e-04]],
[3.609632592453552e-05, 0.0001181406130613886, 0.0002112682471865419, 0.000283560136574258, 0.0003221765543589564]],
#SECOND 2
[[[225.33902961,   0.88437719,  -2.3509448 ,   3.98245214,
          1.15625845,  -3.14158755], [225.077806  ,   0.88451767,  -2.3509823 ,   3.98302775,
          1.15717037,  -3.14160756], [225.35673551,   0.88437006,  -2.3509493 ,   3.98251098,
          1.15623833,  -3.14158333], [225.13993597,   0.88448036,  -2.35097265,   3.98278077,
          1.15691006,  -3.14160659], [225.10390892,   0.88450401,  -2.35097026,   3.98296723,
          1.15706491,  -3.14160569],[225.36107538,   0.88436613,  -2.35095085,   3.98246417,
          1.15622228,  -3.14158534], [225.10539474,   0.88450442,  -2.3509881 ,   3.98302939,
          1.15711154,  -3.14160312], [225.21144065,   0.88444292,  -2.35097445,   3.9826808 ,
          1.15660939,  -3.14159338], [225.08493325,   0.88451356,  -2.35096138,   3.98296068,
          1.15715054,  -3.14161103], [225.15874253,   0.88447316,  -2.35097296,   3.98284137,
          1.15696173,  -3.14160795]],
[[0.00013336, 0.00014696, 0.00017165, 0.0002124 , 0.0003154 ], [0.00019356, 0.00020552, 0.0002145 , 0.00017161, 0.        ], [0.00010056, 0.00011945, 0.00016397, 0.0002606 , 0.0004764 ], [2.51919796e-04, 2.31840077e-04, 1.56159963e-04, 3.33435259e-05,
        0.00000000e+00], [1.71041593e-04, 2.00859990e-04, 2.23601813e-04, 1.87625583e-04,
        3.54808205e-05],[0.00012137, 0.00013076, 0.00016541, 0.00023467, 0.0003877 ], [0.00017274, 0.0001876 , 0.00019842, 0.00020406, 0.00012884], [2.08047627e-04, 1.91283547e-04, 1.58591101e-04, 1.22160779e-04,
        8.28335826e-05], [2.04812899e-04, 2.10311187e-04, 1.98606260e-04, 1.46106645e-04,
        2.57398148e-05], [2.00303386e-04, 1.99098273e-04, 1.99349654e-04, 1.61995804e-04,
        6.99607046e-06]],
[0.00014370151360588277, 0.00017198634513687495, 0.0001905457505668284, 0.00020483338733222063, 0.00021662326462725586]],
#THIRD 3
[[[225.20951931,   0.8844411 ,  -2.35099748,   3.98268325,
          1.15658686,  -3.14159301], [225.18051874,   0.88445956,  -2.35095368,   3.98264169,
          1.15665132,  -3.14160419], [225.18862122,   0.88445843,  -2.35096938,   3.98276925,
          1.15677625,  -3.14159897], [225.26300026,   0.88441478,  -2.3509806 ,   3.98261671,
          1.15650172,  -3.14159239], [225.19527787,   0.88445305,  -2.35097144,   3.98273494,
          1.15668872,  -3.1415948 ],[225.34771004,   0.88437131,  -2.35096029,   3.9824458 ,
          1.15620636,  -3.14158465], [225.09840617,   0.88450307,  -2.35100157,   3.98297025,
          1.15706089,  -3.1416044 ], [225.35113724,   0.88437061,  -2.35095736,   3.98245731,
          1.15614956,  -3.14158211], [225.12483651,   0.88449003,  -2.3509754 ,   3.98287529,
          1.15692594,  -3.14160349], [225.15718576,   0.88447337,  -2.35098285,   3.98284323,
          1.15682299,  -3.14159703]],
[[0.00015036, 0.00015805, 0.0001683 , 0.00017993, 0.00016136], [0.00010602, 0.0001547 , 0.0002227 , 0.00024178, 0.00016364], [6.97812925e-05, 1.29785197e-04, 2.32560180e-04, 3.10715030e-04,
        2.95802062e-04], [0.00010949, 0.00012867, 0.00017513, 0.00024715, 0.00030597], [8.34662972e-05, 1.38097158e-04, 2.16381777e-04, 2.78492626e-04,
        2.74173334e-04],[5.99517476e-05, 9.34975308e-05, 1.68051914e-04, 3.03576000e-04,
        5.17862682e-04], [1.30590912e-04, 1.78145268e-04, 2.31154165e-04, 2.15846418e-04,
        6.33126936e-05], [2.94365028e-05, 8.34462465e-05, 1.89257074e-04, 3.35229196e-04,
        5.47696961e-04], [0.00012441, 0.00015248, 0.00020233, 0.00023158, 0.00020495], [9.36864324e-05, 1.38937804e-04, 2.13134264e-04, 2.68579289e-04,
        2.88954349e-04]],
[1.1236562259431423e-07, 2.948251663966858e-05, 0.0004899036078155068, 0.00041268977094565046, 5.8154009754610896e-05]]]

    plotICRobustness(biglist,names,noisefactor=1)

    # 10 orbits, start from IG2
#     biglist = [[[[225.23340693,   0.88443306,  -2.35095301,   3.98263533,
#           1.15656784,  -3.14159822], [225.2487637 ,   0.88442422,  -2.35095314,   3.98258842,
#           1.15647773,  -3.14158946], [225.2575243 ,   0.88442144,  -2.35096223,   3.98267463,
#           1.15654318,  -3.14158647], [225.23110885,   0.88443858,  -2.35095082,   3.98273731,
#           1.15672479,  -3.1415952 ], [225.23161545,   0.88443066,  -2.35097294,   3.98263078,
#           1.15657211,  -3.14159662], [225.23143723,   0.88443366,  -2.35097067,   3.98268496,
#           1.15663122,  -3.14159544],[225.2449748 ,   0.88442423,  -2.3509796 ,   3.98264934,
#           1.15657481,  -3.14159538], [225.2324573 ,   0.88443264,  -2.35096548,   3.98265189,
#           1.15660406,  -3.14159388], [225.21672471,   0.88444274,  -2.35096891,   3.98273148,
#           1.15668516,  -3.14159817],[225.25586003,   0.88442331,  -2.35096691,   3.98270238,
#           1.15656967,  -3.14158842]], 
# [[7.92284553e-05, 1.16021931e-04, 1.69488586e-04, 2.35747103e-04,
#         3.27197893e-04], [9.07941286e-05, 1.10760008e-04, 1.53194473e-04, 2.17898461e-04,
#         3.43060629e-04], [4.64991855e-05, 9.61671864e-05, 1.91060799e-04, 2.88768239e-04,
#         3.75858265e-04], [2.11879073e-05, 8.94941294e-05, 2.11980032e-04, 3.49850643e-04,
#         4.24395431e-04], [0.00013065, 0.0001377 , 0.00014748, 0.00017028, 0.00017911], [8.26782498e-05, 1.20029150e-04, 1.84291187e-04, 2.43244760e-04,
#         2.56061099e-04],[0.00012065, 0.00012756, 0.00015111, 0.00018166, 0.00023228], [0.00011043, 0.00012012, 0.00014298, 0.00019818, 0.00031889], [7.90447981e-05, 1.11236651e-04, 1.72580825e-04, 2.50131495e-04,
#         3.39846942e-04],[4.09697285e-05, 8.80598692e-05, 1.88646927e-04, 2.99323145e-04,
#         4.27353678e-04]], 
# [3.609632592453552e-05, 0.0001181406130613886, 0.0002112682471865419, 0.000283560136574258, 0.0003221765543589564]], 
# [[[225.2528924 ,   0.88442333,  -2.35095015,   3.98261435,
#           1.15654244,  -3.14159533], [225.22675438,   0.88443811,  -2.35097245,   3.98275226,
#           1.15668639,  -3.14159425], [225.2546953 ,   0.88442414,  -2.35095729,   3.98269623,
#           1.15656329,  -3.14159223], [225.23294745,   0.88443029,  -2.35096746,   3.98260523,
#           1.15660122,  -3.14159808], [225.22936727,   0.88443725,  -2.3509615 ,   3.98273628,
#           1.15665908,  -3.14159465], [225.25507514,   0.88442295,  -2.35095667,   3.98266637,
#           1.15657656,  -3.14159488],[225.23617255,   0.88443119,  -2.35095203,   3.9826397 ,
#           1.15661691,  -3.14159651], [225.24715552,   0.88442496,  -2.35096147,   3.98261993,
#           1.15653079,  -3.14159404], [225.22605995,   0.88443823,  -2.35095775,   3.98270267,
#           1.15670476,  -3.14159568], [225.22671396,   0.88443403,  -2.3509682 ,   3.98263766,
#           1.15666346,  -3.14160015]], 
# [[0.00015375, 0.00016978, 0.00018781, 0.00019368, 0.00020007], [0.00015003, 0.00016597, 0.00019773, 0.00021605, 0.00018717], [0.00012762, 0.0001501 , 0.0001875 , 0.00023658, 0.00029954], [2.23751714e-04, 2.06999005e-04, 1.51586980e-04, 8.75266284e-05,
#         5.23962042e-05], [0.00013379, 0.0001674 , 0.00020705, 0.00022182, 0.00020872], [0.00014788, 0.00015569, 0.00018062, 0.00021268, 0.00026639],[0.00018816, 0.00017583, 0.00016521, 0.00016041, 0.00015731], [0.00017967, 0.00017338, 0.00017053, 0.00017522, 0.00013631], [0.00016001, 0.00017374, 0.00019148, 0.00019809, 0.0001901 ],[2.29046247e-04, 1.96078242e-04, 1.49914065e-04, 1.12055003e-04,
#         4.45093749e-05]],
#   [0.00014370151360588277, 0.00017198634513687495, 0.0001905457505668284, 0.00020483338733222063, 0.00021662326462725586]], 
# [[[225.23996881,   0.88442501,  -2.35099481,   3.98262908,
#           1.15649158,  -3.14159036], [225.23705167,   0.88442941,  -2.35094807,   3.98254203,
#           1.1564763 ,  -3.14159899], [225.23780994,   0.88443204,  -2.35096435,   3.98268259,
#           1.15662459,  -3.14159427], [225.2452743 ,   0.88442447,  -2.35098111,   3.98265271,
#           1.15656416,  -3.14159411], [225.23847366,   0.88443017,  -2.35096785,   3.98265734,
#           1.15655223,  -3.14159102], [225.25375455,   0.88442265,  -2.3509639 ,   3.98263375,
#           1.15653345,  -3.1415938 ], [225.25764105,   0.88441898,  -2.3509637 ,   3.98260215,
#           1.15651735,  -3.14159337], [225.23238423,   0.88443564,  -2.35095983,   3.98268349,
#           1.15660198,  -3.1415945 ], [225.2351721 ,   0.88443266,  -2.35096276,   3.98266687,
#           1.15657392,  -3.1415963 ],[225.25512517,   0.88442378,  -2.3509497 ,   3.9826258 ,
#           1.15652865,  -3.14158844]], 
# [[0.00014031, 0.00014881, 0.00016353, 0.00019007, 0.00021295], [8.20301223e-05, 1.38861094e-04, 2.23914844e-04, 2.67860427e-04,
#         2.42889059e-04], [5.11412366e-05, 1.14793536e-04, 2.29855215e-04, 3.32985508e-04,
#         3.72795462e-04], [0.00011094, 0.0001321 , 0.00018038, 0.00024888, 0.00028527], [6.90323598e-05, 1.25632857e-04, 2.10860192e-04, 2.93384163e-04,
#         3.40600749e-04], [7.76816137e-05, 1.11131697e-04, 1.79494482e-04, 2.95163170e-04,
#         4.38992896e-04],[8.40639978e-05, 1.26136518e-04, 2.00600425e-04, 2.80229529e-04,
#         3.17633359e-04], [4.25638328e-05, 1.13755170e-04, 2.24409520e-04, 3.28960913e-04,
#         4.30160045e-04], [4.98483235e-05, 1.22014686e-04, 2.38504329e-04, 3.18763056e-04,
#         3.28354445e-04],[3.98649368e-05, 1.03617532e-04, 2.16776088e-04, 3.38903526e-04,
#         4.64015460e-04]], 
# [1.1236562259431423e-07, 2.948251663966858e-05, 0.0004899036078155068, 0.00041268977094565046, 5.8154009754610896e-05]]]



#     plotICRobustness(biglist,names,noisefactor=1e-1)




#     # 10 orbits, start from IG3, true initial DM distribution
    # biglist = [[[[225.1445594 ,   0.88448102,  -2.35095614,   3.98281194,
    #       1.15687279,  -3.14160637], [225.29789299,   0.8843991 ,  -2.35094699,   3.98250969,
    #       1.15633682,  -3.14158572], [225.3853529 ,   0.8843545 ,  -2.35094969,   3.98245309,
    #       1.15615237,  -3.14157574], [225.12129068,   0.88449776,  -2.35095526,   3.98295253,
    #       1.15709818,  -3.14160527], [225.12635347,   0.88448746,  -2.35097704,   3.98283736,
    #       1.15693062,  -3.14160627], [225.124494  ,   0.88449119,  -2.3509749 ,   3.98289485,
    #       1.15699397,  -3.14160515], [225.29054247,   0.88440324,  -2.35096043,   3.98259921,
    #       1.15645532,  -3.14158774], [225.46685359,   0.88431005,  -2.35096085,   3.98233703,
    #       1.15581399,  -3.14156207], [225.22961144,   0.884438  ,  -2.35095491,   3.98272263,
    #       1.15662003,  -3.14159317], [225.34180948,   0.88437521,  -2.35096408,   3.98248432,
    #       1.15630382,  -3.14158516]], [[6.06123522e-05, 1.56923110e-04, 2.36360485e-04, 2.49116540e-04,
    #     1.32949912e-04], [5.61648668e-05, 1.04376260e-04, 1.58082624e-04, 2.35873980e-04,
    #     4.44404706e-04], [3.41777739e-06, 4.46821121e-05, 1.78233138e-04, 3.39824094e-04,
    #     6.10298206e-04], [2.77664513e-05, 1.07373992e-04, 2.76059795e-04, 3.73890914e-04,
    #     2.04523504e-04], [0.00012163, 0.00018211, 0.0002087 , 0.00016091, 0.        ], [6.41224612e-05, 1.64570652e-04, 2.66942303e-04, 2.66798321e-04,
    #     0.00000000e+00], [4.50376978e-05, 9.49518930e-05, 1.89241279e-04, 2.89872127e-04,
    #     3.75133750e-04], [8.04801327e-06, 3.99470155e-06, 1.25725304e-04, 3.28212134e-04,
    #     7.89472816e-04], [2.92163073e-05, 1.00589932e-04, 1.90987652e-04, 3.13103502e-04,
    #     4.18764541e-04], [4.44800572e-05, 1.04086925e-04, 1.69997389e-04, 2.51240333e-04,
    #     4.43573325e-04]], [3.609632592453552e-05, 0.0001181406130613886, 0.0002112682471865419, 0.000283560136574258, 0.0003221765543589564]], [[[225.33895987,   0.88437781,  -2.35094303,   3.98246003,
    #       1.15627132,  -3.14158786], [225.07790098,   0.88451819,  -2.35097988,   3.98303663,
    #       1.15718353,  -3.14160781], [225.35671053,   0.88437034,  -2.35094859,   3.98251434,
    #       1.15624439,  -3.14158346], [225.13994896,   0.88448013,  -2.35097342,   3.98277806,
    #       1.15690541,  -3.14160647], [225.10401564,   0.88450464,  -2.35096698,   3.98297863,
    #       1.15708103,  -3.14160596], [225.36097184,   0.88436753,  -2.35094835,   3.98247708,
    #       1.15624264,  -3.14158622], [225.1054858 ,   0.88450482,  -2.3509885 ,   3.98303006,
    #       1.15711021,  -3.14160358], [225.21149397,   0.88444243,  -2.35097592,   3.98267457,
    #       1.15659908,  -3.14159312], [225.08503302,   0.88451339,  -2.35096127,   3.98295928,
    #       1.15714755,  -3.14161091], [225.15879289,   0.88447316,  -2.35097081,   3.98284734,
    #       1.15697135,  -3.14160774]], [[0.000124  , 0.00015201, 0.00017031, 0.00020648, 0.00036058], [0.0001695 , 0.00021432, 0.00024183, 0.00017689, 0.        ], [9.90946874e-05, 1.24123162e-04, 1.53690173e-04, 2.48819759e-04,
    #     5.19392718e-04], [2.43638657e-04, 2.48129697e-04, 1.64714848e-04, 8.07106576e-06,
    #     0.00000000e+00], [0.00013275, 0.00021482, 0.00027058, 0.00020847, 0.        ], [0.00011634, 0.00012771, 0.00015197, 0.0002296 , 0.00047307], [1.59357982e-04, 1.97570754e-04, 2.11390711e-04, 2.14234498e-04,
    #     9.09162969e-05], [2.06143104e-04, 2.03042072e-04, 1.61213723e-04, 1.11226632e-04,
    #     5.66323080e-05], [0.00019295, 0.0002201 , 0.00021766, 0.00014135, 0.        ], [1.82532102e-04, 2.05807926e-04, 2.20096076e-04, 1.70885583e-04,
    #     3.31340071e-07]], [0.00014370151360588277, 0.00017198634513687495, 0.0001905457505668284, 0.00020483338733222063, 0.00021662326462725586]], [[[225.20955146,   0.8844428 ,  -2.35098849,   3.98271584,
    #       1.15663243,  -3.14159372], [225.18056926,   0.88446187,  -2.35094264,   3.98268184,
    #       1.15670797,  -3.14160521], [225.18863552,   0.88445986,  -2.35095893,   3.98280731,
    #       1.15682936,  -3.14159933], [225.26297477,   0.88441679,  -2.35096981,   3.98265699,
    #       1.15655708,  -3.14159323], [225.19529938,   0.88445472,  -2.35096189,   3.98277032,
    #       1.15673672,  -3.14159547], [225.34763399,   0.88437424,  -2.35094896,   3.98249154,
    #       1.15627212,  -3.14158618], [225.09848395,   0.88450556,  -2.35099013,   3.9830118 ,
    #       1.15712189,  -3.14160551], [225.35106205,   0.88437206,  -2.35095033,   3.98248482,
    #       1.15618865,  -3.14158277], [225.12492717,   0.88449239,  -2.35096315,   3.98292045,
    #       1.15698941,  -3.14160449], [225.15722083,   0.88447584,  -2.35097314,   3.98288204,
    #       1.15687448,  -3.14159843]], [[1.01964641e-04, 6.21767689e-05, 3.97234658e-04, 2.52274282e-04,
    #     1.80688385e-06], [3.91374345e-05, 6.79898717e-05, 4.60190630e-04, 3.32073760e-04,
    #     1.71081933e-06], [1.42075604e-06, 3.67747701e-05, 4.86433453e-04, 4.42131211e-04,
    #     4.61520964e-05], [4.25752132e-05, 2.07653061e-05, 4.44742519e-04, 3.83251678e-04,
    #     4.85700174e-05], [1.67865615e-05, 4.84238229e-05, 4.65927936e-04, 4.01008694e-04,
    #     2.51629866e-05], [2.27300828e-06, 1.59242553e-06, 3.66044916e-04, 4.23755504e-04,
    #     3.84654409e-04], [7.39216688e-05, 8.62188465e-05, 4.61399754e-04, 2.52081479e-04,
    #     0.00000000e+00], [0.00000000e+00, 2.43513336e-07, 3.41964489e-04, 4.36962745e-04,
    #     4.01886243e-04], [4.56024652e-05, 5.39893942e-05, 4.83762966e-04, 3.35709419e-04,
    #     3.89995065e-07], [2.42320078e-05, 4.00062872e-05, 4.75879327e-04, 3.97387736e-04,
    #     3.22655318e-05]], [1.1236562259431423e-07, 2.948251663966858e-05, 0.0004899036078155068, 0.00041268977094565046, 5.8154009754610896e-05]]]

    
    
    # plotICRobustness(biglist,names,noisefactor=1)
    
    
    

    
    # # #1e-1 noise: (figure a)
    biglist = [[[[225.24388206,   0.88442599,  -2.35096579,   3.9826981 ,
                  1.15668922,  -3.14166907], [225.23769515,   0.88443258,  -2.35095633,   3.98272148,
                  1.15665682,  -3.14160836], [225.23890218,   0.88442977,  -2.35096627,   3.98269485,
                  1.15665357,  -3.14163448], [225.25340575,   0.88442562,  -2.35097975,   3.98266945,
                  1.15649704,  -3.14151312], [225.25565704,   0.88442229,  -2.35096952,   3.98267161,
                  1.15655526,  -3.14156124], [225.25491169,   0.88442122,  -2.35095647,   3.98265682,
                  1.15655428,  -3.14155953], [225.25315681,   0.88442345,  -2.35096717,   3.98265498,
                  1.15655061,  -3.14156491], [225.23893323,   0.88442782,  -2.3509689 ,   3.98267727,
                  1.15667289,  -3.14166271], [225.23226865,   0.88442937,  -2.35094078,   3.98269883,
                  1.15683962,  -3.14179656], [225.24026667,   0.88442695,  -2.35096855,   3.98270293,
                  1.1567073 ,  -3.14168382]], [[0.        , 0.00028559, 0.        , 0.        , 0.00139881], [0.00000000e+00, 7.97455397e-05, 3.66237461e-04, 2.23447974e-04,
                4.37748330e-04], [1.63121897e-04, 3.63719564e-07, 9.01420754e-05, 1.79560828e-04,
                9.44645166e-04], [0.        , 0.00030825, 0.00015062, 0.        , 0.        ], [7.42330751e-05, 2.05658407e-08, 4.18244420e-04, 1.79359906e-04,
                0.00000000e+00], [4.51635935e-06, 4.63710906e-06, 6.71854560e-04, 4.91880668e-06,
                5.00799168e-06], [1.34116591e-04, 5.85080086e-07, 2.22133909e-04, 3.39214345e-04,
                0.00000000e+00], [1.97464522e-04, 1.78698846e-05, 4.61582620e-05, 0.00000000e+00,
                1.33877562e-03], [0.00014617, 0.        , 0.        , 0.        , 0.00257377], [1.38238235e-04, 6.55425081e-05, 3.76682881e-05, 0.00000000e+00,
                1.55551415e-03]], [3.609632592453552e-05, 0.0001181406130613886, 0.0002112682471865419, 0.000283560136574258, 0.0003221765543589564]], [[[225.24856447,   0.88442693,  -2.35096165,   3.98267997,
                  1.15657965,  -3.14157332], [225.25014522,   0.88442818,  -2.35096985,   3.98266743,
                  1.15654098,  -3.14155403], [225.24776425,   0.88442749,  -2.35098142,   3.98264305,
                  1.15648118,  -3.14151623], [225.23760705,   0.88442969,  -2.35096516,   3.98266776,
                  1.156624  ,  -3.14162789], [225.23520002,   0.88442969,  -2.35096552,   3.9826883 ,
                  1.15669476,  -3.14167482], [225.23553952,   0.88443323,  -2.35095319,   3.98273597,
                  1.15671436,  -3.14165258], [225.24891138,   0.88442704,  -2.35096754,   3.98268016,
                  1.15656127,  -3.14155859], [225.24372971,   0.88442672,  -2.35096769,   3.98266357,
                  1.15661009,  -3.14161639], [225.24705712,   0.88442557,  -2.35097537,   3.98268173,
                  1.1565883 ,  -3.14158772], [225.24939856,   0.88442332,  -2.35096307,   3.98263326,
                  1.15657476,  -3.14161406]], [[0.00016212, 0.00010515, 0.00027123, 0.00022429, 0.        ], [6.13947081e-05, 3.62869680e-04, 1.92502668e-04, 3.95585584e-07,
                0.00000000e+00], [3.54029986e-04, 8.68997797e-05, 0.00000000e+00, 0.00000000e+00,
                0.00000000e+00], [1.91762170e-04, 2.31338379e-04, 3.18844484e-05, 0.00000000e+00,
                7.78000088e-04], [0.00020375, 0.00020344, 0.        , 0.        , 0.0012043 ], [1.49078184e-04, 9.39299476e-05, 2.52848367e-04, 6.03167138e-05,
                9.62859446e-04], [0.00021004, 0.00010515, 0.0001679 , 0.0002108 , 0.        ], [2.44391452e-04, 7.91815450e-05, 2.00474831e-04, 0.00000000e+00,
                5.91439977e-04], [3.20450981e-04, 0.00000000e+00, 1.84826489e-07, 3.95871272e-04,
                1.32157812e-04], [2.29574139e-04, 6.95000900e-05, 2.03671022e-04, 1.35258756e-04,
                3.85495726e-04]], [0.00014370151360588277, 0.00017198634513687495, 0.0001905457505668284, 0.00020483338733222063, 0.00021662326462725586]], [[[225.24948156,   0.88442669,  -2.35098266,   3.98265888,
                  1.15649809,  -3.14152422], [225.25949603,   0.88441797,  -2.35096635,   3.98264585,
                  1.15658959,  -3.14161834], [225.23498254,   0.88442728,  -2.35097865,   3.98270824,
                  1.15676394,  -3.14173332], [225.25684036,   0.88442325,  -2.35096705,   3.98265903,
                  1.15652751,  -3.14154571], [225.22968878,   0.8844366 ,  -2.3509594 ,   3.98271472,
                  1.15664431,  -3.14160335], [225.24301344,   0.88443124,  -2.35096111,   3.98266064,
                  1.15652921,  -3.1415413 ], [225.25444587,   0.88442455,  -2.35096645,   3.98266671,
                  1.15653885,  -3.14154811], [225.24271577,   0.88443065,  -2.35095993,   3.98268842,
                  1.15658587,  -3.14157062], [225.23741911,   0.88443427,  -2.35096126,   3.98272459,
                  1.15662075,  -3.14157114], [225.24668132,   0.88442882,  -2.35096903,   3.98270218,
                  1.15658968,  -3.14156446]], [[1.46291469e-06, 2.31352291e-04, 3.59351057e-04, 0.00000000e+00,
                0.00000000e+00], [0.        , 0.        , 0.00063887, 0.00016227, 0.00034773], [9.16684272e-05, 1.71262839e-04, 0.00000000e+00, 0.00000000e+00,
                2.00159493e-03], [0.00000000e+00, 1.03630920e-05, 7.12882519e-04, 0.00000000e+00,
                0.00000000e+00], [0.00000000e+00, 1.35117824e-05, 6.29336003e-04, 8.60184781e-05,
                4.31127266e-04], [0.00000000e+00, 5.39186863e-05, 6.51979126e-04, 0.00000000e+00,
                0.00000000e+00], [0.        , 0.        , 0.00073625, 0.        , 0.        ], [0.        , 0.        , 0.00062525, 0.0002436 , 0.        ], [0.        , 0.        , 0.00058619, 0.00031847, 0.        ], [5.11546500e-07, 8.52154996e-07, 6.15510029e-04, 2.34842680e-04,
                0.00000000e+00]], [1.1236562259431423e-07, 2.948251663966858e-05, 0.0004899036078155068, 0.00041268977094565046, 5.8154009754610896e-05]]]
    
    plotICRobustness(biglist,names,noisefactor=1e-1)
    
    # #1 noise (figure b)
    biglist = [[[[225.17613775,   0.88443196,  -2.35091802,   3.9826914 ,
              1.15744177,  -3.14241506], [225.29161745,   0.88437399,  -2.35094745,   3.98262678,
              1.15726061,  -3.14228768], [225.37469537,   0.88433151,  -2.35103361,   3.98266479,
              1.1570151 ,  -3.14208775], [225.13465792,   0.884498  ,  -2.35088777,   3.98309394,
              1.15723453,  -3.14179199], [225.17334969,   0.88442285,  -2.35087121,   3.98267483,
              1.15777745,  -3.1426549 ], [225.15993078,   0.88444357,  -2.35095011,   3.98297454,
              1.15770893,  -3.14239688], [225.30803636,   0.88436685,  -2.35095917,   3.98263759,
              1.15721892,  -3.14224479], [225.43195007,   0.88433081,  -2.35101248,   3.98240406,
              1.1561397 ,  -3.1414448 ], [225.24079234,   0.88441684,  -2.35095683,   3.98272658,
              1.15697782,  -3.14197913], [225.32181708,   0.8843784 ,  -2.35106258,   3.98261292,
              1.15640861,  -3.14154978]], [[0.        , 0.        , 0.        , 0.        , 0.00682684], [0.        , 0.        , 0.        , 0.        , 0.00600773], [0.        , 0.        , 0.        , 0.        , 0.00449904], [0.        , 0.        , 0.        , 0.        , 0.00350058], [0.        , 0.        , 0.        , 0.        , 0.00835375], [0.        , 0.        , 0.        , 0.        , 0.00687317], [0.        , 0.        , 0.        , 0.        , 0.00574049], [0.00019214, 0.        , 0.        , 0.        , 0.        ], [0.        , 0.        , 0.        , 0.        , 0.00404087], [0.00000000e+00, 4.14748526e-04, 1.42461966e-08, 0.00000000e+00,
            0.00000000e+00]], [3.609632592453552e-05, 0.0001181406130613886, 0.0002112682471865419, 0.000283560136574258, 0.0003221765543589564]], [[[225.35346264,   0.88434746,  -2.350966  ,   3.9824378 ,
              1.15675448,  -3.14196653], [225.11181028,   0.88447464,  -2.35088211,   3.9830699 ,
              1.1579604 ,  -3.14252734], [225.32249794,   0.8843955 ,  -2.35092253,   3.98240807,
              1.15620303,  -3.14145644], [225.16832618,   0.88444755,  -2.35090484,   3.98310121,
              1.15789202,  -3.1424218 ], [225.14822266,   0.88446057,  -2.35089571,   3.98299205,
              1.15762879,  -3.14230422], [225.34633547,   0.88441655,  -2.35087097,   3.98272131,
              1.15628015,  -3.14122748], [225.14021635,   0.88445342,  -2.35097105,   3.98310426,
              1.15781673,  -3.14246182], [225.2319422 ,   0.88442079,  -2.3510134 ,   3.98254116,
              1.15651814,  -3.14164466], [225.12100921,   0.88448919,  -2.35086746,   3.98300381,
              1.15748799,  -3.14210411], [225.19027144,   0.88446178,  -2.3509415 ,   3.98278337,
              1.15672726,  -3.14159096]], [[0.00015913, 0.00014436, 0.        , 0.        , 0.00306885], [0.        , 0.        , 0.        , 0.        , 0.00819612], [2.72735175e-04, 6.29430762e-05, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00], [0.        , 0.        , 0.        , 0.        , 0.00749131], [0.        , 0.        , 0.        , 0.        , 0.00651676], [8.88771984e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00], [0.       , 0.       , 0.       , 0.       , 0.0074307], [5.32549338e-04, 2.49562101e-06, 0.00000000e+00, 0.00000000e+00,
            2.47588138e-04], [0.00010322, 0.        , 0.        , 0.        , 0.00525354], [0.00036262, 0.        , 0.        , 0.        , 0.00089805]], [0.00014370151360588277, 0.00017198634513687495, 0.0001905457505668284, 0.00020483338733222063, 0.00021662326462725586]], [[[225.22874921,   0.88440155,  -2.35094202,   3.98269597,
              1.15750483,  -3.14246578], [225.18574066,   0.88446994,  -2.35094293,   3.98271002,
              1.15650197,  -3.14143317], [225.20364972,   0.88444648,  -2.35096918,   3.98286543,
              1.15697051,  -3.14176601], [225.26471649,   0.88439246,  -2.3509832 ,   3.98282254,
              1.15732898,  -3.14219971], [225.19543679,   0.88445911,  -2.35088873,   3.98266731,
              1.15668522,  -3.14161223], [225.32042453,   0.88440337,  -2.35098591,   3.98268087,
              1.15635823,  -3.14136731], [225.12545175,   0.88446328,  -2.35083656,   3.98286587,
              1.15790447,  -3.14255326], [225.30870853,   0.88441131,  -2.35097073,   3.98262758,
              1.15624174,  -3.14134911], [225.14809634,   0.88448723,  -2.35091141,   3.98272973,
              1.15665715,  -3.14149171], [225.18793043,   0.88448377,  -2.35096224,   3.9829577 ,
              1.15656287,  -3.14131957]], [[0.        , 0.        , 0.        , 0.        , 0.00739974], [0.00031997, 0.        , 0.        , 0.        , 0.        ], [0.0001918 , 0.        , 0.        , 0.        , 0.00252206], [0.        , 0.        , 0.        , 0.        , 0.00563732], [0.        , 0.        , 0.00052084, 0.00063916, 0.        ], [0.00022359, 0.        , 0.        , 0.        , 0.        ], [0.        , 0.        , 0.        , 0.        , 0.00832728], [0.00014449, 0.        , 0.        , 0.        , 0.        ], [0.00015302, 0.00036932, 0.        , 0.        , 0.        ], [0.00019983, 0.        , 0.        , 0.        , 0.        ]], [1.1236562259431423e-07, 2.948251663966858e-05, 0.0004899036078155068, 0.00041268977094565046, 5.8154009754610896e-05]]]
    

    plotICRobustness(biglist,names,noisefactor=1)
    
    
    
    
    # #1 noise, 10x observations in 1 orbit, 2k iters:
    # biglist = [[[[225.2596557 ,   0.88442674,  -2.35097084,   3.98265397,
    #           1.15643737,  -3.14145945], [225.13552606,   0.8844819 ,  -2.3508924 ,   3.98283971,
    #           1.15707614,  -3.1418489 ], [224.97774986,   0.88457498,  -2.35085537,   3.982917  ,
    #           1.15702025,  -3.14165489], [225.01723938,   0.88454078,  -2.35085992,   3.98302305,
    #           1.15745014,  -3.14201368], [225.17870641,   0.88446652,  -2.35090756,   3.98284646,
    #           1.15692288,  -3.14171116], [224.99417054,   0.88455537,  -2.35084905,   3.98305349,
    #           1.15744269,  -3.14197102], [225.15040406,   0.8844846 ,  -2.35094407,   3.98279649,
    #           1.1566547 ,  -3.14149934], [225.28605502,   0.88439065,  -2.35101238,   3.98288791,
    #           1.15713399,  -3.14197067], [225.18972892,   0.88445941,  -2.35090593,   3.982881  ,
    #           1.15702406,  -3.14178678], [225.23121067,   0.88443265,  -2.35096125,   3.98262985,
    #           1.15656858,  -3.14160952]], [[1.78399914e-04, 1.14136554e-04, 3.35253575e-06, 0.00000000e+00,
    #         0.00000000e+00], [0.        , 0.        , 0.00025141, 0.00075265, 0.00172685], [0.00012909, 0.00017789, 0.0002769 , 0.00033518, 0.00040398], [0.        , 0.        , 0.00014314, 0.00105074, 0.00260468], [0.        , 0.        , 0.00022838, 0.00063357, 0.00114278], [0.        , 0.        , 0.00013715, 0.00104898, 0.0023294 ], [1.77771661e-04, 1.60258877e-04, 7.69604282e-05, 3.93560589e-05,
    #         2.19277303e-06], [0.        , 0.        , 0.        , 0.00046446, 0.00325816], [3.21711803e-08, 1.08088949e-07, 0.00000000e+00, 6.91910904e-04,
    #         2.02243731e-03], [7.55981052e-05, 1.31725588e-04, 1.91196067e-04, 2.50957111e-04,
    #         3.10236008e-04]], [3.609632592453552e-05, 0.0001181406130613886, 0.0002112682471865419, 0.000283560136574258, 0.0003221765543589564]], [[[225.17230351,   0.88446176,  -2.3509343 ,   3.9827995 ,
    #           1.15689088,  -3.14173948], [225.28144881,   0.88440255,  -2.35097873,   3.98270228,
    #           1.15672435,  -3.14172435], [225.07160627,   0.88452238,  -2.35084541,   3.98292935,
    #           1.15716692,  -3.14183018], [225.53046681,   0.8842698 ,  -2.35106892,   3.98240818,
    #           1.15617108,  -3.14155392], [225.21719067,   0.88443867,  -2.35096267,   3.98282969,
    #           1.15687684,  -3.14174113], [225.2558271 ,   0.88442107,  -2.35095738,   3.98262376,
    #           1.15655144,  -3.14159274], [225.18349627,   0.88446643,  -2.35090774,   3.9826006 ,
    #           1.15649732,  -3.14150908], [225.18402078,   0.88445624,  -2.35092617,   3.98265224,
    #           1.15669592,  -3.14167245], [225.16627384,   0.88447429,  -2.35091564,   3.9828351 ,
    #           1.15682805,  -3.14164235], [225.20059927,   0.88445358,  -2.35093752,   3.98266427,
    #           1.15657793,  -3.14155333]], [[8.41552453e-05, 1.59999714e-04, 2.67048479e-04, 4.48190288e-04,
    #         8.25145052e-04], [9.43815436e-05, 9.92764149e-05, 1.86628720e-04, 3.56859704e-04,
    #         1.09604036e-03], [0.00000000e+00, 6.45432547e-05, 3.88774918e-04, 8.20263807e-04,
    #         1.37722571e-03], [9.76436569e-05, 7.36369072e-05, 6.76044715e-05, 1.34552563e-04,
    #         3.59996528e-04], [5.66088372e-05, 1.16086616e-04, 2.35107105e-04, 4.81782167e-04,
    #         1.06152008e-03], [1.48836585e-04, 1.70093882e-04, 1.98759380e-04, 2.77341453e-04,
    #         2.68356421e-05], [2.88891789e-04, 1.91830224e-04, 1.11645527e-05, 0.00000000e+00,
    #         0.00000000e+00], [0.000157  , 0.00018509, 0.00024295, 0.00030968, 0.00034824], [7.47684891e-05, 1.33305515e-04, 2.63797134e-04, 4.80514022e-04,
    #         4.25818855e-04], [2.00134246e-04, 2.16855153e-04, 1.64034622e-04, 5.14134095e-05,
    #         4.92485581e-07]], [0.00014370151360588277, 0.00017198634513687495, 0.0001905457505668284, 0.00020483338733222063, 0.00021662326462725586]], [[[225.38638678,   0.88435522,  -2.35098248,   3.98234795,
    #           1.15610442,  -3.14143978], [225.13472703,   0.88448049,  -2.35091343,   3.98282923,
    #           1.15702219,  -3.14182733], [225.16236018,   0.88446605,  -2.35092684,   3.98277236,
    #           1.15691009,  -3.14177372], [225.15142323,   0.88447762,  -2.35092407,   3.98276707,
    #           1.15679469,  -3.14166243], [225.1121772 ,   0.88450299,  -2.35092825,   3.98270059,
    #           1.15658337,  -3.1415026 ], [225.00279637,   0.88456257,  -2.35084487,   3.98288352,
    #           1.15699806,  -3.14166333], [225.1294258 ,   0.88448354,  -2.35090989,   3.98290558,
    #           1.15715946,  -3.14189077], [225.00702526,   0.88455014,  -2.35085681,   3.98296765,
    #           1.15728761,  -3.14190214], [225.19525848,   0.88444661,  -2.35091611,   3.98273452,
    #           1.15696436,  -3.14185395], [225.41971498,   0.88433402,  -2.35103008,   3.98242278,
    #           1.15615411,  -3.14146485]], [[0.00022741, 0.        , 0.        , 0.        , 0.        ], [0.        , 0.        , 0.00031923, 0.00081146, 0.00142654], [0.        , 0.00010097, 0.00024229, 0.00066989, 0.00116277], [2.77578526e-05, 1.28032012e-04, 2.87573648e-04, 4.60107370e-04,
    #         5.99198005e-04], [2.80376304e-04, 1.32371267e-04, 5.01168413e-05, 2.74413115e-05,
    #         0.00000000e+00], [4.22872854e-05, 1.71668109e-04, 3.72277556e-04, 5.38803386e-04,
    #         3.81837658e-04], [0.        , 0.        , 0.00013488, 0.00088643, 0.00218167], [0.        , 0.        , 0.00038   , 0.00092548, 0.0017688 ], [0.        , 0.        , 0.00027763, 0.00073062, 0.00180504], [9.97977185e-05, 1.08871121e-04, 8.05466716e-05, 0.00000000e+00,
    #         0.00000000e+00]], [1.1236562259431423e-07, 2.948251663966858e-05, 0.0004899036078155068, 0.00041268977094565046, 5.8154009754610896e-05]]]
    
    #1 orbit, 10xobs 10k iters: (figure d)
    biglist = [[[[225.25561319,   0.88442869,  -2.35097124,   3.98264581,
              1.15642794,  -3.14145582], [225.14614078,   0.88446155,  -2.35089528,   3.98290317,
              1.15754145,  -3.14226463], [224.99503643,   0.884534  ,  -2.35085956,   3.9831384 ,
              1.15807867,  -3.14253254], [225.0416201 ,   0.88451204,  -2.35087221,   3.98307156,
              1.1579061 ,  -3.14243951], [225.18395224,   0.88445498,  -2.35089854,   3.98290236,
              1.15728231,  -3.14202009], [225.01835448,   0.88452385,  -2.35087248,   3.98311682,
              1.15794031,  -3.14243175], [225.15571964,   0.88446788,  -2.35091183,   3.98299142,
              1.15738802,  -3.14205178], [225.2805112 ,   0.88440927,  -2.3509984 ,   3.98260873,
              1.15641658,  -3.14148979], [225.19465769,   0.8844629 ,  -2.35091684,   3.98280182,
              1.15671706,  -3.14155572], [225.23298642,   0.88442499,  -2.35095582,   3.98268241,
              1.15684654,  -3.14183965]], [[0.00026843, 0.        , 0.        , 0.        , 0.        ], [0.        , 0.        , 0.        , 0.        , 0.00615754], [0.        , 0.        , 0.        , 0.        , 0.00797493], [0.        , 0.        , 0.        , 0.        , 0.00733935], [0.        , 0.        , 0.        , 0.        , 0.00471236], [0.        , 0.        , 0.        , 0.        , 0.00722512], [0.        , 0.        , 0.        , 0.        , 0.00488505], [0.00029401, 0.        , 0.        , 0.        , 0.        ], [0.        , 0.        , 0.00044831, 0.00040838, 0.        ], [0.00014947, 0.        , 0.        , 0.        , 0.00276585]], [3.609632592453552e-05, 0.0001181406130613886, 0.0002112682471865419, 0.000283560136574258, 0.0003221765543589564]], [[[225.18057678,   0.88444181,  -2.35089898,   3.98295818,
              1.15765117,  -3.14234954], [225.27830919,   0.88440782,  -2.3509879 ,   3.98263965,
              1.15650831,  -3.14156877], [225.0897708 ,   0.88449161,  -2.35084323,   3.98303953,
              1.15785844,  -3.14242925], [225.51411988,   0.88429043,  -2.35100232,   3.98230692,
              1.15599812,  -3.14142135], [225.21795577,   0.88443561,  -2.35097887,   3.9827724 ,
              1.1568315 ,  -3.14175313], [225.25410091,   0.88441797,  -2.3509576 ,   3.98264025,
              1.15668509,  -3.14171161], [225.18992732,   0.88444259,  -2.35088886,   3.98294727,
              1.15757302,  -3.14227626], [225.19128583,   0.88443511,  -2.35088473,   3.98288442,
              1.15762774,  -3.14238434], [225.17415314,   0.88446143,  -2.3509081 ,   3.98289071,
              1.15718268,  -3.14194768], [225.2039703 ,   0.88444361,  -2.35092277,   3.98276726,
              1.15698948,  -3.14187101]], [[0.        , 0.        , 0.        , 0.        , 0.00687447], [1.73532901e-07, 4.25505640e-04, 2.20940818e-04, 0.00000000e+00,
            0.00000000e+00], [0.        , 0.        , 0.        , 0.        , 0.00759034], [0.00019597, 0.        , 0.        , 0.        , 0.        ], [0.00030992, 0.        , 0.        , 0.        , 0.00190244], [2.70024378e-04, 3.06180001e-05, 5.16567555e-05, 9.71651829e-05,
            1.40010636e-03], [0.        , 0.        , 0.        , 0.        , 0.00651466], [0.        , 0.        , 0.        , 0.        , 0.00711833], [0.0001598 , 0.        , 0.        , 0.        , 0.00388341], [2.36661467e-04, 5.68318465e-07, 0.00000000e+00, 0.00000000e+00,
            3.01239760e-03]], [0.00014370151360588277, 0.00017198634513687495, 0.0001905457505668284, 0.00020483338733222063, 0.00021662326462725586]], [[[225.37381967,   0.88435908,  -2.35100103,   3.98238976,
              1.15615749,  -3.14146097], [225.14520046,   0.88446199,  -2.35090464,   3.98290014,
              1.15751095,  -3.14225396], [225.17314549,   0.88444836,  -2.35090883,   3.98285714,
              1.15742554,  -3.14221205], [225.16216921,   0.88445976,  -2.350899  ,   3.98287969,
              1.1573766 ,  -3.14213961], [225.1232691 ,   0.88447719,  -2.35090697,   3.98298166,
              1.15754805,  -3.14221782], [225.01924802,   0.88452754,  -2.35084317,   3.98306317,
              1.15791455,  -3.14243165], [225.13878736,   0.88446789,  -2.35090651,   3.98294928,
              1.15753118,  -3.1422262 ], [225.03100596,   0.88451895,  -2.35086289,   3.98305452,
              1.15789246,  -3.14244115], [225.20242795,   0.88443336,  -2.35090431,   3.98278687,
              1.15735301,  -3.14219471], [225.41039293,   0.8843439 ,  -2.35099719,   3.98237721,
              1.15609647,  -3.14142338]], [[0.0002551, 0.       , 0.       , 0.       , 0.       ], [0.        , 0.        , 0.        , 0.        , 0.00623636], [0.        , 0.        , 0.        , 0.        , 0.00594766], [2.11179148e-06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            5.56604219e-03], [0.        , 0.        , 0.        , 0.        , 0.00608238], [0.        , 0.        , 0.        , 0.        , 0.00756404], [0.        , 0.        , 0.        , 0.        , 0.00610812], [0.        , 0.        , 0.        , 0.        , 0.00756164], [9.95901635e-06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            5.82291692e-03], [0.0001977, 0.       , 0.       , 0.       , 0.       ]], [1.1236562259431423e-07, 2.948251663966858e-05, 0.0004899036078155068, 0.00041268977094565046, 5.8154009754610896e-05]]]
    plotICRobustness(biglist,names,noisefactor=1)
    
    
    # # No noise:
    #plotICRobustness(biglist,names,noisefactor=0)
    
    
    
    #New results 27/3:
        
    #1e-1 precision, 10x observations in 1 orbit (figure c)
    biglist = [[[[225.23817387,   0.88443119,  -2.35096066,   3.98269364,
             1.15664699,  -3.14162625], [225.22536047,   0.8844332 ,  -2.35095748,   3.98271707,
             1.15678462,  -3.14174015], [225.24925667,   0.88442679,  -2.35096539,   3.98268371,
             1.15657263,  -3.14156538], [225.268125  ,   0.88441686,  -2.35098043,   3.98262429,
             1.15645292,  -3.14151232], [225.237973  ,   0.88442956,  -2.35096343,   3.9826943 ,
             1.15667302,  -3.14164967], [225.23938284,   0.88442868,  -2.35096349,   3.98268945,
             1.1566643 ,  -3.14164671], [225.23505407,   0.88443297,  -2.3509586 ,   3.98271137,
             1.15666415,  -3.14162665], [225.24924553,   0.88442718,  -2.35096232,   3.98269022,
             1.1565792 ,  -3.14156442], [225.24055582,   0.884431  ,  -2.35095978,   3.98270399,
             1.15662819,  -3.14159832], [225.25766243,   0.88442314,  -2.35097035,   3.982664  ,
             1.15651641,  -3.14153139]], [[4.64019053e-05, 1.24302877e-04, 2.13170047e-04, 1.05511183e-04,
           7.76869560e-04], [1.73231328e-04, 1.95815502e-05, 5.72611876e-08, 0.00000000e+00,
           2.02692907e-03], [1.93369470e-05, 1.37123381e-04, 2.12653024e-04, 3.66378024e-04,
           0.00000000e+00], [7.93609383e-06, 3.21310822e-04, 1.00157984e-04, 0.00000000e+00,
           0.00000000e+00], [7.30384696e-05, 1.44644734e-04, 1.21795752e-04, 2.51096744e-08,
           1.16393751e-03], [6.28270433e-05, 1.53045111e-04, 1.38007125e-04, 7.41145467e-06,
           1.09870613e-03], [2.39534192e-05, 1.28363708e-04, 2.33822850e-04, 1.31001945e-04,
           7.75839509e-04], [6.69628621e-06, 1.10016380e-04, 3.03338266e-04, 3.19698086e-04,
           0.00000000e+00], [1.92054393e-05, 1.36777600e-04, 2.26878829e-04, 2.14278779e-04,
           4.63081520e-04], [3.23315847e-06, 1.55740911e-04, 3.90962899e-04, 0.00000000e+00,
           0.00000000e+00]], [3.609632592453552e-05, 0.0001181406130613886, 0.0002112682471865419, 0.000283560136574258, 0.0003221765543589564]], [[[225.26048755,   0.88442101,  -2.35097415,   3.98264069,
             1.15649259,  -3.14153112], [225.2577512 ,   0.88442226,  -2.35096822,   3.98266893,
             1.15655874,  -3.14156143], [225.27217436,   0.88441375,  -2.35098717,   3.982609  ,
             1.15644108,  -3.14151699], [225.22650087,   0.88443484,  -2.35095602,   3.98271532,
             1.15674426,  -3.14169934], [225.24314375,   0.88442858,  -2.35096593,   3.9826919 ,
             1.15661384,  -3.14159836], [225.25258086,   0.88442502,  -2.35097052,   3.98266698,
             1.15653484,  -3.14154592], [225.24952118,   0.88442713,  -2.35096544,   3.98267342,
             1.15655096,  -3.14155282], [225.24192169,   0.88443003,  -2.35096104,   3.98269513,
             1.15661375,  -3.14159192], [225.25101401,   0.88442519,  -2.35096622,   3.9826752 ,
             1.15657153,  -3.14157114], [225.26607921,   0.88441802,  -2.35097805,   3.98262436,
             1.15646548,  -3.14152104]], [[1.46029186e-04, 3.01390689e-04, 8.53544747e-05, 0.00000000e+00,
           0.00000000e+00], [9.19665366e-05, 1.88045289e-04, 3.90621486e-04, 2.05001351e-05,
           0.00000000e+00], [0.0003191, 0.0001168, 0.       , 0.       , 0.       ], [0.00021398, 0.00015397, 0.        , 0.        , 0.00149826], [0.00015071, 0.00019383, 0.00012862, 0.00020535, 0.00029355], [0.00015058, 0.0002014 , 0.00025335, 0.        , 0.        ], [1.24622316e-04, 1.77791621e-04, 3.50987798e-04, 1.63187321e-08,
           0.00000000e+00], [0.0001298 , 0.00019307, 0.00021003, 0.0001543 , 0.00025196], [0.00013518, 0.00017749, 0.00023977, 0.00018439, 0.        ], [0.00018446, 0.00029618, 0.        , 0.        , 0.        ]], [0.00014370151360588277, 0.00017198634513687495, 0.0001905457505668284, 0.00020483338733222063, 0.00021662326462725586]], [[[225.23311754,   0.88443219,  -2.35095993,   3.98269652,
             1.15668172,  -3.14165774], [225.26744243,   0.88441663,  -2.35097994,   3.98260519,
             1.15644011,  -3.14151735], [225.23349869,   0.88443227,  -2.35095939,   3.98270119,
             1.15667554,  -3.14165018], [225.2522822 ,   0.88442492,  -2.35096516,   3.98266584,
             1.15655515,  -3.14156387], [225.24276092,   0.88443055,  -2.35096292,   3.98269311,
             1.15658619,  -3.14156729], [225.26880604,   0.88441546,  -2.35098317,   3.98260274,
             1.15643874,  -3.14151817], [225.22811121,   0.88443236,  -2.35096176,   3.98270867,
             1.15675145,  -3.14171537], [225.2411062 ,   0.88442854,  -2.35096322,   3.98268899,
             1.1566485 ,  -3.14163322], [225.25421146,   0.8844246 ,  -2.35097061,   3.98264739,
             1.15650625,  -3.14153518], [225.24548514,   0.88442855,  -2.35096334,   3.98268364,
             1.15658456,  -3.1415746 ]], [[9.85632697e-07, 1.67723969e-04, 3.00479618e-04, 8.31895589e-05,
           1.06422855e-03], [2.77628267e-07, 3.01298683e-04, 2.43006894e-04, 0.00000000e+00,
           0.00000000e+00], [5.54028858e-07, 1.44516470e-04, 3.33099013e-04, 1.22443096e-04,
           9.43138542e-04], [0.00000000e+00, 1.61812279e-05, 6.49212203e-04, 1.44184227e-04,
           0.00000000e+00], [0.00000000e+00, 2.02258562e-05, 6.00786630e-04, 2.27651980e-04,
           0.00000000e+00], [6.88606205e-08, 3.08775642e-04, 2.28000727e-04, 0.00000000e+00,
           0.00000000e+00], [1.56850571e-05, 2.23447178e-04, 1.35231414e-04, 8.01219982e-08,
           1.72069145e-03], [0.        , 0.00010973, 0.00036273, 0.00025438, 0.0006327 ], [0.        , 0.00012087, 0.00054111, 0.        , 0.        ], [0.00000000e+00, 4.00096162e-07, 6.26289179e-04, 2.47387066e-04,
           0.00000000e+00]], [1.1236562259431423e-07, 2.948251663966858e-05, 0.0004899036078155068, 0.00041268977094565046, 5.8154009754610896e-05]]]

    plotICRobustness(biglist,names,noisefactor=1e-1)


    #1e-1 precision (IG2), 10 orbits (figure e)
    biglist = [[[[225.25586595,   0.88442105,  -2.35096412,   3.98265892,
          1.15655064,  -3.14159021], [225.26871897,   0.88441484,  -2.35096328,   3.98263279,
          1.15650644,  -3.14159013], [225.23028356,   0.88443498,  -2.35096454,   3.9827024 ,
          1.15664179,  -3.14159393], [225.23386603,   0.88443301,  -2.35096679,   3.9826936 ,
          1.15661819,  -3.14159305], [225.23347245,   0.88443324,  -2.3509679 ,   3.98269621,
          1.1566214 ,  -3.14159314], [225.241957  ,   0.88442873,  -2.35096702,   3.982682  ,
          1.15659605,  -3.14159267], [225.2165461 ,   0.88444268,  -2.35096555,   3.98272575,
          1.15667875,  -3.14159489], [225.24213115,   0.88442883,  -2.35096646,   3.98268252,
          1.15659129,  -3.14159179], [225.25496737,   0.88442201,  -2.35096447,   3.98265494,
          1.15654374,  -3.14159102], [225.24706367,   0.88442623,  -2.35096531,   3.98266965,
          1.15657812,  -3.14159207]], [[6.53911373e-05, 1.04044599e-04, 1.76001069e-04, 2.62832394e-04,
        3.63410227e-04], [6.09440808e-05, 1.00722818e-04, 1.72615280e-04, 2.63890216e-04,
        3.84599834e-04], [7.29729860e-05, 1.11849615e-04, 1.80391729e-04, 2.56338251e-04,
        3.20874055e-04], [7.52638332e-05, 1.13059833e-04, 1.78149555e-04, 2.49445716e-04,
        3.11370695e-04], [7.53486862e-05, 1.13009989e-04, 1.78593346e-04, 2.49709431e-04,
        3.10926995e-04], [7.51753344e-05, 1.09854843e-04, 1.72822825e-04, 2.49532805e-04,
        3.32353461e-04], [7.36972809e-05, 1.14490796e-04, 1.85812109e-04, 2.56524955e-04,
        2.97959663e-04], [7.03653024e-05, 1.08441400e-04, 1.76996876e-04, 2.56965968e-04,
        3.33498042e-04], [6.54510325e-05, 1.04915241e-04, 1.76104092e-04, 2.60173116e-04,
        3.53973589e-04], [6.96572292e-05, 1.07643976e-04, 1.75630321e-04, 2.56720455e-04,
        3.43086740e-04]], [3.609632592453552e-05, 0.0001181406130613886, 0.0002112682471865419, 0.000283560136574258, 0.0003221765543589564]], [[[225.22671062,   0.88443735,  -2.35096456,   3.98271174,
          1.15665968,  -3.14159464], [225.25575951,   0.88442238,  -2.35096142,   3.98266867,
          1.1565651 ,  -3.14159161], [225.22734594,   0.88443744,  -2.35096294,   3.98271683,
          1.1566606 ,  -3.14159468], [225.27375157,   0.88441266,  -2.35096221,   3.98263763,
          1.1565081 ,  -3.14158951], [225.23935264,   0.88443085,  -2.35096511,   3.98269984,
          1.1566248 ,  -3.14159329], [225.25792534,   0.88442094,  -2.35096342,   3.98266434,
          1.15656426,  -3.14159114], [225.24172013,   0.88442964,  -2.35096303,   3.98268303,
          1.15660832,  -3.14159349], [225.27676091,   0.88441112,  -2.35096016,   3.98262979,
          1.15649848,  -3.14158896], [225.25273854,   0.88442339,  -2.35096142,   3.98266007,
          1.15657845,  -3.14159308], [225.2376393 ,   0.88443158,  -2.35096502,   3.98269772,
          1.15662545,  -3.14159319]], [[0.00016663, 0.00017209, 0.00018124, 0.00018831, 0.00017943], [0.00014985, 0.00015997, 0.00018004, 0.00020637, 0.00024547], [0.00015881, 0.00016934, 0.00018559, 0.00019774, 0.00019662], [0.000148  , 0.00015596, 0.00017388, 0.0002069 , 0.00026979], [0.00016003, 0.0001663 , 0.00017978, 0.00019631, 0.00021001], [0.00015331, 0.0001624 , 0.00018005, 0.00020058, 0.00023177], [0.00016144, 0.00016735, 0.00017821, 0.00019164, 0.00020753], [0.00014646, 0.00015412, 0.00017348, 0.00020973, 0.00027814], [0.00015904, 0.00016682, 0.00017998, 0.00019338, 0.00020837], [0.00016243, 0.00016916, 0.00017917, 0.00019052, 0.00020079]], [0.00014370151360588277, 0.00017198634513687495, 0.0001905457505668284, 0.00020483338733222063, 0.00021662326462725586]], [[[225.25511848,   0.88442117,  -2.35096987,   3.98263041,
          1.15651916,  -3.1415906 ], [225.23833542,   0.88442876,  -2.35097332,   3.98266317,
          1.15656896,  -3.1415911 ], [225.24649018,   0.88442705,  -2.3509739 ,   3.98264393,
          1.15654263,  -3.14159311], [225.25707569,   0.88441983,  -2.35097237,   3.9826276 ,
          1.15650835,  -3.14159043], [225.2600216 ,   0.88441831,  -2.35097128,   3.98261922,
          1.15649794,  -3.14159091], [225.23251799,   0.88443306,  -2.35097448,   3.98267808,
          1.15659825,  -3.14159258], [225.23795158,   0.88442985,  -2.35097426,   3.98266036,
          1.15657809,  -3.1415932 ], [225.21882263,   0.88444104,  -2.35097524,   3.98270115,
          1.15663512,  -3.1415944 ], [225.24874711,   0.88442422,  -2.35097173,   3.98263803,
          1.15653813,  -3.14159147], [225.2445968 ,   0.8844269 ,  -2.35097101,   3.98265592,
          1.15655787,  -3.14159164]], [[7.28071171e-05, 1.20913989e-04, 2.03381312e-04, 2.88271907e-04,
        3.66234468e-04], [8.08895186e-05, 1.27740907e-04, 2.06866175e-04, 2.81495353e-04,
        3.31810261e-04], [7.86124236e-05, 1.25520927e-04, 2.02929747e-04, 2.75667997e-04,
        3.27920021e-04], [7.54478329e-05, 1.22237141e-04, 2.02197444e-04, 2.83313050e-04,
        3.56800068e-04], [7.43127428e-05, 1.20825384e-04, 2.01735245e-04, 2.85460544e-04,
        3.63237539e-04], [8.11182370e-05, 1.27321977e-04, 2.06330659e-04, 2.82271743e-04,
        3.27791664e-04], [8.50268068e-05, 1.29141581e-04, 2.02853635e-04, 2.74330979e-04,
        3.21582964e-04], [8.26159135e-05, 1.31531384e-04, 2.10558450e-04, 2.77723371e-04,
        2.99268306e-04], [7.94111754e-05, 1.25510977e-04, 2.02779170e-04, 2.79179638e-04,
        3.41876249e-04], [7.28367400e-05, 1.22831701e-04, 2.07887771e-04, 2.90188368e-04,
        3.52036461e-04]], [1.1236562259431423e-07, 2.948251663966858e-05, 0.0004899036078155068, 0.00041268977094565046, 5.8154009754610896e-05]]]

    plotICRobustness(biglist,names,noisefactor=1e-1)

    #1e-1 precision, 10 orbits start from true  (wrong)
    # biglist = [[[[225.22961113,   0.88443533,  -2.35096746,   3.98270119,
    #           1.15663236,  -3.14159368], [225.22553388,   0.88443762,  -2.3509673 ,   3.98271196,
    #           1.15665824,  -3.14159415], [225.25587379,   0.88442178,  -2.35096394,   3.98265857,
    #           1.15655176,  -3.14159105], [225.22570724,   0.88443795,  -2.35096583,   3.98271508,
    #           1.15663812,  -3.14159324], [225.23985914,   0.88443027,  -2.35096529,   3.98268574,
    #           1.15659778,  -3.14159264], [225.24954589,   0.88442481,  -2.35096768,   3.98267228,
    #           1.15657004,  -3.14159126], [225.2623608 ,   0.88441784,  -2.35096521,   3.98263897,
    #           1.15652577,  -3.14159042], [225.23791014,   0.88443125,  -2.35096532,   3.98269115,
    #           1.15661429,  -3.14159303], [225.24111721,   0.88442941,  -2.35096511,   3.98268047,
    #           1.15659818,  -3.14159223], [225.24395346,   0.8844275 ,  -2.35096705,   3.9826723 ,
    #           1.1565769 ,  -3.14159176]], [[7.42077193e-05, 1.15230063e-04, 1.81935602e-04, 2.48692749e-04,
    #         3.04253631e-04], [7.69399058e-05, 1.14938545e-04, 1.80146380e-04, 2.49449209e-04,
    #         3.04333900e-04], [6.39387616e-05, 1.02915504e-04, 1.74661355e-04, 2.64083709e-04,
    #         3.68525006e-04], [6.59494501e-05, 1.10964374e-04, 1.86181586e-04, 2.63038564e-04,
    #         3.19515106e-04], [6.77974455e-05, 1.07529542e-04, 1.78546733e-04, 2.60611053e-04,
    #         3.41018327e-04], [7.09418236e-05, 1.06854964e-04, 1.72299703e-04, 2.54217150e-04,
    #         3.48733275e-04], [6.85599576e-05, 1.05572844e-04, 1.71972966e-04, 2.54097045e-04,
    #         3.56658097e-04], [6.95991323e-05, 1.09503891e-04, 1.79348005e-04, 2.57421151e-04,
    #         3.33774788e-04], [6.92093940e-05, 1.09987253e-04, 1.80052063e-04, 2.57038010e-04,
    #         3.28751604e-04], [7.47671126e-05, 1.10619360e-04, 1.73896746e-04, 2.48412861e-04,
    #         3.24865479e-04]], [3.609632592453552e-05, 0.0001181406130613886, 0.0002112682471865419, 0.000283560136574258, 0.0003221765543589564]],
    #         [[[225.25129822,   0.88442492,  -2.35096143,   3.98268034,
    #             1.15658712,  -3.14159183], [225.25010924,   0.88442522,  -2.35096423,   3.98268095,
    #             1.15658221,  -3.14159158], [225.24385729,   0.88442861,  -2.35096315,   3.9826911 ,
    #             1.15661161,  -3.14159274], [225.25070693,   0.88442486,  -2.35096253,   3.98267384,
    #             1.15658484,  -3.14159216], [225.25815068,   0.88442066,  -2.35096363,   3.98265901,
    #             1.15655657,  -3.14159158], [225.24187403,   0.88442969,  -2.35096276,   3.98269322,
    #             1.15660933,  -3.14159219], [225.24181072,   0.88442939,  -2.35096354,   3.98268528,
    #             1.15660606,  -3.14159279], [225.24912607,   0.88442571,  -2.3509643 ,   3.98268244,
    #             1.15658219,  -3.14159129], [225.24648881,   0.88442741,  -2.35096396,   3.98269523,
    #             1.15660822,  -3.14159215], [225.22311939,   0.88443958,  -2.35096168,   3.98271644,
    #             1.15666414,  -3.14159525]], 
    #         [[0.00015111, 0.00016001, 0.00017865, 0.00020699, 0.00024838], [0.00015478, 0.00016274, 0.00017846, 0.00019979, 0.00023175], [0.00015411, 0.00016537, 0.000184  , 0.00020209, 0.000216  ], [0.0001563 , 0.00016328, 0.00017737, 0.00019904, 0.00023107], [0.00015626, 0.00016304, 0.00017647, 0.00019678, 0.00023056], [0.00015448, 0.00016476, 0.00018285, 0.00020165, 0.00021845], [0.00016048, 0.00016719, 0.0001791 , 0.00019382, 0.0002072 ], [0.00015492, 0.00016321, 0.00017805, 0.00019847, 0.00023264], [0.00015249, 0.00016243, 0.00018066, 0.0002047 , 0.0002365 ], [0.00016078, 0.00017036, 0.00018665, 0.00019715, 0.00018271]],
    #         [0.00014370151360588277, 0.00017198634513687495, 0.0001905457505668284, 0.00020483338733222063, 0.00021662326462725586]],
    #     [[[225.25255358,   0.8844223 ,  -2.3509739 ,   3.98264103,
    #               1.15651972,  -3.14159021], [225.26553168,   0.88441507,  -2.35097035,   3.98261128,
    #               1.15648553,  -3.14158965], [225.25372401,   0.88442135,  -2.35097342,   3.98262823,
    #               1.1565132 ,  -3.14159084], [225.23810101,   0.88442994,  -2.35097094,   3.98266213,
    #               1.15657139,  -3.14159186], [225.24128891,   0.88442791,  -2.35097455,   3.98265276,
    #               1.15656193,  -3.14159223], [225.24540579,   0.88442625,  -2.35097294,   3.98265068,
    #               1.15655251,  -3.14159196], [225.26954502,   0.88441301,  -2.35097078,   3.98259769,
    #               1.15646747,  -3.14158996], [225.23370898,   0.88443192,  -2.35097306,   3.98266081,
    #               1.15657464,  -3.14159262], [225.22587156,   0.8844366 ,  -2.35097258,   3.98268436,
    #               1.15661599,  -3.14159366], [225.23887323,   0.88442966,  -2.35097223,   3.98265909,
    #               1.15656602,  -3.14159199]], [[7.74224071e-05, 1.21490770e-04, 1.99287556e-04, 2.82324927e-04,
    #             3.60845616e-04], [7.33781138e-05, 1.20417012e-04, 2.00868652e-04, 2.85478017e-04,
    #             3.71452900e-04], [8.21068382e-05, 1.24589669e-04, 1.97497761e-04, 2.74336997e-04,
    #             3.47663145e-04], [7.54746535e-05, 1.26583337e-04, 2.10059303e-04, 2.85606346e-04,
    #             3.35493078e-04], [8.53912920e-05, 1.29578698e-04, 2.02498583e-04, 2.70793551e-04,
    #             3.20052980e-04], [7.91134749e-05, 1.24396043e-04, 2.00935592e-04, 2.80773616e-04,
    #             3.52444859e-04], [7.66762489e-05, 1.20112167e-04, 1.95918895e-04, 2.80459722e-04,
    #             3.72493992e-04], [8.53846275e-05, 1.31655992e-04, 2.05049573e-04, 2.69974231e-04,
    #             3.09800977e-04], [8.01140331e-05, 1.31041164e-04, 2.12349968e-04, 2.80336198e-04,
    #             3.10056798e-04], [7.79620617e-05, 1.26343131e-04, 2.07508359e-04, 2.83397696e-04,
    #             3.30844676e-04]], [1.1236562259431423e-07, 2.948251663966858e-05, 0.0004899036078155068, 0.00041268977094565046, 5.8154009754610896e-05]]]

    # start from true distribution, IG2, 10 orbits
    biglist = [[[[225.22960789,   0.88443749,  -2.35096551,   3.98271607,
             1.1566508 ,  -3.14159542], [225.22553354,   0.88443838,  -2.3509641 ,   3.98272508,
             1.15667746,  -3.1415945 ], [225.25586902,   0.8844227 ,  -2.35096027,   3.98267366,
             1.1565742 ,  -3.14159148], [225.22570726,   0.88443887,  -2.35096272,   3.98272842,
             1.15665729,  -3.14159375], [225.23985921,   0.88443105,  -2.35096205,   3.98269899,
             1.15661726,  -3.14159299], [225.24954368,   0.88442576,  -2.35096433,   3.98268665,
             1.15659103,  -3.14159176], [225.2623557 ,   0.88441872,  -2.35096166,   3.98265359,
             1.15654748,  -3.14159083], [225.23790797,   0.88443215,  -2.35096187,   3.9827056 ,
             1.15663549,  -3.14159348], [225.24111731,   0.88443027,  -2.35096159,   3.98269495,
             1.15661958,  -3.14159263], [225.24395224,   0.88442829,  -2.35096378,   3.98268575,
             1.1565967 ,  -3.14159213]], 
         [[3.92839407e-05, 1.24426543e-04, 2.15664893e-04, 2.73338638e-04,
           2.82279553e-04], [4.39769795e-05, 1.25451193e-04, 2.14689984e-04, 2.75096541e-04,
           2.83180243e-04], [2.99885950e-05, 1.12277846e-04, 2.08476804e-04, 2.91434412e-04,
           3.56472245e-04], [3.26666372e-05, 1.21342727e-04, 2.20745241e-04, 2.88682987e-04,
           2.98253972e-04], [3.50695830e-05, 1.17829278e-04, 2.12464200e-04, 2.86020194e-04,
           3.21868566e-04], [3.74822881e-05, 1.16534980e-04, 2.05917868e-04, 2.80521131e-04,
           3.33674136e-04], [3.50640875e-05, 1.15238933e-04, 2.05486145e-04, 2.80333788e-04,
           3.43973942e-04], [3.54877348e-05, 1.19421115e-04, 2.14247850e-04, 2.84569069e-04,
           3.16518557e-04], [3.50101682e-05, 1.19960203e-04, 2.15124159e-04, 2.84461458e-04,
           3.11545716e-04], [4.19097554e-05, 1.20886723e-04, 2.07775550e-04, 2.73931784e-04,
           3.06472701e-04]], 
         [3.609632592453552e-05, 0.0001181406130613886, 0.0002112682471865419, 0.000283560136574258, 0.0003221765543589564]],
        [[[225.25129871,   0.88442512,  -2.35096054,   3.98268399,
                 1.15659254,  -3.14159191], [225.25010962,   0.88442542,  -2.35096331,   3.98268477,
                 1.15658791,  -3.14159166], [225.24385804,   0.88442883,  -2.35096223,   3.982695  ,
                 1.15661729,  -3.14159284], [225.25070559,   0.88442501,  -2.35096151,   3.9826774 ,
                 1.15659032,  -3.14159217], [225.25814607,   0.88442091,  -2.35096269,   3.98266301,
                 1.15656248,  -3.1415917 ], [225.24187444,   0.8844299 ,  -2.35096187,   3.98269696,
                 1.15661476,  -3.14159228], [225.24181241,   0.88442958,  -2.35096278,   3.98268856,
                 1.15661078,  -3.14159288], [225.24912313,   0.88442597,  -2.35096329,   3.98268664,
                 1.15658844,  -3.1415914 ], [225.2464856 ,   0.88442764,  -2.35096304,   3.98269908,
                 1.15661388,  -3.14159225], [225.22312002,   0.88444043,  -2.35096118,   3.98272067,
                 1.1566684 ,  -3.141596  ]], 
         [[0.00014011, 0.00016704, 0.0001867 , 0.00021052, 0.00024695], [0.00014359, 0.00016976, 0.0001867 , 0.00020339, 0.00023084], [0.00014207, 0.00017265, 0.00019369, 0.00020675, 0.00021178], [0.00014515, 0.00017047, 0.00018584, 0.00020283, 0.00022893], [0.00014483, 0.00017003, 0.00018482, 0.00020059, 0.00022991], [0.00014274, 0.00017202, 0.00019222, 0.00020598, 0.0002143 ], [0.00014936, 0.00017458, 0.00018784, 0.00019733, 0.00020254], [0.00014316, 0.00017021, 0.0001867 , 0.00020254, 0.00023248], [0.00014082, 0.00016959, 0.00018964, 0.00020889, 0.00023381], [0.0001483 , 0.00017708, 0.00019577, 0.00020104, 0.00017578]], 
             [0.00014370151360588277, 0.00017198634513687495, 0.0001905457505668284, 0.00020483338733222063, 0.00021662326462725586]],
        [[[225.25255407,   0.88442505,  -2.35096334,   3.98268342,
                 1.15657584,  -3.14159174], [225.26552973,   0.88441736,  -2.35095962,   3.98265258,
                 1.15654146,  -3.14159076], [225.25372274,   0.88442334,  -2.3509623 ,   3.98266962,
                 1.15656975,  -3.14159164], [225.23810777,   0.88443263,  -2.35096019,   3.98270213,
                 1.1566231 ,  -3.14159347], [225.24128953,   0.88442987,  -2.35096348,   3.98269385,
                 1.15661788,  -3.14159301], [225.24540736,   0.88442824,  -2.35096196,   3.98269156,
                 1.1566082 ,  -3.14159277], [225.26954128,   0.88441502,  -2.35095966,   3.9826391 ,
                 1.15652424,  -3.14159078], [225.23371312,   0.88443378,  -2.35096255,   3.98269973,
                 1.1566272 ,  -3.1415934 ], [225.22587368,   0.88443846,  -2.35096204,   3.98272342,
                 1.15666884,  -3.14159441], [225.23887736,   0.88443153,  -2.35096149,   3.98269881,
                 1.15661992,  -3.14159273]], [[1.59783112e-06, 1.68821174e-05, 4.81531725e-04, 4.17046788e-04,
               9.13308079e-05], [2.03152918e-06, 1.20551596e-05, 4.81528107e-04, 4.20392452e-04,
               1.04958809e-04], [4.00658292e-06, 2.52391783e-05, 4.81712154e-04, 4.08024091e-04,
               7.57727603e-05], [1.40654978e-06, 2.35814634e-05, 4.91610764e-04, 4.16358621e-04,
               5.71357178e-05], [6.34829062e-06, 3.11233122e-05, 4.88317454e-04, 4.04903995e-04,
               4.36282670e-05], [2.47983817e-06, 2.39004164e-05, 4.84420681e-04, 4.14183193e-04,
               7.97156203e-05], [2.86641197e-06, 1.53542023e-05, 4.77633654e-04, 4.15050909e-04,
               1.06063673e-04], [1.18457863e-05, 2.99019491e-05, 4.87604222e-04, 4.01494235e-04,
               3.19415737e-05], [7.27616800e-06, 2.78962664e-05, 4.94979799e-04, 4.12814297e-04,
               3.25139594e-05], [2.75195916e-06, 2.52822965e-05, 4.91126359e-04, 4.16401112e-04,
               5.36200352e-05]], [1.1236562259431423e-07, 2.948251663966858e-05, 0.0004899036078155068, 0.00041268977094565046, 5.8154009754610896e-05]]                          ]

# [[[[225.25129871,   0.88442512,  -2.35096054,   3.98268399,
#          1.15659254,  -3.14159191], [225.25010962,   0.88442542,  -2.35096331,   3.98268477,
#          1.15658791,  -3.14159166], [225.24385804,   0.88442883,  -2.35096223,   3.982695  ,
#          1.15661729,  -3.14159284], [225.25070559,   0.88442501,  -2.35096151,   3.9826774 ,
#          1.15659032,  -3.14159217], [225.25814607,   0.88442091,  -2.35096269,   3.98266301,
#          1.15656248,  -3.1415917 ], [225.24187444,   0.8844299 ,  -2.35096187,   3.98269696,
#          1.15661476,  -3.14159228], [225.24181241,   0.88442958,  -2.35096278,   3.98268856,
#          1.15661078,  -3.14159288], [225.24912313,   0.88442597,  -2.35096329,   3.98268664,
#          1.15658844,  -3.1415914 ], [225.2464856 ,   0.88442764,  -2.35096304,   3.98269908,
#          1.15661388,  -3.14159225], [225.22312002,   0.88444043,  -2.35096118,   3.98272067,
#          1.1566684 ,  -3.141596  ]], 
#  [[0.00014011, 0.00016704, 0.0001867 , 0.00021052, 0.00024695], [0.00014359, 0.00016976, 0.0001867 , 0.00020339, 0.00023084], [0.00014207, 0.00017265, 0.00019369, 0.00020675, 0.00021178], [0.00014515, 0.00017047, 0.00018584, 0.00020283, 0.00022893], [0.00014483, 0.00017003, 0.00018482, 0.00020059, 0.00022991], [0.00014274, 0.00017202, 0.00019222, 0.00020598, 0.0002143 ], [0.00014936, 0.00017458, 0.00018784, 0.00019733, 0.00020254], [0.00014316, 0.00017021, 0.0001867 , 0.00020254, 0.00023248], [0.00014082, 0.00016959, 0.00018964, 0.00020889, 0.00023381], [0.0001483 , 0.00017708, 0.00019577, 0.00020104, 0.00017578]], 
#      [0.00014370151360588277, 0.00017198634513687495, 0.0001905457505668284, 0.00020483338733222063, 0.00021662326462725586]]]

# [[[[225.25255407,   0.88442505,  -2.35096334,   3.98268342,
#          1.15657584,  -3.14159174], [225.26552973,   0.88441736,  -2.35095962,   3.98265258,
#          1.15654146,  -3.14159076], [225.25372274,   0.88442334,  -2.3509623 ,   3.98266962,
#          1.15656975,  -3.14159164], [225.23810777,   0.88443263,  -2.35096019,   3.98270213,
#          1.1566231 ,  -3.14159347], [225.24128953,   0.88442987,  -2.35096348,   3.98269385,
#          1.15661788,  -3.14159301], [225.24540736,   0.88442824,  -2.35096196,   3.98269156,
#          1.1566082 ,  -3.14159277], [225.26954128,   0.88441502,  -2.35095966,   3.9826391 ,
#          1.15652424,  -3.14159078], [225.23371312,   0.88443378,  -2.35096255,   3.98269973,
#          1.1566272 ,  -3.1415934 ], [225.22587368,   0.88443846,  -2.35096204,   3.98272342,
#          1.15666884,  -3.14159441], [225.23887736,   0.88443153,  -2.35096149,   3.98269881,
#          1.15661992,  -3.14159273]], [[1.59783112e-06, 1.68821174e-05, 4.81531725e-04, 4.17046788e-04,
#        9.13308079e-05], [2.03152918e-06, 1.20551596e-05, 4.81528107e-04, 4.20392452e-04,
#        1.04958809e-04], [4.00658292e-06, 2.52391783e-05, 4.81712154e-04, 4.08024091e-04,
#        7.57727603e-05], [1.40654978e-06, 2.35814634e-05, 4.91610764e-04, 4.16358621e-04,
#        5.71357178e-05], [6.34829062e-06, 3.11233122e-05, 4.88317454e-04, 4.04903995e-04,
#        4.36282670e-05], [2.47983817e-06, 2.39004164e-05, 4.84420681e-04, 4.14183193e-04,
#        7.97156203e-05], [2.86641197e-06, 1.53542023e-05, 4.77633654e-04, 4.15050909e-04,
#        1.06063673e-04], [1.18457863e-05, 2.99019491e-05, 4.87604222e-04, 4.01494235e-04,
#        3.19415737e-05], [7.27616800e-06, 2.78962664e-05, 4.94979799e-04, 4.12814297e-04,
#        3.25139594e-05], [2.75195916e-06, 2.52822965e-05, 4.91126359e-04, 4.16401112e-04,
#        5.36200352e-05]], [1.1236562259431423e-07, 2.948251663966858e-05, 0.0004899036078155068, 0.00041268977094565046, 5.8154009754610896e-05]]]


    plotICRobustness(biglist,names,noisefactor=1e-1)

    
    #30K obs (10 orbits + 10x obs), IG3
    # biglist = [[[[225.36882758,   0.88436328,  -2.35095197,   3.98247956,
    #          1.15621716,  -3.14158189], [225.49726433,   0.8842944 ,  -2.35094834,   3.982233  ,
    #          1.15577964,  -3.1415692 ], [225.11310929,   0.88449871,  -2.35097304,   3.98294387,
    #          1.15703655,  -3.1416035 ], [225.14892354,   0.88447992,  -2.35096663,   3.98287148,
    #          1.15691552,  -3.1416019 ], [225.14496612,   0.88448089,  -2.35097624,   3.98288223,
    #          1.15691878,  -3.14159993], [225.22976479,   0.88443593,  -2.3509667 ,   3.98270576,
    #          1.15664543,  -3.14159371], [224.97556669,   0.88457184,  -2.35098566,   3.98320131,
    #          1.15748922,  -3.14161437], [225.23150585,   0.88443494,  -2.35096145,   3.98268789,
    #          1.15662123,  -3.14159389], [225.35972771,   0.88436719,  -2.35096111,   3.98249986,
    #          1.15624712,  -3.1415807 ], [225.28077846,   0.88440888,  -2.35096483,   3.98262479,
    #          1.15648929,  -3.14158959]],
    #      [[1.72217632e-05, 5.92017106e-05, 1.51389025e-04, 3.14163683e-04,
    #        6.35254271e-04], [1.19317434e-05, 1.72826350e-05, 8.86200919e-05, 3.08688179e-04,
    #        8.86375217e-04], [7.46313108e-05, 1.39249660e-04, 2.36035820e-04, 2.73059816e-04,
    #        1.23762034e-04], [6.35438116e-05, 1.29345348e-04, 2.21370680e-04, 2.77771929e-04,
    #        2.14731048e-04], [7.85399911e-05, 1.35341443e-04, 2.18286397e-04, 2.55228951e-04,
    #        1.71126485e-04], [6.48227543e-05, 1.11129251e-04, 1.89718910e-04, 2.66223158e-04,
    #        3.19352939e-04], [0.00013112, 0.00017957, 0.00023142, 0.00018757, 0.        ], [6.53807259e-05, 1.09746953e-04, 1.86269502e-04, 2.64059976e-04,
    #        3.27982366e-04], [3.56545791e-05, 6.74335865e-05, 1.45351587e-04, 2.87336762e-04,
    #        5.89551389e-04], [5.45389626e-05, 9.62911262e-05, 1.71870505e-04, 2.64275553e-04,
    #        4.30377459e-04]],
    #      [3.609632592453552e-05, 0.0001181406130613886, 0.0002112682471865419, 0.000283560136574258, 0.0003221765543589564]],
    #       [[[225.07722326,   0.88451735,  -2.35097417,   3.98298419,
    #            1.15713124,  -3.141608  ], [225.36775974,   0.88436236,  -2.35095358,   3.98245214,
    #            1.15619572,  -3.14158334], [225.08370117,   0.88451378,  -2.35097513,   3.98298542,
    #            1.15714913,  -3.14160701], [225.54761272,   0.88426779,  -2.35093978,   3.98214534,
    #            1.15564086,  -3.14156501], [225.20379847,   0.88444846,  -2.35096737,   3.98272841,
    #            1.15671505,  -3.14159762], [225.38934891,   0.88435207,  -2.3509534 ,   3.9824451 ,
    #            1.15616771,  -3.14157971], [225.22735834,   0.88443822,  -2.35096736,   3.98269934,
    #            1.1566461 ,  -3.14159683], [225.5777353 ,   0.88425168,  -2.35093915,   3.98209388,
    #            1.15555471,  -3.14156208], [225.3375449 ,   0.88437825,  -2.35095944,   3.98250644,
    #            1.15627767,  -3.14158302], [225.18660697,   0.88445951,  -2.35096462,   3.98279976,
    #            1.15681185,  -3.14159757]],
    #        [[0.00020325, 0.00021312, 0.00020695, 0.00013974, 0.        ], [0.00013263, 0.00012915, 0.00014432, 0.00021312, 0.00043151], [2.05168049e-04, 2.15489208e-04, 2.07206788e-04, 1.32636982e-04,
    #          4.65827358e-07], [7.42981388e-05, 7.08216160e-05, 1.06592606e-04, 2.72204750e-04,
    #          7.73112199e-04], [1.82244493e-04, 1.88839236e-04, 1.89409090e-04, 1.66570780e-04,
    #          7.84476962e-05], [0.00011992, 0.00011319, 0.00013446, 0.00023535, 0.00053131], [0.00017295, 0.00017709, 0.00017908, 0.00016924, 0.00014061], [6.52255156e-05, 6.19432458e-05, 1.07888305e-04, 2.88913707e-04,
    #          7.92742934e-04], [0.00014882, 0.00013722, 0.00013818, 0.00019305, 0.00039972], [0.00015623, 0.0001832 , 0.00021234, 0.00020282, 0.00011584]], [0.00014370151360588277, 0.00017198634513687495, 0.0001905457505668284, 0.00020483338733222063, 0.00021662326462725586]],
    #        [[[225.3613116 ,   0.8843646 ,  -2.3509596 ,   3.98245992,
    #                 1.15620237,  -3.14157936], [225.19362055,   0.8844543 ,  -2.35097126,   3.98274732,
    #                 1.15674545,  -3.14159807], [225.27507788,   0.88441128,  -2.35096784,   3.98261649,
    #                 1.15648342,  -3.14158919], [225.3809316 ,   0.88435646,  -2.35095583,   3.98242645,
    #                 1.15613586,  -3.14157859], [225.41030576,   0.88433945,  -2.35096267,   3.98237431,
    #                 1.15604003,  -3.1415758 ], [225.135509  ,   0.88448331,  -2.3509868 ,   3.98284435,
    #                 1.15692287,  -3.14160329], [225.18976786,   0.88445627,  -2.35097742,   3.98277611,
    #                 1.15671601,  -3.14159404], [224.99846544,   0.88456014,  -2.35098459,   3.9831401 ,
    #                 1.15739598,  -3.1416148 ], [225.29768342,   0.88439913,  -2.35096335,   3.98255435,
    #                 1.15640282,  -3.14158814], [225.25618088,   0.8844201 ,  -2.35096862,   3.98261806,
    #                 1.15652534,  -3.14159085]],
    #         [[3.24350009e-05, 8.05387312e-05, 1.86883292e-04, 3.40363388e-04,
    #               5.90681985e-04], [7.46315315e-05, 1.39201285e-04, 2.38843619e-04, 2.99308436e-04,
    #               2.35954003e-04], [5.68679262e-05, 1.11879361e-04, 2.04936590e-04, 3.06573567e-04,
    #               4.26145387e-04], [2.39775823e-05, 6.80812107e-05, 1.69332858e-04, 3.42266775e-04,
    #               6.82887218e-04], [3.71976051e-05, 6.81387050e-05, 1.48373381e-04, 3.07297260e-04,
    #               7.20035880e-04], [1.09843480e-04, 1.75397503e-04, 2.48596287e-04, 2.42912364e-04,
    #               5.68458059e-05], [7.17445189e-05, 1.38074877e-04, 2.33284971e-04, 2.91313296e-04,
    #               2.57354664e-04], [1.07197039e-04, 1.91095870e-04, 2.84872467e-04, 2.53776685e-04,
    #               1.53889804e-06], [4.52357854e-05, 1.11948528e-04, 2.14642848e-04, 3.10372495e-04,
    #               4.33287653e-04], [7.90167308e-05, 1.27704051e-04, 2.04150816e-04, 2.73973911e-04,
    #               3.40420589e-04]],
    #     [1.1236562259431423e-07, 2.948251663966858e-05, 0.0004899036078155068, 0.00041268977094565046, 5.8154009754610896e-05]]]
    #
    #
    # plotICRobustness(biglist,names,noisefactor=1)
                                    
                
    
    
    
    
    
    
    
    
    
    