# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:25:26 2022

@author: Thibault
"""

import numpy as np
from matplotlib.pylab import plt
import orbitModule


timegrid = orbitModule.getObservationTimes()

#Alternative timegrid:
# comparedData = np.loadtxt('Datasets/1PN.txt')
# timegrid = comparedData[:,0]

M_0, D_0, T_0 = orbitModule.getBaseUnitConversions()
xlim = orbitModule.get_xlim()
rDM = np.linspace(0,xlim,1000)


def enclosedMassPlum(a,rho0):
    r0 = 2474.01
    return (4 * a**3 * np.pi * r0**3 * rho0) / ( 3 * (a**2 + r0**2)**(3/2))

def enclosedMassCusp(a,rho0):
    r0 = 2474.01
    return (4 * a**3 * np.pi * (a/r0)**(-7/4) * rho0) / (3 - (7/4))


def plotMasconsMass(N=20,k=0.1,name='Plummer'):
    #Possible names: Plummer, BahcallWolf, Sinusoidal, Uniform, ConstantDensity
    
    
    rho0plum = 1.69*10**(-10) * (D_0**3) / M_0
    rho0cusp = 2.24*10**(-11) * (D_0**3) / M_0
    
    # #Plummer model:
    # rhoPlum = rho0plum *( 1. + ((rDM**2) / (r0**2)))**(-5/2)
    # #BahcallWolfCusp model:
    # rhoCusp = rho0cusp * (rDM / r0)**(-7/4)
    
    rp = 119.52867
    ra = 1948.96214
    
        
    getTrueDM = getattr(orbitModule,'get_'+name+'_DM')
    mis, ris = getTrueDM(N)
    
    
    # Mascon model (mi, ri), sigmoid approximation of step function
    listOfSigs = [0.5 + 0.5 * np.tanh( k * (rDM - ris[i])) for i in range(N)]
    
    listOfRis = [mis[i]* listOfSigs[i] for i in range(N)]
    
    suml = listOfSigs[0]
    for i in range(1,len(listOfSigs)):
        suml = suml + listOfSigs[i]
        
    sumRis = listOfRis[0]
    for i in range(1,len(listOfRis)):
        sumRis = sumRis + listOfRis[i]
    
    #Plot enclosed mass
    plt.figure()
    plt.xlabel('Distance from MBH [AU]')
    if name == 'Plummer':
        plt.plot(rDM,enclosedMassPlum(rDM,rho0plum),label='Plummer model')
    elif name == 'BahcallWolf':
        plt.plot(rDM,np.append(0,enclosedMassCusp(rDM[1:],rho0cusp)),label='Bahcall-wolf model')
    
    plt.plot(rDM,sumRis,label='Mascon shell model',color='tab:orange')
    plt.ylabel('Enclosed mass [MBH masses]')
    plt.axvline(rp,linestyle='--',label='rp and ra',color='black')
    plt.axvline(ra,linestyle='--',color='black')
    # plt.scatter(ris,np.cumsum(mis),label='Mascon shells',color='orange')
    # plt.bar(ris,np.cumsum(mis),width=(xlim)/(N),alpha=0.2,align='edge',edgecolor='orange',color='orange')
    plt.legend()
    # plt.title('Enclosed mass')
    
    
    #Plot mass of mascons
    plt.figure()
    plt.scatter(ris,mis,label='Mascon shell masses',color='orange')
    plt.axvline(rp,linestyle='--',label='rp and ra',color='black')
    plt.axvline(ra,linestyle='--',color='black')
    plt.ylabel('Mass [MBH masses]')
    plt.xlabel('Distance from MBH [AU]')
    plt.ylim(0)
    plt.legend()
    # plt.title('Masses of DM shells')
    
    
    #Plot theoretical plum model in kg/m^3 and mpc
    # rDMmpc = rDM*0.004848
    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # ax1.plot(rDMmpc,rhoPlum*M_0/(D_0**3)*10**10,color='b')
    # # ax1.plot(rDMmpc,rhoCusp*M_0/(D_0**3)*10**10,'--',color='b')
    # ax1.set_xlabel('r [mpc]')
    # ax1.set_ylabel('rho [10^(-10) kg/m³]',color='b')
    # ax1.set_ylim(0,4)
    # ax2.plot(rDMmpc,enclosedMass(rDM,rho0plum)*M_0/(M_sol*1e3),color='orange')
    # ax2.set_ylabel('enclosed mass [10³ solar masses]',color='orange')
    # plt.show()


def effectOfAmountOfMascons(N1=10,N2=100):
    IC = orbitModule.get_S2_IC()
    
    # misN1, risN1 = orbitModule.get_Plummer_DM(N1)
    # misN2, risN2 = orbitModule.get_Plummer_DM(N2)
    
    misN1, risN1 = orbitModule.get_Sinusoidal_DM(N1)
    misN2, risN2 = orbitModule.get_Sinusoidal_DM(N2)
    
    rxDMN1,ryDMN1,rzDMN1 , vxDMN1,vyDMN1,vzDMN1  = \
        orbitModule.simulateOrbitsCartesian(True, IC, misN1, risN1,timegrid)
    
    rxDMN2,ryDMN2,rzDMN2,vxDMN2,vyDMN2,vzDMN2 = \
        orbitModule.simulateOrbitsCartesian(True, IC, misN2, risN2,timegrid)
    
    
    xdifs = 1e6*(orbitModule.AU_to_arcseconds(rxDMN2)-orbitModule.AU_to_arcseconds(rxDMN1))
    ydifs = 1e6*(orbitModule.AU_to_arcseconds(ryDMN2)-orbitModule.AU_to_arcseconds(ryDMN1))
    vzdifs = (vzDMN2-vzDMN1)/1000
    
    print('Max X difference:',round(max(abs(xdifs)),2),'[µas]')
    print('Max Y difference:',round(max(abs(ydifs)),2),'[µas]')
    print('Max VZ difference:',round(max(abs(vzdifs)),2),'[km/s]')
    
    
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('Difference [µas]')
    plt.title('X ({} mascons) - X ({} mascons)'.format(N2,N1))
    plt.scatter(timegrid,xdifs,s=10)
    plt.plot(timegrid,len(timegrid)*[50],'--',label='Precision',color='red')
    plt.plot(timegrid,len(timegrid)*[-50],'--',color='red')
    plt.legend()
    
    
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('Difference [µas]')
    plt.title('Y ({} mascons) - Y ({} mascons)'.format(N2,N1))
    plt.scatter(timegrid,ydifs,s=10)
    plt.plot(timegrid,len(timegrid)*[50],'--',label='Precision',color='red')
    plt.plot(timegrid,len(timegrid)*[-50],'--',color='red')
    plt.legend()
    
    
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('Difference [km/s]')
    plt.title('VZ ({} mascons) - VZ ({} mascons)'.format(N2,N1))
    plt.scatter(timegrid,vzdifs,s=10)
    plt.plot(timegrid,len(timegrid)*[10],'--',label='Precision',color='red')
    plt.plot(timegrid,len(timegrid)*[-10],'--',color='red')
    plt.legend()
    
    

def effectOfIndividualMascons():
    mis0 = [0] #-> 0 dark matter, has no effect
    ris0 = np.linspace(0,1,1)
    
    IC = orbitModule.get_S2_IC()
    
    rxPN,ryPN,rzPN, vxPN,vyPN,vzPN = orbitModule.simulateOrbitsCartesian(True, IC, mis0, ris0,timegrid)
    
    mis,ris = orbitModule.get_Plummer_DM(N=20)
    
    #Plot individual differences of mascons:
    mis1 = mis.copy()
    mis1[4] = 0
    mis1[7] = 0
    
    rxDM1,ryDM1,rzDM1,vxDM1,vyDM1,vzDM1 = orbitModule.simulateOrbitsCartesian(True, IC, mis1, ris,timegrid)
    
    mis4 = mis.copy()
    mis4[1] = 0
    mis4[7] = 0
    
    rxDM4,ryDM4,rzDM4 ,vxDM4,vyDM4,vzDM4 = orbitModule.simulateOrbitsCartesian(True, IC, mis4, ris,timegrid)
    
    mis7 = mis.copy()
    mis7[1] = 0
    mis7[4] = 0
    
    rxDM7,ryDM7,rzDM7, vxDM7,vyDM7,vzDM7 = orbitModule.simulateOrbitsCartesian(True, IC, mis7, ris,timegrid)
    
    
    
    # #Plot difference of DM in function of time:
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('X PN - X DM  [µas]')
    plt.scatter(timegrid,orbitModule.AU_to_arcseconds(rxPN-rxDM1)*1e6,s=10,label='mascon 1')
    plt.scatter(timegrid,orbitModule.AU_to_arcseconds(rxPN-rxDM4)*1e6,s=10,label='mascon 4')
    plt.scatter(timegrid,orbitModule.AU_to_arcseconds(rxPN-rxDM7)*1e6,s=10,label='mascon 7')
    plt.plot(timegrid,len(timegrid)*[50],'--',label='Precision',color='red')
    plt.plot(timegrid,len(timegrid)*[-50],'--',color='red')
    plt.legend()
    
    
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('Y PN - Y DM  [µas]')
    plt.scatter(timegrid,orbitModule.AU_to_arcseconds(ryPN-ryDM1)*1e6,s=10,label='mascon 1')
    plt.scatter(timegrid,orbitModule.AU_to_arcseconds(ryPN-ryDM4)*1e6,s=10,label='mascon 4')
    plt.scatter(timegrid,orbitModule.AU_to_arcseconds(ryPN-ryDM7)*1e6,s=10,label='mascon 7')
    plt.plot(timegrid,len(timegrid)*[50],'--',label='Precision',color='red')
    plt.plot(timegrid,len(timegrid)*[-50],'--',color='red')
    plt.legend()
    
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('Z PN - Z DM  [µas]')
    plt.scatter(timegrid,orbitModule.AU_to_arcseconds(rzPN-rzDM1)*1e6,s=10,label='mascon 1')
    plt.scatter(timegrid,orbitModule.AU_to_arcseconds(rzPN-rzDM4)*1e6,s=10,label='mascon 4')
    plt.scatter(timegrid,orbitModule.AU_to_arcseconds(rzPN-rzDM7)*1e6,s=10,label='mascon 7')
    plt.plot(timegrid,len(timegrid)*[50],'--',label='Precision',color='red')
    plt.plot(timegrid,len(timegrid)*[-50],'--',color='red')
    plt.legend()
    
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('VX PN - VX DM [km/s]')
    plt.scatter(timegrid,(vxPN-vxDM1)/1000,s=10,label='mascon 1')
    plt.scatter(timegrid,(vxPN-vxDM4)/1000,s=10,label='mascon 4')
    plt.scatter(timegrid,(vxPN-vxDM7)/1000,s=10,label='mascon 7')
    plt.plot(timegrid,len(timegrid)*[10],'--',label='Precision',color='red')
    plt.plot(timegrid,len(timegrid)*[-10],'--',color='red')
    plt.legend()
    
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('VY PN - VY DM  [km/s]')
    plt.scatter(timegrid,(vyPN-vyDM1)/1000,s=10,label='mascon 1')
    plt.scatter(timegrid,(vyPN-vyDM4)/1000,s=10,label='mascon 4')
    plt.scatter(timegrid,(vyPN-vyDM7)/1000,s=10,label='mascon 7')
    plt.plot(timegrid,len(timegrid)*[10],'--',label='Precision',color='red')
    plt.plot(timegrid,len(timegrid)*[-10],'--',color='red')
    plt.legend()
    
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('VZ PN - VZ DM  [km/s]')
    plt.scatter(timegrid,(vzPN-vzDM1)/1000,s=10,label='mascon 1')
    plt.scatter(timegrid,(vzPN-vzDM4)/1000,s=10,label='mascon 4')
    plt.scatter(timegrid,(vzPN-vzDM7)/1000,s=10,label='mascon 7')
    plt.plot(timegrid,len(timegrid)*[10],'--',label='Precision',color='red')
    plt.plot(timegrid,len(timegrid)*[-10],'--',color='red')
    plt.legend()


def plotDifferenceWith1PN():
    mis0 = [0] #-> 0 dark matter, has no effect
    ris0 = np.linspace(0,1,1)
    
    IC = orbitModule.get_S2_IC()
    
    rxPN,ryPN,rzPN, vxPN,vyPN,vzPN = orbitModule.simulateOrbitsCartesian(True, IC, mis0, ris0,timegrid)
    
    mis,ris = orbitModule.get_Plummer_DM(100)
    
    rxDM,ryDM,rzDM,vxDM,vyDM,vzDM = orbitModule.simulateOrbitsCartesian(True, IC, mis, ris,timegrid)
    
    #Plot difference of DM:
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('X DM - X PN  [µas]')
    plt.scatter(timegrid,orbitModule.AU_to_arcseconds(rxDM-rxPN)*1e6,s=10,label='Difference')
    plt.plot(timegrid,len(timegrid)*[50],'--',label='Precision',color='red')
    plt.plot(timegrid,len(timegrid)*[-50],'--',color='red')
    plt.legend()
    
    #Plot difference of DM:
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('Y DM - Y PN  [µas]')
    plt.scatter(timegrid,orbitModule.AU_to_arcseconds(ryDM-ryPN)*1e6,s=10,label='Difference')
    plt.plot(timegrid,len(timegrid)*[50],'--',label='Precision',color='red')
    plt.plot(timegrid,len(timegrid)*[-50],'--',color='red')
    plt.legend()
    
    
    #Plot difference of DM:
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('VZ DM - VZ PN  [km/s]')
    plt.scatter(timegrid,(vzDM-vzPN)/1000,s=10,label='Difference')
    plt.plot(timegrid,len(timegrid)*[10],'--',label='Precision',color='red')
    plt.plot(timegrid,len(timegrid)*[-10],'--',color='red')
    plt.legend()
    
    
    
    #Plot effects of DM:
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('Value')
    plt.scatter(timegrid,orbitModule.AU_to_arcseconds(rxDM)*1e6,label='X DM , [µas]',s=10)
    plt.scatter(timegrid,orbitModule.AU_to_arcseconds(rxPN)*1e6,label='X PN , [µas]',s=10)
    plt.legend()
    
    
    #Plot effects of DM:
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('Value')
    plt.scatter(timegrid,orbitModule.AU_to_arcseconds(ryDM)*1e6,label='Y DM , [µas]',s=10)
    plt.scatter(timegrid,orbitModule.AU_to_arcseconds(ryPN)*1e6,label='Y PN , [µas]',s=10)
    plt.legend()
    
    
    #Plot effects of DM:
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('Value')
    plt.scatter(timegrid,-vzDM/1000,label='VZ DM , [km/s]',s=10)
    plt.scatter(timegrid,-vzPN/1000,label='VZ PN , [km/s]',s=10)
    plt.legend()

def plotDifferencePlumVsBahcall(N=100):
    IC = orbitModule.get_S2_IC()
    
    mis, ris = orbitModule.get_BahcallWolf_DM(N)
    
    rxBW,ryBW,rzBW, vxBW,vyBW,vzBW = orbitModule.simulateOrbitsCartesian(True, IC, mis, ris,timegrid)
    
    mis,ris = orbitModule.get_Plummer_DM(N)
    
    rxDM,ryDM,rzDM,vxDM,vyDM,vzDM = orbitModule.simulateOrbitsCartesian(True, IC, mis, ris,timegrid)
    
    #Plot difference of DM:
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('X DM - X PN  [µas]')
    plt.scatter(timegrid,orbitModule.AU_to_arcseconds(rxDM-rxBW)*1e6,s=10,label='Difference')
    plt.plot(timegrid,len(timegrid)*[50],'--',label='Precision',color='red')
    plt.plot(timegrid,len(timegrid)*[-50],'--',color='red')
    plt.legend()
    
    #Plot difference of DM:
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('Y DM - Y PN  [µas]')
    plt.scatter(timegrid,orbitModule.AU_to_arcseconds(ryDM-ryBW)*1e6,s=10,label='Difference')
    plt.plot(timegrid,len(timegrid)*[50],'--',label='Precision',color='red')
    plt.plot(timegrid,len(timegrid)*[-50],'--',color='red')
    plt.legend()
    
    
    #Plot difference of DM:
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('VZ DM - VZ PN  [km/s]')
    plt.scatter(timegrid,(vzDM-vzBW)/1000,s=10,label='Difference')
    plt.plot(timegrid,len(timegrid)*[10],'--',label='Precision',color='red')
    plt.plot(timegrid,len(timegrid)*[-10],'--',color='red')
    plt.legend()
    
    
    
    #Plot effects of DM:
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('Value')
    plt.scatter(timegrid,orbitModule.AU_to_arcseconds(rxDM)*1e6,label='X DM , [µas]',s=10)
    plt.scatter(timegrid,orbitModule.AU_to_arcseconds(rxBW)*1e6,label='X PN , [µas]',s=10)
    plt.legend()
    
    
    #Plot effects of DM:
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('Value')
    plt.scatter(timegrid,orbitModule.AU_to_arcseconds(ryDM)*1e6,label='Y DM , [µas]',s=10)
    plt.scatter(timegrid,orbitModule.AU_to_arcseconds(ryBW)*1e6,label='Y PN , [µas]',s=10)
    plt.legend()
    
    
    #Plot effects of DM:
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('Value')
    plt.scatter(timegrid,-vzDM/1000,label='VZ DM , [km/s]',s=10)
    plt.scatter(timegrid,-vzBW/1000,label='VZ PN , [km/s]',s=10)
    plt.legend()



"""
#Uncomment functions here to use them:
"""

if __name__ == "__main__":
    #Plot the mascon enclosed mass and individual masses for different profiles:
    plotMasconsMass(N=5,k=0.01)
    # plotMasconsMass(N=10,k=0.01,name='ConstantDensity')
    # plotMasconsMass(N=30,k=0.01,name='Sinusoidal')
    
    #Calculates the effect on the orbit of 3 different mascons
    # effectOfIndividualMascons()
    
    #Plot the difference in the orbit of post-newtonian vs with added dark matter
    # plotDifferenceWith1PN()
    
    #Discretization error:
    # Ns = [2,5,10,25,50,100]
    # for N in Ns:
    #     print('\nFor {} mascons:'.format(N))
    #     effectOfAmountOfMascons(N1=N,N2=10000)
            
    #Plot the difference of the Plummer distribution vs the Bahcallwolf:
    # plotDifferencePlumVsBahcall(N=1000)