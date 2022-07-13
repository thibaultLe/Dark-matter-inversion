# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:25:26 2022

@author: Thibault
"""

import numpy as np
from matplotlib.pylab import plt
from simulateOrbits import simulateOrbits

#1AU in meters
D_0 = 149597870700
#Solar mass
M_sol = 1.98841 * 10**30 
# mass m' = m/M_0 -> MBH = 1
M_0 = 4.2970174 * 10**6 * M_sol

def AU_to_arcseconds(dist):
    R = 2.5540153e+20
    return 2 * np.arctan(dist*D_0/(2*R)) * 206264.8

def enclosedMass(a,rho0):
    return (4 * a**3 * np.pi * r0**3 * rho0) / ( 3 * (a**2 + r0**2)**(3/2))


comparedData = np.loadtxt('Kepler.txt')
timegrid = comparedData[:,0]

#Dark matter:
#Amount of mascons:
N = 20



#Plot dark matter distribution
#AU limit
xlim = 10000
#Amount of points in linspace
n = 1000
#Bahcall-Wolf cusp model:
rDM = np.linspace(0,xlim,n)

rho0plum = 1.69*10**(-10) * (D_0**3) / M_0
# rho0cusp = 2.24*10**(-11) * (D_0**3) / M_0
r0 = 2474.01

#Plummer model:
rhoPlum = rho0plum *( 1. + ((rDM**2) / (r0**2)))**(-5/2)
#Cusp model:
# rhoCusp = rho0cusp * (rDM / r0)**(-7/4)

#Convert enclosed mass to mascons
x_right = rDM[::round(n/(N+1))] # ri's of mascon shells
ris = x_right[1:]
y_right = enclosedMass(rDM,rho0plum)[::round(n/(N+1))] # enclosed mi's of mascon shells
#mascon masses = difference in enclosed mass:
mis = [t - s for s, t in zip(y_right, y_right[1:])]

# print('ris:',ris)
# print('mis:',mis)

#Check numerical stability by setting most mascons to zero:
# for i in range(len(mis)):
#     if i != 1 and i != 4 and i != 7:
#         mis[i] = 0
# print('mis:',mis)

#Plot enclosed mass
# plt.figure()
# plt.xlabel('Distance from MBH [AU]')
# plt.plot(rDM,enclosedMass(rDM,rho0plum),label='Plum model')
# plt.ylabel('enclosed mass [MBH masses]')
# plt.axvline(rp,linestyle='--',label='rp and ra',color='black')
# plt.axvline(ra,linestyle='--',color='black')
# plt.scatter(x_right[1:],y_right[1:],color='orange',label='Mascon enclosed mass')
# plt.bar(x_right,y_right,width=(xlim)/(N+1),alpha=0.2,align='edge',edgecolor='orange',color='orange')
# plt.legend()

#Plot mass of mascons
# plt.figure()
# #Plot rp, ra for ellipses:
# a = 1034.2454074981154
# e = 0.884429099282 
# rp = a*(1-e)
# ra = a*(1+e)
# plt.scatter(ris,mis,label='mascons',color='orange')
# plt.axvline(rp,linestyle='--',label='rp and ra',color='black')
# plt.axvline(ra,linestyle='--',color='black')
# plt.ylabel('Mass [MBH masses]',color='orange')
# plt.xlabel('Distance from MBH [AU]')
# plt.legend()



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


def effectOfAmountOfMascons():
    rp = 119.52867
    ra = 1948.96214
    
    
    N = 20
    k = 0.01
    
    
    # x_right = rDM[::round(n/(N+1))] # ri's of mascon shells
    # ris20 = x_right[1:]
    # y_right = enclosedMass(rDM,rho0plum)[::round(n/(N+1))] # enclosed mi's of mascon shells
    # #mascon masses = difference in enclosed mass:
    # mis20 = [t - s for s, t in zip(y_right, y_right[1:])]
    
    # [rxDM20,ryDM20,rzDM20] , [vxDM20,vyDM20,vzDM20] ,lf20 = \
    #     simulateOrbits(True, mis20, ris20)
    
    
    # plt.figure()
    # plt.xlabel('Distance from MBH [AU]')
    # plt.plot(rDM,enclosedMass(rDM,rho0plum),label='Plum model')
    # plt.ylabel('enclosed mass [MBH masses]')
    # plt.axvline(rp,linestyle='--',label='rp and ra',color='black')
    # plt.axvline(ra,linestyle='--',color='black')
    # plt.scatter(x_right[1:],y_right[1:],color='orange',label='Mascon enclosed mass')
    # plt.bar(x_right,y_right,width=(xlim)/(N+1),alpha=0.2,align='edge',edgecolor='orange',color='orange')
    # plt.legend()
    
    N = 20
    
    x_right = rDM[::round(n/(N+1))] # ri's of mascon shells
    ris100 = (x_right[:-1] - x_right[1:])/2
    y_right = enclosedMass(rDM,rho0plum)[::round(n/(N+1))] # enclosed mi's of mascon shells
    #mascon masses = difference in enclosed mass:
    mis100 = [t - s for s, t in zip(y_right, y_right[1:])]
    # print(mis20)
    # print(mis100)
    
    # print(sum(mis20))
    # print(sum(mis100))
    
    [rxDM100,ryDM100,rzDM100] , [vxDM100,vyDM100,vzDM100] ,lf100 = \
        simulateOrbits(True, mis100, ris100)
    
    
    
    
    # Mascon model (mi, ri), sigmoid approximation of step function
    # heyoka parameter encoding: [m1,m2,...mn,r1,r2,...rn]
    #          ->  par[0..i] for mi, par[n+0..i] for ri
    listOfSigs = [0.5 + 0.5 * np.tanh( k * (rDM - ris[i])) for i in range(N)]
    # print(listOfSigs)
    
    listOfRis = [mis100[i]* listOfSigs[i] for i in range(N)]
    
    suml = listOfSigs[0]
    for i in range(1,len(listOfSigs)):
        suml = suml + listOfSigs[i]
        
    
    sumRis = listOfRis[0]
    for i in range(1,len(listOfRis)):
        sumRis = sumRis + listOfRis[i]
        
        
    #Plot sigmoids:
    # plt.figure()
    # plt.plot(rDM,suml)
    # plt.xlabel('Distance from MBH [AU]')
    # plt.axvline(rp,linestyle='--',label='rp and ra',color='black')
    # plt.axvline(ra,linestyle='--',color='black')
    # plt.ylabel('Sum of sigmoids')
    


    
    # plt.figure()
    # plt.xlabel('True anomaly (f)')
    # plt.ylabel('Difference [µas]')
    # # plt.title('X (k=1000) - X (k=100) , 100 mascons')
    # plt.title('X (NOTcompact) - X (compact) , k={}, N={} mascons'.format(k,N))
    # plt.scatter(lf100,AU_to_arcseconds(rxDM100-rxDM20)*1e6,s=10)
    # plt.plot(lf100,len(lf100)*[50],'--',label='Precision',color='red')
    # plt.plot(lf100,len(lf100)*[-50],'--',color='red')
    # plt.legend()
    
    
    # plt.figure()
    # plt.xlabel('True anomaly (f)')
    # plt.ylabel('X 1000mascs - X 100mascs  [µas]')
    # plt.scatter(lf100,AU_to_arcseconds(rxDM100-rxDM20)*1e6,s=10,label='1000 masc - 100 masc')
    # plt.plot(lf100,len(lf100)*[50],'--',label='Precision',color='red')
    # plt.plot(lf100,len(lf100)*[-50],'--',color='red')
    # plt.legend()
    
    
    
    plt.figure()
    plt.xlabel('Distance from MBH [AU]')
    plt.plot(rDM,enclosedMass(rDM,rho0plum),label='Plum model')
    plt.plot(rDM,sumRis,label='Sum of mass sigmoids')
    plt.ylabel('enclosed mass [MBH masses]')
    plt.axvline(rp,linestyle='--',label='rp and ra',color='black')
    plt.axvline(ra,linestyle='--',color='black')
    plt.scatter(x_right[1:],y_right[1:],color='orange',label='Mascon enclosed mass')
    plt.bar(x_right,y_right,width=(xlim)/(N+1),alpha=0.2,align='edge',edgecolor='orange',color='orange')
    plt.legend()


def effectOfIndividualMascons():
    
    mis0 = N*[0] #-> 0 dark matter, has no effect
    ris0 = np.linspace(0,1000,N)
    
    [rxPN,ryPN,rzPN] , [vxPN,vyPN,vzPN] , lf= simulateOrbits(True, mis0, ris0)
    
    #Plot individual differences of mascons:
    mis1 = mis.copy()
    # mis1[1] = 0.0001597836898638425
    mis1[4] = 0
    mis1[7] = 0
    
    [rxDM1,ryDM1,rzDM1] , [vxDM1,vyDM1,vzDM1] ,lf1 = simulateOrbits(True, mis1, ris)
    print(mis1)
    
    mis4 = mis.copy()
    # mis4[4] = 0.0001597836898638425
    mis4[1] = 0
    mis4[7] = 0
    
    [rxDM4,ryDM4,rzDM4] , [vxDM4,vyDM4,vzDM4],lf4 = simulateOrbits(True, mis4, ris)
    print(mis4)
    
    mis7 = mis.copy()
    # mis7[7] = 0.0001597836898638425
    mis7[1] = 0
    mis7[4] = 0
    
    [rxDM7,ryDM7,rzDM7] , [vxDM7,vyDM7,vzDM7],lf7 = simulateOrbits(True, mis7, ris)
    print(mis7)
    
    
    
    # #Plot difference of DM in function of time:
    # plt.figure()
    # plt.xlabel('Time (years)')
    # plt.ylabel('X PN - X DM  [µas]')
    # plt.scatter(timegrid,AU_to_arcseconds(rxPN-rxDM1)*1e6,s=10,label='mascon 1')
    # plt.scatter(timegrid,AU_to_arcseconds(rxPN-rxDM4)*1e6,s=10,label='mascon 4')
    # plt.scatter(timegrid,AU_to_arcseconds(rxPN-rxDM7)*1e6,s=10,label='mascon 7')
    # plt.plot(timegrid,len(timegrid)*[50],'--',label='Precision',color='red')
    # plt.plot(timegrid,len(timegrid)*[-50],'--',color='red')
    # plt.legend()
    
    #Plot in function of f:
    plt.figure()
    plt.xlabel('True anomaly (f)')
    plt.ylabel('X PN - X DM  [µas]')
    plt.scatter(lf1,AU_to_arcseconds(rxPN-rxDM1)*1e6,s=10,label='mascon 1')
    plt.scatter(lf4,AU_to_arcseconds(rxPN-rxDM4)*1e6,s=10,label='mascon 4')
    plt.scatter(lf7,AU_to_arcseconds(rxPN-rxDM7)*1e6,s=10,label='mascon 7')
    plt.plot(lf4,len(lf4)*[50],'--',label='Precision',color='red')
    plt.plot(lf4,len(lf4)*[-50],'--',color='red')
    plt.legend()
    
    plt.figure()
    plt.xlabel('True anomaly (f)')
    plt.ylabel('Y PN - Y DM  [µas]')
    plt.scatter(lf1,AU_to_arcseconds(ryPN-ryDM1)*1e6,s=10,label='mascon 1')
    plt.scatter(lf4,AU_to_arcseconds(ryPN-ryDM4)*1e6,s=10,label='mascon 4')
    plt.scatter(lf7,AU_to_arcseconds(ryPN-ryDM7)*1e6,s=10,label='mascon 7')
    plt.plot(lf4,len(lf4)*[50],'--',label='Precision',color='red')
    plt.plot(lf4,len(lf4)*[-50],'--',color='red')
    plt.legend()
    
    plt.figure()
    plt.xlabel('True anomaly (f)')
    plt.ylabel('Z PN - Z DM  [µas]')
    plt.scatter(lf1,AU_to_arcseconds(rzPN-rzDM1)*1e6,s=10,label='mascon 1')
    plt.scatter(lf4,AU_to_arcseconds(rzPN-rzDM4)*1e6,s=10,label='mascon 4')
    plt.scatter(lf7,AU_to_arcseconds(rzPN-rzDM7)*1e6,s=10,label='mascon 7')
    plt.plot(lf4,len(lf4)*[50],'--',label='Precision',color='red')
    plt.plot(lf4,len(lf4)*[-50],'--',color='red')
    plt.legend()
    
    plt.figure()
    plt.xlabel('True anomaly (f)')
    plt.ylabel('VX PN - VX DM [km/s]')
    plt.scatter(lf1,(vxPN-vxDM1)/1000,s=10,label='mascon 1')
    plt.scatter(lf4,(vxPN-vxDM4)/1000,s=10,label='mascon 4')
    plt.scatter(lf7,(vxPN-vxDM7)/1000,s=10,label='mascon 7')
    plt.plot(lf4,len(lf4)*[10],'--',label='Precision',color='red')
    plt.plot(lf4,len(lf4)*[-10],'--',color='red')
    plt.legend()
    
    plt.figure()
    plt.xlabel('True anomaly (f)')
    plt.ylabel('VY PN - VY DM  [km/s]')
    plt.scatter(lf1,(vyPN-vyDM1)/1000,s=10,label='mascon 1')
    plt.scatter(lf4,(vyPN-vyDM4)/1000,s=10,label='mascon 4')
    plt.scatter(lf7,(vyPN-vyDM7)/1000,s=10,label='mascon 7')
    plt.plot(lf4,len(lf4)*[10],'--',label='Precision',color='red')
    plt.plot(lf4,len(lf4)*[-10],'--',color='red')
    plt.legend()
    
    plt.figure()
    plt.xlabel('True anomaly (f)')
    plt.ylabel('VZ PN - VZ DM  [km/s]')
    plt.scatter(lf1,(vzPN-vzDM1)/1000,s=10,label='mascon 1')
    plt.scatter(lf4,(vzPN-vzDM4)/1000,s=10,label='mascon 4')
    plt.scatter(lf7,(vzPN-vzDM7)/1000,s=10,label='mascon 7')
    plt.plot(lf4,len(lf4)*[10],'--',label='Precision',color='red')
    plt.plot(lf4,len(lf4)*[-10],'--',color='red')
    plt.legend()


def plotDifferenceWIth1PN():
    mis0 = N*[0] #-> 0 dark matter, has no effect
    ris0 = np.linspace(0,1000,N)
    [rxPN,ryPN,rzPN] , [vxPN,vyPN,vzPN] , lf= simulateOrbits(True, mis0, ris0)
    
    [rxDM,ryDM,rzDM] , [vxDM,vyDM,vzDM] = simulateOrbits(True, mis, ris)
    
    #Plot difference of DM:
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('X PN - X DM  [µas]')
    plt.scatter(timegrid,AU_to_arcseconds(rxPN-rxDM)*1e6,s=10,label='Difference')
    plt.plot(timegrid,len(timegrid)*[50],'--',label='Precision',color='red')
    plt.plot(timegrid,len(timegrid)*[-50],'--',color='red')
    plt.legend()
    
    #Plot difference of DM:
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('Y PN - Y DM  [µas]')
    plt.scatter(timegrid,AU_to_arcseconds(ryPN-ryDM)*1e6,s=10,label='Difference')
    plt.plot(timegrid,len(timegrid)*[50],'--',label='Precision',color='red')
    plt.plot(timegrid,len(timegrid)*[-50],'--',color='red')
    plt.legend()
    
    
    #Plot difference of DM:
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('VZ PN - VZ DM  [km/s]')
    plt.scatter(timegrid,-(vzPN-vzDM)/1000,s=10,label='Difference')
    plt.plot(timegrid,len(timegrid)*[10],'--',label='Precision',color='red')
    plt.plot(timegrid,len(timegrid)*[-10],'--',color='red')
    plt.legend()
    
    
    
    #Plot effects of DM:
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('Value')
    plt.scatter(timegrid,AU_to_arcseconds(rxDM)*1e6,label='X DM , [µas]',s=10)
    plt.scatter(timegrid,AU_to_arcseconds(rxPN)*1e6,label='X PN , [µas]',s=10)
    plt.legend()
    
    
    #Plot effects of DM:
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('Value')
    plt.scatter(timegrid,AU_to_arcseconds(ryDM)*1e6,label='Y DM , [µas]',s=10)
    plt.scatter(timegrid,AU_to_arcseconds(ryPN)*1e6,label='Y PN , [µas]',s=10)
    plt.legend()
    
    
    #Plot effects of DM:
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('Value')
    plt.scatter(timegrid,-vzDM/1000,label='VZ DM , [km/s]',s=10)
    plt.scatter(timegrid,-vzPN/1000,label='VZ PN , [km/s]',s=10)
    plt.legend()


# effectOfIndividualMascons()
# plotDifferenceWIth1PN()
effectOfAmountOfMascons()