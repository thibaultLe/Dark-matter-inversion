# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 17:08:53 2022

@author: Thibault
"""
import numpy as np
from matplotlib.pylab import plt
import orbitModule


def validateSimulations(N,PNCORRECTION):
    def getDataFromText(filename):
        return np.loadtxt(filename)
    
    
    mis = N*[0] #-> 0 dark matter, has no effect
    ris = orbitModule.get_DM_distances(N, 2100)
    
    
    
    """
    Read data from files
    """
    if PNCORRECTION:
        txtfile = 'Datasets/1PN.txt'
    else:
        txtfile = 'Datasets/Kepler.txt'
        
    comparedData = getDataFromText(txtfile)
    timegrid = comparedData[:,0]
    comparedYs = comparedData[:,1]
    comparedXs = comparedData[:,2]
    comparedVZs = comparedData[:,3]
    
    
    
    
    IC = orbitModule.get_S2_IC()
    
    rx,ry,rz, vx,vy,vz = orbitModule.simulateOrbitsCartesian(PNCORRECTION,IC, mis, ris,timegrid)
    
    
    ydifs = []
    xdifs = []
    vzdifs = []
    for i in range(len(timegrid)):
        ydifs.append(orbitModule.AU_to_arcseconds(ry[i]) - comparedYs[i])
        xdifs.append(orbitModule.AU_to_arcseconds(rx[i]) - comparedXs[i])
        vzdifs.append(-vz[i] / (1000) - comparedVZs[i])
    
    
    
    # Plot difference with the baseline's solutions
    
    plt.figure()
    plt.plot(timegrid,orbitModule.AU_to_arcseconds(rx),label='DEC (x), [as]')
    plt.plot(timegrid,comparedXs,label='Truth')
    plt.xlabel('Time (years)')
    plt.ylabel('Value')
    plt.title('Comparison of X solutions')
    plt.legend()
    
    plt.figure()
    plt.scatter(timegrid,orbitModule.AU_to_arcseconds(ry),label='RA (y), [as]',s=10)
    plt.scatter(timegrid,comparedYs,label='Truth',s=10)
    plt.xlabel('Time (years)')
    plt.ylabel('Value')
    plt.title('Comparison of Y solutions')
    plt.legend()
    
    plt.figure()
    plt.plot(timegrid,-vz / 1000 ,label='RV (vz), [km/s]')
    plt.plot(timegrid,comparedVZs,label='Truth')
    plt.xlabel('Time (years)')
    plt.ylabel('Value')
    plt.title('Comparison of VZ solutions')
    plt.legend()
    
    
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('Difference')
    plt.scatter(timegrid,xdifs,label='X_heyoka - X_scipy , [as]',s=10)
    plt.title('Difference of X solutions')
    plt.legend()
    
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('Difference')
    plt.scatter(timegrid,ydifs,label='Y_heyoka - Y_scipy , [as]',s=10)
    plt.title('Difference of Y solutions')
    plt.legend()
    
    plt.figure()
    plt.xlabel('Time (years)')
    plt.ylabel('Difference')
    plt.scatter(timegrid,vzdifs,label='VZ_heyoka - VZ_scipy , [km/s]',s=10)
    plt.title('Difference of VZ solutions')
    plt.legend()
    




def compareDifferentTimegridSpacingTypes():
    
    comparedData = np.loadtxt('Datasets/Kepler.txt')
    timegrid = comparedData[:,0]
    comparedYs = comparedData[:,1]
    comparedXs = comparedData[:,2]
    
    #Compare spacings:
    _, _, T_0 = orbitModule.getBaseUnitConversions()
    ourt_grid =  np.append(0,(np.linspace(0,16.056,228) * 365.25 * 24 * 60**2 /T_0 ) + 84187.772)
    #Convert time to years
    ourtimegrid = 2.010356112597776246e+03 + ourt_grid * T_0 / (365.25 * 24 * 60**2 )   
    # t_grid = orbitModule.convertYearsTimegridToOurFormat(ourtimegrid)
    
    
    IC = orbitModule.get_S2_IC()
    rx,ry,rz, vx,vy,vz = orbitModule.simulateOrbitsCartesian(True,IC, [0], [1],ourtimegrid)
    
    plt.figure()
    plt.hist(timegrid,bins=20)
    plt.title('Equal spatial spacing')
    plt.xlabel('Time [years]')
    plt.ylabel('Number of observations')
    plt.figure()
    plt.hist(ourtimegrid,bins=20)
    plt.title('Equal temporal spacing')
    plt.xlabel('Time [years]')
    plt.ylabel('Number of observations')
    
    plt.figure()
    plt.hist(np.sqrt(rx**2+ry**2),bins=20)
    plt.ylabel('Number of observations')
    plt.xlabel('Distance from MBH [AU]')
    plt.title('Equal temporal spacing')
    plt.figure()
    plt.hist(np.sqrt(comparedXs**2+comparedYs**2),bins=20)
    plt.ylabel('Number of observations')
    plt.xlabel('Distance from MBH [AU]')
    plt.title('Equal spatial spacing')



if __name__ == "__main__":
    #Amount of dark matter shells
    N = 10
    #Newtonian or post newtonian
    PNCORRECTION = False
    
    #Compare heyoka and scipy solutions 
    validateSimulations(N,PNCORRECTION)
    
    #Compare timegrid spacing
    compareDifferentTimegridSpacingTypes()