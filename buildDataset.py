# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 14:56:35 2022

@author: Thibault
"""

import numpy as np
import orbitModule


def buildDatasets(Ns,xlim):
    print("Building datasets")
    IC = orbitModule.get_S2_IC()
    
    timegrid = orbitModule.getObservationTimes()
    
    #Distributions:
    names = ['Plummer','BahcallWolf','Uniform','Sinusoidal']
    
    for N in Ns:
        for name in names:
            getTrueDM = getattr(orbitModule,'get_'+name+'_DM')
            mis, ris = getTrueDM(N,xlim)
            
            rx,ry,rz,vx,vy,vz = orbitModule.simulateOrbitsCartesian(True, IC, mis, ris, timegrid)
            rx, ry, vz = orbitModule.convertXYVZtoArcsec(rx, ry, vz)
            data = np.column_stack((timegrid,ry,rx,vz))
            np.savetxt('Datasets/{}_N={}.txt'.format(name,N),data)
        
        
def setupTaylorIntegrators(Ns):
    print("Setting up taylor integrators, this could take some time")
    for N in Ns:
        orbitModule.buildTaylorIntegrator(True,N,SAVE_PICKLE=True)    


if __name__ == "__main__":
    #Dark matter:
    #Amount of mascons:
    Ns = [5,10]
    
    #Set dark matter distribution AU limit:
    xlim = 2100
    
    #Build the datasets
    buildDatasets(Ns, xlim)
    
    #Setup the taylor integrators
    # setupTaylorIntegrators(Ns)
    


