# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 17:01:05 2022

@author: Thibault
"""
import numpy as np
from matplotlib.pylab import plt

import orbitModule


def simulateOrbit(N,nbrOfOrbits=1):
    #Dark matter mascons (in MBH masses units), Mascon distance from MBH (in AU)
    mis,ris = orbitModule.get_Plummer_DM(N)
    
    # mis = N *[0]

    M_0, D_0, T_0 = orbitModule.getBaseUnitConversions()
    
    IC = orbitModule.get_S2_IC()
        
    timegrid = orbitModule.getObservationTimes(nbrOfOrbits)
    
    rx,ry,rz,vx,vy,vz= orbitModule.simulateOrbitsCartesian(True, IC, mis, ris, timegrid)
    
    
    #Plot position and MBH
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(rx, ry, rz, label='Orbit of S2')
    # ax.scatter(rx[0], ry[0], rz[0], label='Start',color='lawngreen')
    # ax.scatter(rx[-1], ry[-1], rz[-1], label='End',color="red")
    ax.scatter(0,0,0,color='black',label="MBH")
    ax.set_xlabel('X [AU]')
    ax.set_ylabel('Y [AU]')
    ax.set_zlabel('Z [AU]')
    
    #Plot DM shells (3D spheres)
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    
    x_sphere = 1 * np.outer(np.cos(u), np.sin(v))
    y_sphere = 1 * np.outer(np.sin(u), np.sin(v))
    z_sphere = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
    
    for i in range(N):
        #Only plot if dark matter mass is not zero
        if mis[i] != 0:
            surf =ax.plot_surface(ris[i]*x_sphere, ris[i]*y_sphere, ris[i]*z_sphere,  \
                rstride=1, cstride=1, color='black', linewidth=0, alpha=0.05,label='DM shell(s)')
            surf._facecolors2d = surf._facecolor3d
            surf._edgecolors2d = surf._edgecolor3d
            
    plotlim = (max(max(abs(rx)),max(abs(ry)),max(abs(rz))))
    ax.set_xlim(-plotlim,plotlim)
    ax.set_ylim(-plotlim,plotlim)
    ax.set_zlim(-plotlim,plotlim)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()
    
    
    #Plot velocity
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(vx[1:], vy[1:], vz[1:], label='Velocity')
    # ax.set_xlabel('vX')
    # ax.set_ylabel('vY')
    # ax.set_zlabel('vZ')
    # ax.legend()
    # plt.show()
    
    # velocity = np.sqrt(vx**2 + vy**2 + vz**2)
    # print(max(velocity))
    
    # Plot parameters in function of time
    # plt.figure()
    # plt.plot(t_grid,p,label='p')
    # plt.plot(t_grid,e,label='e')
    # plt.plot(t_grid,i,label='i')
    # plt.plot(t_grid,om,label='Om')
    # plt.plot(t_grid,w,label='w')
    # plt.plot(t_grid,f,label='f')
    # plt.plot(t_grid,len(t_grid)*[np.pi])
    # plt.plot(t_grid,len(t_grid)*[-np.pi])
    # plt.plot(t_grid,len(t_grid)*[np.pi*4])
    # plt.xlabel("t")
    # plt.ylabel("Value")
    # plt.legend()
    
    # Plot period:
    # plt.figure()
    # plt.scatter(t_grid,f,label='f')
    # plt.plot(t_grid, len(t_grid)*[ f[0] +2*np.pi])
    # plt.plot(t_grid,len(t_grid)*[ f[0]])
    # # plt.plot(t_grid,len(t_grid)*[np.pi*4])
    # plt.xlabel("t")
    # plt.ylabel("Value")
    # plt.legend()
    
#Visualize the orbits of the initial guess to the reconstruction and true values
#Note: very hard to see any difference with the default parameters
def compareInitialToReconstructed(N):
    
    #Reconstructed:
    reconstructedMis = [0.0001140864401540135, 0.00011429616443128367, 0.00011451944876234744, 0.0001146896144624966, 0.00011479470708703137, 0.00011461909284082165, 0.00011367249001702631, 0.00011040129335069722, 0.00010256063244500486, 5.376469046985896e-11]
    reconstructedIC = [225.24331,   0.88443,  -2.35101,   3.98273,   1.15656,  -3.14159]
    
    #Initial guess:
    initialguessMis = N*[0]
    initialguessIC = orbitModule.get_S2_IC()
    
    
    #Dark matter mascons (in MBH masses units), Mascon distance from MBH (in AU)
    mis,ris = orbitModule.get_Plummer_DM(N)

    M_0, D_0, T_0 = orbitModule.getBaseUnitConversions()
    
    
    IC = orbitModule.get_S2_IC()
        
    #Time grid:
    t_grid = orbitModule.getObservationTimes()
    
    rx,ry,rz,vx,vy,vz= orbitModule.simulateOrbitsCartesian(True, IC, mis, ris, t_grid)
    
    rxR,ryR,rzR,vxR,vyR,vzR= orbitModule.simulateOrbitsCartesian(True, reconstructedIC, reconstructedMis, ris, t_grid)
    
    rxI,ryI,rzI,vxI,vyI,vzI= orbitModule.simulateOrbitsCartesian(True, initialguessIC, initialguessMis, ris, t_grid)
    
    
    #Plot position and MBH
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(rxI, ryI, rzI, label='Initial guess')
    ax.plot(rxR, ryR, rzR, label='Reconstructed',color='tab:orange')
    ax.plot(rx, ry, rz, label='True',color='tab:blue')
    # ax.scatter(rx[0], ry[0], rz[0], label='Start',color='lawngreen')
    # ax.scatter(rx[-1], ry[-1], rz[-1], label='End',color="red")
    ax.scatter(0,0,0,color='black',label="MBH")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    #Plot DM shells (3D spheres)
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    
    x_sphere = 1 * np.outer(np.cos(u), np.sin(v))
    y_sphere = 1 * np.outer(np.sin(u), np.sin(v))
    z_sphere = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
    
    for i in range(N):
        #Only plot if dark matter mass is not zero
        if mis[i] != 0:
            surf =ax.plot_surface(ris[i]*x_sphere, ris[i]*y_sphere, ris[i]*z_sphere,  \
                rstride=1, cstride=1, color='black', linewidth=0, alpha=0.02,label='DM shell(s)')
            surf._facecolors2d = surf._facecolor3d
            surf._edgecolors2d = surf._edgecolor3d
            
    plotlim = (max(max(abs(rx)),max(abs(ry)),max(abs(rz))))
    ax.set_xlim(-plotlim,plotlim)
    ax.set_ylim(-plotlim,plotlim)
    ax.set_zlim(-plotlim,plotlim)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()



if __name__ == "__main__":
    #Amount of dark matter shells
    N = 10
    
    simulateOrbit(N,nbrOfOrbits=1)
    
    # compareInitialToReconstructed(N)
    






