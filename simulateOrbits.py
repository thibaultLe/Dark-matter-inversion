# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 17:01:05 2022

@author: Thibault
"""
import numpy as np
from matplotlib.pylab import plt

import orbitModule


if __name__ == "__main__":
    
    #Amount of dark matter shells
    N = 20
    
    #Dark matter mascons (in MBH masses units)
    # mis = N*[0] #-> 0 dark matter, has no effect    
    mis = [8.265361613774615e-07, 5.686349100247396e-06, 1.4927307091440782e-05, 2.7672594520147764e-05, 4.2787249490168856e-05, 5.9043390760264746e-05, 7.527496444156235e-05, 9.049525234007644e-05, 0.00010396460998861513, 0.00011520962910339889, 0.00012400425819432555, 0.0001303272388261693, 0.00013430953743146022, 0.00013618220727268707, 0.00013623112544037478, 0.00013476151923237666, 0.00013207265668939558, 0.00012844157430656538, 0.00012411404664116991, 0.00011930089158700878]

    
    #Mascon distance from MBH (in AU)
    # ris = np.linspace(0,1000,N)
    ris =  [144.14414414414415, 288.2882882882883, 432.4324324324324, 576.5765765765766, 720.7207207207207, 864.8648648648648, 1009.009009009009, 1153.1531531531532, 1297.2972972972973, 1441.4414414414414, 1585.5855855855855, 1729.7297297297296, 1873.8738738738737, 2018.018018018018, 2162.162162162162, 2306.3063063063064, 2450.4504504504503, 2594.5945945945946, 2738.7387387387384, 2882.8828828828828]


    M_0, D_0, T_0 = orbitModule.getBaseUnitConversions()
    
    
    IC = orbitModule.get_S2_IC()
        
        
    #Time grid:
    t_grid =  np.append(0,(np.linspace(0,16.056,228) * 365.25 * 24 * 60**2 /T_0 ) + 84187.772)
    
    
    rx,ry,rz,vx,vy,vz= orbitModule.simulateOrbitsCartesian(False, IC, mis, ris, t_grid)
    
    
    
    #Plot position and MBH
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(rx[1:], ry[1:], rz[1:], label='Position')
    ax.scatter(rx[0], ry[0], rz[0], label='Start',color='lawngreen')
    ax.scatter(rx[-1], ry[-1], rz[-1], label='End',color="red")
    ax.scatter(0,0,0,color='black',label="MBH")
    ax.set_xlabel('rX')
    ax.set_ylabel('rY')
    ax.set_zlabel('rZ')
    
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
                rstride=1, cstride=1, color='black', linewidth=0, alpha=0.01,label='DM shell(s)')
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










