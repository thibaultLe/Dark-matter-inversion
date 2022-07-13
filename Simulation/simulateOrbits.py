# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 17:01:05 2022

@author: Thibault
"""
import heyoka as hy
import numpy as np
from matplotlib.pylab import plt
import time


"""
Simulates orbits of S2 around MBH

@param: PNCORRECTION: True if using 1PN correction
@param: mis: masses of dark matter shells in MBH masses
@param: ris: distances of dark matter shells in AU

@return: [rx,ry,rz],[vx,vy,vz] in metric units
"""
def simulateOrbits(PNCORRECTION,mis,ris):
    # Create the symbolic variables.
    p, e, i, om, w, f = hy.make_vars("p", "e", "i", "om", "w", "f")
    
    """
    Set parameters and initial conditions.
    """
    #Gravitational constant
    G_orig = 6.67430 * 10**(-11)
    #Solar mass
    M_sol = 1.98841 * 10**30
    
    #Using unit conversion to avoid huge numbers 
    # mass m' = m/M_0 -> MBH = 1
    M_0 = 4.2970174 * 10**6 * M_sol
    # distance r' = r/R_0 -> 1 AU = 1)
    D_0 = 149597870700
    # time t' = t/T_0 -> 1 time unit ~= 40 minutes (induced by G'=1))
    T_0 = np.sqrt((D_0**3)/(G_orig * M_0))
    
    # technically G'
    G = 1
    #MBH mass
    m1 = 1
    #S2 mass
    m2 = 0
    #Speed of light (in m/s, then converted) ~= 4.85 AU / 40 minutes
    c = 299792458 * T_0 / D_0
    #Constant that dictates steepness of sigmoid
    k = 10000
    #Amount of dark matter shells
    n = len(mis)
    
    
    #alpha in arcseconds
    alpha_mpe = 0.1249527719 
    #R in parsec
    R_mpe = 8277.09055007
    e_mpe = 0.884429099282  
    a_mpe = 2 * R_mpe * np.tan(alpha_mpe * np.pi / (2*648000)) * 3.08567758149e+16 / D_0
    p_mpe = a_mpe * (1-e_mpe**2) 
    T_period = np.sqrt(a_mpe**3)*2*np.pi
    # print(T_period)
    # T_0mpe =  2010.3561125977762 * 365.25 * 24 * 60**2 /T_0
    
    #Initial conditions:
    IC= [p_mpe, e_mpe, -134.700204975 / 180 * np.pi, 228.191510132 / 180 * np.pi, \
      66.2689390128 / 180 * np.pi, -np.pi]
    
    
    
    #Create a realistic observation time grid
    comparedData = np.loadtxt('Kepler.txt')
    timegrid = comparedData[:,0]
    #Use an offset so that t=0 corresponds to the first observation
    timeoffset = timegrid[0]
    timegrid = timegrid - timeoffset
    
    
    
    """
    Some common subformulas
    """
    M = m1 + m2
    GM = G*M
    pGM = hy.sqrt(p/GM)
    GMCP = GM**2 / (c**2 * p**3)
    
    ecf = e * hy.cos(f)
    ecf1 = 1 + ecf
    r = p / ecf1
    
    nu = m1 * m2 / (M**2)
    
    
    """
    Equations (21)-(24):
    """
    #(21)
    #  Using slight alternate form than in paper so we can reuse ecf1:
    R1PN = GMCP * ecf1**2 * ((3 * e**2) + 1. + 2 * ecf1  - (4 * ecf**2) \
          + 5 * nu * (1 - (7/10) * e**2) - 8 * nu * hy.cos(f)  + (1/2) * nu * ecf**2)
    
    # Mascon model (mi, ri), sigmoid approximation of step function
    # heyoka parameter encoding: [m1,m2,...mn,r1,r2,...rn]
    #          ->  par[0..i] for mi, par[n+0..i] for ri
    listOfSigs = [0.5 + 0.5 * hy.tanh( k * (r - hy.par[n+i])) for i in range(n)]
    listOfRis = [-G * hy.par[i] / (r**2) * listOfSigs[i] for i in range(n)]
    
    #(23)
    RDM = hy.sum(listOfRis)
    
    #(24),(22)
    if PNCORRECTION:
        R = R1PN + RDM
        S = GMCP * 2 * (2 - nu) * (ecf1**3) * e * hy.sin(f)
    else:
        R= RDM
        S = 0
    
    W = 0
    
    
    """
    Osculating equations
    """
    #(15)
    dpdt = pGM * p * (2/ecf1) * S
    #(16)
    dedt = pGM * (R * hy.sin(f) + S * (2 * hy.cos(f) + e*(1 + (hy.cos(f)**2)))/ecf1)
    #(17)
    didt = pGM * W * hy.cos(w+f)/ecf1
    #(18)
    domdt = pGM * W * hy.sin(w+f)/(ecf1 * hy.sin(i))
    #(19)
    #cot = 1/tan
    dwdt = pGM * (1/e) * (-R * hy.cos(f) + S * (1. + (1/ecf1) ) * hy.sin(f)  \
                          - W * e * (1/hy.tan(i)) * (hy.sin(w+f)/ecf1))
    #(20)
    dfdt = (1/(pGM*p)) * ecf1**2 + \
            pGM * (1/e) * (R * hy.cos(f) -  S * (1. + (1/ecf1) ) * hy.sin(f))
    
    
    """
    Instantiate the Taylor integrator
    """
    
    start_time = time.time()
    ta = hy.taylor_adaptive(
        # The ODEs.
        [(p, dpdt), (e, dedt), (i, didt), (om, domdt), (w, dwdt), (f, dfdt)],
        # The initial conditions 
        IC
    
    )
    print("--- %s seconds --- to build the Taylor integrator" % (time.time() - start_time))
    
    t_grid = timegrid * 365.25 * 24 * 60**2 /T_0
    #Roughly approximated by:
    # t_grid =  np.append(0,(np.linspace(0,16.056,228) * 365.25 * 24 * 60**2 /T_0 ) + 84187.772)

    
    
    """
    Set dark matter distribution (masses and radii of shells), in units of MBH masses!
    """
    ta.pars[:n] = mis
    
    ta.pars[n:] = ris
    
    
    start_time = time.time()
    out = ta.propagate_grid(t_grid)
    print("--- %s seconds --- to propagate" % (time.time() - start_time))
    
    
    #Convert to numpy arrays for plotting in 3D with x,y,z
    lp  = np.asarray(out[4][:, 0])
    le  = np.asarray(out[4][:, 1])
    li  = np.asarray(out[4][:, 2])
    lom = np.asarray(out[4][:, 3])
    lw  = np.asarray(out[4][:, 4])
    lf  = np.asarray(out[4][:, 5])
    
    
    lr = lp / (1 + le * np.cos(lf))
    
    # Plot parameters in function of time
    # plt.figure()
    # plt.plot(t_grid,lp,label='p')
    # plt.plot(t_grid,le,label='e')
    # plt.plot(t_grid,li,label='i')
    # plt.plot(t_grid,lom,label='Om')
    # plt.plot(t_grid,lw,label='w')
    # plt.plot(t_grid,lf,label='f')
    # plt.plot(t_grid,len(t_grid)*[np.pi])
    # plt.plot(t_grid,len(t_grid)*[-np.pi])
    # plt.plot(t_grid,len(t_grid)*[np.pi*4])
    # plt.xlabel("t")
    # plt.ylabel("Value")
    # plt.legend()
    
    # Plot period:
    # plt.figure()
    # plt.plot(t_grid,lf,label='f')
    # plt.plot(t_grid, len(t_grid)*[ lf[0] +2*np.pi])
    # plt.plot(t_grid,len(t_grid)*[ lf[0]])
    # # plt.plot(t_grid,len(t_grid)*[np.pi*4])
    # plt.xlabel("t")
    # plt.ylabel("Value")
    # plt.legend()
    
    
    # Position and velocity conversion
    rx = lr * (np.cos(lom) * np.cos(lw + lf) - np.cos(li)*np.sin(lom)*np.sin(lw+lf))
    ry = lr * (np.sin(lom) * np.cos(lw + lf) + np.cos(li)*np.cos(lom)*np.sin(lw+lf))
    rz = lr * np.sin(li) * np.sin(lw + lf)
    
    
    vx = -np.sqrt(GM/lp) * (np.cos(lom) * (np.sin(lw+lf) + le*np.sin(lw)) + \
             np.cos(li) * np.sin(lom) * (np.cos(lw+lf) + le*np.cos(lw)))
    vy = -np.sqrt(GM/lp) * (np.sin(lom) * (np.sin(lw+lf) + le*np.sin(lw)) - \
             np.cos(li) * np.cos(lom) * (np.cos(lw+lf) + le*np.cos(lw)))
    vz = np.sqrt(GM/lp) * np.sin(li) * (np.cos(lw+lf) + le * np.cos(lw))
    
    
    #Returns  observations (AU, meters/second)
    return [rx,ry,rz] , [vx * D_0 / T_0,vy * D_0 / T_0,vz * D_0 / T_0]



if __name__ == "__main__":
    
    #Amount of dark matter shells
    N = 5
    
    #Dark matter mascons (in MBH masses units)
    mis = N*[0] #-> 0 dark matter, has no effect
    
    #Mascon distance from MBH (in AU)
    ris = np.linspace(0,1000,N)
    
    [rx,ry,rz] , [vx,vy,vz] = simulateOrbits(False, mis, ris)
    
    
    
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
                rstride=1, cstride=1, color='black', linewidth=0, alpha=0.1,label='DM shell(s)')
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










