# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 17:01:05 2022

@author: Thibault
"""
import heyoka as hy
import numpy as np
from matplotlib.pylab import plt
import time
import pickle


"""
Simulates orbits of S2 around MBH

@param: PNCORRECTION: True if using 1PN correction
@param: mis: masses of dark matter shells in MBH masses
@param: ris: distances of dark matter shells in AU

@return: [rx,ry,rz],[vx,vy,vz] in metric units
"""
def simulateOrbits(PNCORRECTION,mis,ris):
    
    #Can use cached version of system or not
    SAME_PARAMS = False
    
    
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
    k = 0.01
    #Amount of dark matter shells
    N = len(mis)
    
    
    
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
    # IC= [p_mpe, e_mpe, -134.700204975 / 180 * np.pi, 228.191510132 / 180 * np.pi, \
    #   66.2689390128 / 180 * np.pi, -np.pi]
        
    # IC= [227.49574 ,  0.88196 , -2.36897 ,  4.02457 ,  1.16789 ,  1.00998]
    IC= [225.24331 ,  0.88443 , -2.35096 ,  3.98269 ,  1.15661 ,  1.     ]
    
    
    
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
    listOfSigs = [0.5 + 0.5 * hy.tanh( k * (r - hy.par[N+i])) for i in range(N)]
    listOfRis = [-G * hy.par[i] / (r**2) * listOfSigs[i] for i in range(N)]
    
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
    
    
    if not SAME_PARAMS:
        start_time = time.time()
        ta = hy.taylor_adaptive(
            # The ODEs.
            [(p, dpdt), (e, dedt), (i, didt), (om, domdt), (w, dwdt), (f, dfdt)],
            # The initial conditions 
            IC,
            compact_mode = True
        )
        print("--- %s seconds --- to build the Taylor integrator" % (time.time() - start_time))
        
        # ## Pickle save/load
        # ta_file = open("ta_saved","wb")
        # pickle.dump(ta,ta_file)
        # ta_file.close()
        
    else:
        start_time = time.time()
        ta_file = open("ta_saved",'rb')
        ta = pickle.load(ta_file)
        ta_file.close()
        print("--- %s seconds --- to load the Taylor integrator" % (time.time() - start_time))
    
    
    
    t_grid = timegrid * 365.25 * 24 * 60**2 /T_0
    #Roughly approximated by:
    # t_grid =  np.append(0,(np.linspace(0,10*16.056,10*228) * 365.25 * 24 * 60**2 /T_0 ) + 84187.772)
    
    last_t = t_grid[-1]
    # print(last_t)

    
    
    """
    Set dark matter distribution (masses and radii of shells), in units of MBH masses!
    """
    ta.pars[:N] = mis
    
    ta.pars[N:] = ris
    
    
    start_time = time.time()
    
    out = ta.propagate_grid(t_grid)
    # print(ta.state)
    # print(out)
    
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
    # plt.scatter(t_grid,lf,label='f')
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
    return [rx,ry,rz] , [vx * D_0 / T_0,vy * D_0 / T_0,vz * D_0 / T_0], lf



if __name__ == "__main__":
    
    #Amount of dark matter shells
    N = 20
    
    #Dark matter mascons (in MBH masses units)
    mis = N*[0] #-> 0 dark matter, has no effect
    
    #Mascon distance from MBH (in AU)
    ris = np.linspace(0,1000,N)
    
    ris =  [144.14414414414415, 288.2882882882883, 432.4324324324324, 576.5765765765766, 720.7207207207207, 864.8648648648648, 1009.009009009009, 1153.1531531531532, 1297.2972972972973, 1441.4414414414414, 1585.5855855855855, 1729.7297297297296, 1873.8738738738737, 2018.018018018018, 2162.162162162162, 2306.3063063063064, 2450.4504504504503, 2594.5945945945946, 2738.7387387387384, 2882.8828828828828]
    mis = [8.265361613774615e-07, 5.686349100247396e-06, 1.4927307091440782e-05, 2.7672594520147764e-05, 4.2787249490168856e-05, 5.9043390760264746e-05, 7.527496444156235e-05, 9.049525234007644e-05, 0.00010396460998861513, 0.00011520962910339889, 0.00012400425819432555, 0.0001303272388261693, 0.00013430953743146022, 0.00013618220727268707, 0.00013623112544037478, 0.00013476151923237666, 0.00013207265668939558, 0.00012844157430656538, 0.00012411404664116991, 0.00011930089158700878]

    
    [rx,ry,rz] , [vx,vy,vz] ,lf= simulateOrbits(False, mis, ris)
    
    print(rx[-1],ry[-1],vz[-1])
    
    
    
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










