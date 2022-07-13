# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 21:28:46 2022

@author: Thibault
"""

import heyoka as hy
import numpy as np
from matplotlib.pylab import plt
import time
import pickle


"""
Reconstructs dark matter distribution starting from an initial guess

@param: PNCORRECTION: True if using 1PN correction
@param: mis: initial guesses of masses of dark matter shells in MBH masses
@param: ris: distances of dark matter shells in AU

@return: mis distribution
"""
def reconstructDistribution(PNCORRECTION,mis,ris):
    
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
    Variational equations
    """
    x = np.array([p,e,i,om,w,f])
    func = [dpdt,dedt,didt,domdt,dwdt,dfdt]
    
    #Phi:
    symbols_phi = []
    for i in range(6):
        for j in range(6):
            symbols_phi.append("phi_"+str(i)+str(j))  
    phi = np.array(hy.make_vars(*symbols_phi)).reshape((6,6))
    
    dfdx = []
    for i in range(6):
        for j in range(6):
            dfdx.append(hy.diff(func[i],x[j]))
    dfdx = np.array(dfdx).reshape((6,6))
    
    dphidt = dfdx@phi
    
    #Psi:
    symbols_psi = []
    for i in range(6):
        for j in range(6):
            symbols_psi.append("psi_"+str(i)+str(j))  
    psi = np.array(hy.make_vars(*symbols_psi)).reshape((6,6))
    
    #ris is a parameter in heyoka, but should be fixed value. Impact on variational eqs?
    dfdx = []
    for i in range(6):
        for j in range(6):
            dfdx.append(hy.diff(func[i],hy.par[j]))
    dfdx = np.array(dfdx).reshape((6,6))
    
    # The (variational) equations of motion
    dpsidt = dfdx@psi
    
    
    
    dyn = []
    for state, rhs in zip(x,func):
        dyn.append((state, rhs))
    for state, rhs in zip(phi.reshape((36,)),dphidt.reshape((36,))):
        dyn.append((state, rhs))
    for state, rhs in zip(psi.reshape((36,)),dpsidt.reshape((36,))):
        dyn.append((state, rhs))
    # These are the initial conditions on the variational equations (the identity matrix)
    ic_var_phi = np.eye(6).reshape((36,)).tolist()
    ic_var_psi = np.zeros((36,)).tolist()
    
    
    
    """
    Instantiate the Taylor integrator
    """
    #Optionally use pickle to save/load ta
    if not SAME_PARAMS:
        start_time = time.time()
        ta = hy.taylor_adaptive(
            # The ODEs.
            dyn,
            # [(p, dpdt), (e, dedt), (i, didt), (om, domdt), (w, dwdt), (f, dfdt)],
            # The initial conditions 
            IC + ic_var_phi + ic_var_psi,
            compact_mode=True
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
    # t_grid =  np.append(0,(np.linspace(0,16.056,10*228) * 365.25 * 24 * 60**2 /T_0 ) + 84187.772)

    
    
    """
    Set dark matter distribution (masses and radii of shells), in units of MBH masses!
    """
    ta.pars[:N] = mis
    
    ta.pars[N:] = ris
    
    
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
    return [rx,ry,rz] , [vx * D_0 / T_0,vy * D_0 / T_0,vz * D_0 / T_0]



reconstructDistribution(True,[1],[1])

