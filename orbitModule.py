# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:11:01 2022

@author: Thibault
"""


import heyoka as hy
import numpy as np
import time
import pickle



def getBaseUnitConversions():
    """

    Returns
    -------
    M_0 : float
        Mass in kg.
    D_0 : float
        Distance in meters.
    T_0 : float
        Time in seconds.

    """
    #Gravitational constant
    G_orig = 6.67430 * 10**(-11)
    #Solar mass
    M_sol = 1.98841 * 10**30
    
    #Using unit conversion to avoid huge numbers 
    # mass m' = m/M_0       -> 1 MBH == m=1
    M_0 = 4.2970174 * 10**6 * M_sol
    # distance r' = r/R_0   -> 1 AU == r=1
    D_0 = 149597870700
    # time t' = t/T_0       -> 2422.97 seconds == t=1
    T_0 = np.sqrt((D_0**3)/(G_orig * M_0))
    
    return M_0, D_0, T_0

def convertYearsTimegridToOurFormat(timegrid):
    """
    Converts a given timegrid in years format to a format compatible with ours

    Parameters
    ----------
    timegrid : list of floats
        Observation times in years (e.g. [2010.356, 2013.23, ...]).

    Returns
    -------
    timegrid : list of floats
        Observation times in our base units.

    """
    _, _, T_0 = getBaseUnitConversions()
    #Use an offset so that t=0 corresponds to the first observation
    timeoffset = timegrid[0]
    timegrid = timegrid - timeoffset
    #Multiply by (amount of seconds in a year) / amount of seconds in T_0
    timegrid = timegrid * 365.25 * 24 * 60**2 / T_0
    
    return timegrid

def get_S2_IC():
    """
    Returns the initial conditions for S2 at t0 = 2010.3561125977762.

    Returns
    -------
    IC : list of floats
        [p,e,i,om,w,f].

    """
    _,D_0,_ = getBaseUnitConversions()
    
    # alpha in arcseconds
    alpha_mpe = 0.1249527719 
    
    #R in parsec
    R_mpe = 8277.09055007
    
    #Amount of AU in parsec
    AUsin1Parsec = 648000 / np.pi
    
    #Eccentricity
    e_mpe = 0.884429099282  
    a_mpe = 2 * R_mpe * np.tan(alpha_mpe / (2*AUsin1Parsec)) * AUsin1Parsec
    #Semi-latus rectum
    p_mpe = a_mpe * (1-e_mpe**2) 
    #Inclination
    i_mpe = -134.700204975
    #Longitude of ascending node
    om_mpe = 228.191510132
    #Periapsis
    w_mpe = 66.2689390128
    #True anomaly
    f_mpe = -np.pi
    
    #Some useful derived parameters:
    #Period:
    # T_period = np.sqrt(a_mpe**3)*2*np.pi
    #Initial time
    # T_0mpe =  2010.3561125977762 * 365.25 * 24 * 60**2 /T_0
    
    #Initial conditions:
    IC= [p_mpe, e_mpe, i_mpe / 180 * np.pi, om_mpe / 180 * np.pi, \
      w_mpe / 180 * np.pi, f_mpe]
        
    return IC

def get_Plummer_DM(N,xlim):
    """
    Returns the Plummer Dark matter model, discretised in mascons

    Parameters
    ----------
    N : int
        Amount of mascon shells.
    xlim : float
        Maximum distance of shells from 0.

    Returns
    -------
    mis : list of floats
        Dark matter masses.
    ris : list of floats
        Dark matter distances.

    """
    
    def enclosedMass(a,rho0):
        return (4 * a**3 * np.pi * r0**3 * rho0) / ( 3 * (a**2 + r0**2)**(3/2))
    
    M_0, D_0, T_0 = getBaseUnitConversions()
    
    rho0plum = 1.69*10**(-10) * (D_0**3) / M_0
    # rho0cusp = 2.24*10**(-11) * (D_0**3) / M_0
    r0 = 2474.01
    
    x_right = np.linspace(0,xlim,N+1) # ri's of mascon shells
    ris = x_right[1:]
    y_right = enclosedMass(x_right,rho0plum) # enclosed mi's of mascon shells
    #mascon masses = difference in enclosed mass:
    mis = [t - s for s, t in zip(y_right, y_right[1:])]
    
    return mis, ris

def AU_to_arcseconds(dist):
    """
    

    Parameters
    ----------
    dist : float or np.array
        Distance in [AU] to be converted.

    Returns
    -------
    float or np.array
        The converted distance in arcseconds.

    """
    D_0 = 149597870700
    R = 2.5540153e+20
    return 2 * np.arctan(dist*D_0/(2*R)) * 206264.8




def convertToCartesian(p,e,i,om,w,f):
    """
    

    Parameters
    ----------
    p : float
        semi-latus rectum.
    e : float
        eccentricity.
    i : float
        inclination.
    om : float
        longitude of ascending node.
    w : float
        periapsis.
    f : float
        true anomaly.

    Returns
    -------
    rx : float
        radial x value in [AU]. Multiply by D_0 to get metric m
    ry : float
        radial y value in [AU]. Multiply by D_0 to get metric m
    rz : float
        radial z value in [AU]. Multiply by D_0 to get metric m
    vx : float
        vx value in [AU/T_0]. Multiply by D_0 / T_0 to get metric m/s
    vy : float
        vy value in [AU/T_0]. Multiply by D_0 / T_0 to get metric m/s
    vz : float
        vz value in [AU/T_0]. Multiply by D_0 / T_0 to get metric m/s

    """
    
    
    r = p / (1 + e * np.cos(f))
    
    # Position 
    rx = r * (np.cos(om) * np.cos(w + f) - np.cos(i)*np.sin(om)*np.sin(w+f))
    ry = r * (np.sin(om) * np.cos(w + f) + np.cos(i)*np.cos(om)*np.sin(w+f))
    rz = r * np.sin(i) * np.sin(w + f)
    
    # Veocity 
    vx = -np.sqrt(1/p) * (np.cos(om) * (np.sin(w+f) + e*np.sin(w)) + \
             np.cos(i) * np.sin(om) * (np.cos(w+f) + e*np.cos(w)))
    vy = -np.sqrt(1/p) * (np.sin(om) * (np.sin(w+f) + e*np.sin(w)) - \
             np.cos(i) * np.cos(om) * (np.cos(w+f) + e*np.cos(w)))
    vz = np.sqrt(1/p) * np.sin(i) * (np.cos(w+f) + e * np.cos(w))
    
    return rx,ry,rz,vx,vy,vz


def cartesianConversionGradient(): 
    """
    
    Returns
    -------
    derobsdx : list of hy.expression
        List of derivatives of cartesian observation wrt orbital parameters.

    """
    
    # Create the symbolic variables.
    p, e, i, om, w, f = hy.make_vars("p", "e", "i", "om", "w", "f")
    x    = np.array([p,e,i,om,w,f])
    
    """
    Cartesian position and velocity conversion
    """
    r = p / (1 + e * hy.cos(f))
    
    rx = r * (hy.cos(om) * hy.cos(w + f) - hy.cos(i)*hy.sin(om)*hy.sin(w+f))
    ry = r * (hy.sin(om) * hy.cos(w + f) + hy.cos(i)*hy.cos(om)*hy.sin(w+f))
    rz = r * hy.sin(i) * hy.sin(w + f)
    
    vx = -hy.sqrt(1/p) * (hy.cos(om) * (hy.sin(w+f) + e * hy.sin(w)) + \
              hy.cos(i) * hy.sin(om) * (hy.cos(w+f) + e * hy.cos(w)))
    vy = -hy.sqrt(1/p) * (hy.sin(om) * (hy.sin(w+f) + e * hy.sin(w)) - \
              hy.cos(i) * hy.cos(om) * (hy.cos(w+f) + e * hy.cos(w)))
    vz = hy.sqrt(1/p) * hy.sin(i) * (hy.cos(w+f) + e * hy.cos(w))

    cart = np.array([rx,ry,rz,vx,vy,vz])
    
    #Derivative of cartesian observation wrt orbital parameters
    derobsdx = []
    #rows
    for i in range(6):
        #columns
        for j in range(6):
            derobsdx.append(hy.diff(cart[i],x[j]))
            
    return derobsdx
    
def variationalEqsInitialConditions(N):
    """
    

    Parameters
    ----------
    N : int
        Amount of dark matter shells.

    Returns
    -------
    list of ints
        The initial conditions for the variational equations.

    """
    ic_var_phi = np.eye(6).reshape((36,)).tolist()
    ic_var_psi = np.zeros((6*N,)).tolist()
    
    return ic_var_phi + ic_var_psi


def buildTaylorIntegrator(PNCORR,N,include_variational_eqs=True,LOAD_PICKLE=False,verbose=False):
    """
    
    Parameters
    ----------
    PNCORR : boolean
        True if using the 1PN correction, false if using Kepler mechanics.
    N : int
        Amount of dark matter shells.
    include_variational_eqs : bool, optional
        True if you want to include variational equations (e.g. to reconstruct IC's and DM distribution). Slows the process down a bit. The default is True.
    LOAD_PICKLE : bool, optional
        True if you want to load a previously saved version of ta. The default is False.
    verbose : bool, optional
        True if you want to print information on the process. The default is False.

    Returns
    -------
    ta : heyoka.core._taylor_adaptive_dbl
        The heyoka taylor integrator.
    
    """
    
    
    
    if LOAD_PICKLE:
        start_time = time.time()
        ta_file = open("ta_saved",'rb')
        ta = pickle.load(ta_file)
        ta_file.close()
        if verbose:
            print("--- %s seconds --- to load the Taylor integrator" % (time.time() - start_time))
        return ta
    
    else:
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
        k = 0.1
        
        
        
        #alpha in arcseconds
        alpha_mpe = 0.1249527719 
        #R in parsec
        R_mpe = 8277.09055007
        e_mpe = 0.884429099282  
        a_mpe = 2 * R_mpe * np.tan(alpha_mpe * np.pi / (2*648000)) * 3.08567758149e+16 / D_0
        p_mpe = a_mpe * (1-e_mpe**2) 
        # T_period = np.sqrt(a_mpe**3)*2*np.pi
        
        #Initial conditions:
        IC= [p_mpe, e_mpe, -134.700204975 / 180 * np.pi, 228.191510132 / 180 * np.pi, \
          66.2689390128 / 180 * np.pi, -np.pi]
        
           
        
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
        if PNCORR:
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
        
        
        x    = np.array([p,e,i,om,w,f])
        func = np.array([dpdt,dedt,didt,domdt,dwdt,dfdt])
        
        dyn = []
        for state, rhs in zip(x,func):
            dyn.append((state, rhs))
            
        """
        Variational equations
        """
        if include_variational_eqs:
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
            for i in range(N):
                for j in range(6):
                    symbols_psi.append("psi_"+str(i)+str(j))  
            psi = np.array(hy.make_vars(*symbols_psi)).reshape((6,N))
            
            dpsidt = []
            for i in range(6):
                for j in range(N):
                    dpsidt.append(hy.diff(func[i],hy.par[j]))
            dpsidt = np.array(dpsidt).reshape((6,N))
            
            #Phi
            for state, rhs in zip(phi.reshape((36,)),dphidt.reshape((36,))):
                dyn.append((state, rhs))
            #Psi
            for state, rhs in zip(psi.reshape((6*N,)),dpsidt.reshape((6*N,))):
                dyn.append((state, rhs))
            # Initial conditions on the variational equations
            ic_var_phi = np.eye(6).reshape((36,)).tolist()
            ic_var_psi = np.zeros((6*N,)).tolist()
            
            IC = IC + ic_var_phi + ic_var_psi
            
        """
        Instantiate the Taylor integrator
        """
        
        start_time = time.time()
        ta = hy.taylor_adaptive(
            # The ODEs.
            dyn,
            # The initial conditions 
            IC,
            compact_mode = True
        )
        if verbose:
            print("--- %s seconds --- to build the Taylor integrator" % (time.time() - start_time))
        
        # ## Pickle save/load
        ta_file = open("ta_saved","wb")
        pickle.dump(ta,ta_file)
        ta_file.close()
        
        return ta
    


def simulateOrbits(PNCORRECTION,IC,mis,ris,t_grid):
    """
    

    Parameters
    ----------
    PNCORRECTION : boolean
        True if using the 1PN correction, false if using Kepler mechanics.
    IC : list of floats
        Initial conditions of orbital parameters at t=0.
    mis : list of floats
        Dark matter shell masses in [MBH masses].
    ris : list of floats
        Dark matter shell distances from 0 in [AU].
    t_grid : list of floats
        Observation times.

    Raises
    ------
    RuntimeError
        If the length of dark matter masses and distances do not match, an error is raised.

    Returns
    -------
    p : list of floats
        semi-latus rectum.
    e : list of floats
        eccentricity.
    i : list of floats
        inclination.
    om : list of floats
        longitude of ascending node.
    w : list of floats
        periapsis.
    f : list of floats
        true anomaly.

    """
    
    if len(mis) != len(ris):
        raise RuntimeError("Lengths of DM masses and distances does not match")
        
    N = len(mis)
    
    ta = buildTaylorIntegrator(PNCORRECTION, N, include_variational_eqs=False)
    
    #Setup for fake reconstruction:
    ta.state[:6] = IC
    ta.pars[:N] = mis
    ta.pars[N:] = ris
    ta.time = 0
    
    out = ta.propagate_grid(t_grid)
    
    #Convert to numpy arrays for plotting in 3D with x,y,z
    p  = np.asarray(out[4][:, 0])
    e  = np.asarray(out[4][:, 1])
    i  = np.asarray(out[4][:, 2])
    om = np.asarray(out[4][:, 3])
    w  = np.asarray(out[4][:, 4])
    f  = np.asarray(out[4][:, 5])
    
    return p,e,i,om,w,f


def simulateOrbitsCartesian(PNCORRECTION,IC,mis,ris,t_grid):
    """

    Parameters
    ----------
    PNCORRECTION : boolean
        True if using the 1PN correction, false if using Kepler mechanics.
    IC : list of floats
        Initial conditions of orbital parameters at t=0.
    mis : list of floats
        Dark matter shell masses in [MBH masses].
    ris : list of floats
        Dark matter shell distances from 0 in [AU].
    t_grid : list of floats
        Observation times.

    Returns
    -------
    rx : list of floats
        radial x value in [AU]. Multiply by D_0 to get metric m
    ry : list of floats
        radial y value in [AU]. Multiply by D_0 to get metric m
    rz : list of floats
        radial z value in [AU]. Multiply by D_0 to get metric m
    vx : list of floats
        vx value in [m/s]. 
    vy : list of floats
        vy value in [m/s]. 
    vz : list of floats
        vz value in [m/s]. 

    """
    p,e,i,om,w,f = simulateOrbits(PNCORRECTION, IC, mis, ris, t_grid)
    rx,ry,rz,vx,vy,vz = convertToCartesian(p, e, i, om, w, f)
    M_0, D_0, T_0 = getBaseUnitConversions()
    
    #Returns  observations (AU, meters/second)
    return rx,ry,rz, vx * D_0 / T_0, vy * D_0 / T_0, vz * D_0 / T_0


def corrector(ta, x0, DMm0, obs, t_obs, alpha, beta1, beta2, eps, m, v, t,CARTESIANOBS=True):
    """
    

    Parameters
    ----------
    ta : hy.taylor_adaptive
        System of equations.
    x0 : list of floats
        initial condition guess (p,e,i,om,w,f)
    DMm0 : list of floats
        initial dark matter mass guess (m1,m2,...,mN).
    obs : list of list of floats
        observation at time tj of (p,e,i,om,w,f)
    t_obs : list of floats
        observation times tj
    alpha : float
        Learning rate for Adam.
    beta1 : float
        Exponential decay rate for the 1st moment estimates.
    beta2 : float
        Exponential decay rate for the 2nd moment estimates.
    eps : float
        A small constant for numerical stability.
    m : list of floats
        1st moment estimates.
    v : list of floats
        2nd moment estimates.
    t : int
        Iteration step number.
    CARTESIANOBS : boolean, optional
        True if using cartesian observations instead of orbital parameters. The default is True.

    Returns
    -------
    ta : hy.taylor_adaptive
        System of equations.
    x0_new : list of floats
        The corrected initial conditions (p,e,i,om,w,f).
    DM_new : list of floats
        The corrected dark matter mass guess (m1,m2,...,mN).
    simulatedlist : list of list of floats
        The simulation results of the given initial guesses, has the same shape as the 'obs' parameter
    m : list of floats
        1st moment estimates.
    v : list of floats
        2nd moment estimates.
        
    Performs a step of a corrector algorithm that simulates an orbit from given initial conditions and DM distribution. 
    The result is a new tentative x0 and DM0 that should result in a closer observation.
    Note: this assumes that the dark matter distance parameters are already set in ta.

    """
    
    
    N = len(DMm0)
    
    #If observation consists of only 3 measured parameters, this is assumed to be
    # the first, second and last parameter of the presumed 6 parameters.
    # In the cartesian observation case, this is X, Y and VZ.
    OBS3 = False
    if len(obs[0]) == 3:
        OBS3 = True
    
    
    #Reset the state
    ta.state[:] = np.append(x0,np.array(variationalEqsInitialConditions(N)))
    ta.pars[:N] = DMm0
    ta.time = 0
    #Simulate ta from initial guess (t=0) until t_obs
    out = ta.propagate_grid(t_obs)
    
    orbparamvalues = np.asarray(out[4][:,[0,1,2,3,4,5]]).copy()
    
    simulatedlist = orbparamvalues.copy()
    
    
    if CARTESIANOBS:
        simulatedlist = convertToCartesian(simulatedlist[:,0], simulatedlist[:,1], simulatedlist[:,2],\
                simulatedlist[:,3], simulatedlist[:,4], simulatedlist[:,5])
        
        if OBS3:
            simulatedlist = np.array(simulatedlist)
            simulatedlist =  simulatedlist[[0,1,-1],:].transpose()
    
    
    #Take difference of observation with simulation from initial guess
    difference = np.subtract(simulatedlist, obs)
    
    Phi = ta.state[6:6+36].reshape((6,6))
    
    Psi = ta.state[6+36:].reshape((6,N))
    
    
    gradx0 = np.zeros((1,6))
    gradDM0 = np.zeros((1,N))
    
    varlist = ["p", "e", "i", "om", "w", "f"]
    derobsdx = cartesianConversionGradient()
    #Iterate over observations:
    for oj in range(len(obs)):
    #TODO: add stochasticity (random sample instead of full list)
        if CARTESIANOBS:
            #Need to multiply by gradient of observed cartesian vs orbital parameters
            valuelist = orbparamvalues[oj]
            peixyzDict = dict(zip(varlist, valuelist))
            
            dobsdx = []
            for i in range(6):
                for j in range(6):
                    dobsdx.append(hy.eval(derobsdx[i*6+j],peixyzDict))
            dobsdx = np.array(dobsdx).reshape((6,6))
            
        #TODO: optimize this
        if OBS3:
            dobsdx = np.delete(dobsdx, (2,3,4), axis=0)
        
        #Calculate gradient wrt initial conditions (phi)
        if CARTESIANOBS:
            gradx0 = gradx0 + (2 * difference[oj] @ dobsdx @ Phi ).reshape(1,-1)[0]
            gradDM0 = gradDM0 + (2 * difference[oj] @ dobsdx @ Psi ).reshape(1,-1)[0]
        else:
            gradx0 = gradx0 + (2 * difference[oj] @ Phi ).reshape(1,-1)[0]
            gradDM0 = gradDM0 + (2 * difference[oj] @ Psi ).reshape(1,-1)[0]
    
    
    #Plot of gradient matrices:
    # print(dobsdx)
    # plt.matshow(dobsdx,cmap='RdYlGn')
    # ax = plt.gca()
    # plt.colorbar()
    # plt.clim(-1,1)
    # ax.xaxis.set_ticks_position('bottom')
    # plt.xticks(range(6),['dp','de','di','dOm','dw','df'])
    # plt.yticks(range(6),['dx','dy','dz','dvx','dvy','dvz'])
    # plt.xticks(range(6),['dp0','de0','di0','dOm0','dw0','df0'])
    # for (i, j), z in np.ndenumerate(dobsdx):
    #     if z == 0:
    #         plt.text(j, i, '0', ha='center', va='center')
    #     else:
    #         plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
    # plt.title("Gradient of cartesian over orbital elements")
    # plt.show()
    
    #Adam optimizer:
    grad = np.append(gradx0,gradDM0)
    
    m = beta1 * m + (1.0 - beta1) * grad
    v = beta2 * v + (1.0 - beta2) * grad**2
    mhat = m / (1.0 - beta1**(t+1))
    vhat = v / (1.0 - beta2**(t+1))
    xDM_new = np.append(x0,DMm0) - alpha * mhat / (np.sqrt(vhat) + eps)
    
    x_new = xDM_new[:6]
    DM_new = xDM_new[6:]
    
    
    #Basic gradient descent:
    # delta = - alpha * grad
    # delta = delta.reshape(1,-1)[0]
    # x_new = x0+delta[:6]
    # DM_new = DMm0 + delta[6:]
    
    
    #Additional constraints:
    #DM mass can become negative, don't allow this
    for i in range(len(DM_new)):
        if DM_new[i] < 0:
            DM_new[i] = 0
    
     
    return ta, x_new, DM_new, simulatedlist, m, v
    
    