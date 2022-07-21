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
    