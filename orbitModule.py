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


def buildTaylorIntegrator(PNCORR,N,LOAD_PICKLE=False,verbose=False):
    """
    

    Parameters
    ----------
    PNCORR : boolean
        True if using the 1PN correction, false if using Kepler mechanics.
    N : int
        Amount of dark matter shells.
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
        k = 0.01
        
        
        
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
        
        
        """
        Instantiate the Taylor integrator
        """
        
        start_time = time.time()
        ta = hy.taylor_adaptive(
            # The ODEs.
            [(p, dpdt), (e, dedt), (i, didt), (om, domdt), (w, dwdt), (f, dfdt)],
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
    

