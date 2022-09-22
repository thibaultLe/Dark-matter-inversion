# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:11:01 2022

@author: Thibault
"""


import heyoka as hy
import pygmo as pg
import numpy as np
import time
import pickle
from matplotlib.pylab import plt

from functools import lru_cache



def getBaseUnitConversions():
    """
    Returns the base unit conversions, as we are using scaled units

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


def getObservationTimes(nbrOfOrbits=1):
    """
    Returns observation times in years
    
    Parameters
    ----------
    nbrOfOrbits : float, optional.
        Amount of orbits to observe. The default is 1.
    
    Returns 
    -------
    obstimes : np.array
        List of observation times.

    """
    M_0, D_0, T_0 = getBaseUnitConversions()
    
    #Time of apocentre + 1 full orbit
    obstimes = 2.010356112597776246e+03 + np.linspace(0,16.056740695411154*nbrOfOrbits,300*nbrOfOrbits)
    
    return obstimes

def convertYearsTimegridToOurFormat(timegrid):
    """
    Converts a given timegrid in years format to a format compatible with our base units
    Sets the first observation time to t=0

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
    #Multiply by (number of seconds in a year) / number of seconds in T_0
    timegrid = timegrid * 365.25 * 24 * 60**2 / T_0
    
    return timegrid

def convertXYVZtoArcsec(rx,ry,vz):
    """
    Converts a format of [AU, AU, m/s] to [arcsec, arcsec, km/s]

    Parameters
    ----------
    rx : list of floats
        x values.
    ry : list of floats
        y values.
    vz : list of floats
        vz values.

    Returns
    -------
    list of floats
        x in arcseconds.
    list of floats
        y in arcseconds.
    list of floats
        vz in km/s.

    """
    return AU_to_arcseconds(rx), AU_to_arcseconds(ry), vz/1000

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
    # T_period = np.sqrt(a_mpe**3)*2*np.pi # = 16.056740695411154 years
    #Initial time
    # T_0mpe =  2010.3561125977762 * 365.25 * 24 * 60**2 /T_0
    
    #Initial conditions:
    IC= [p_mpe, e_mpe, i_mpe / 180 * np.pi, \
         om_mpe / 180 * np.pi, w_mpe / 180 * np.pi, f_mpe]
        
    return IC

def get_DM_distances(N,xlim):
    """
    Returns the dark matter distances.

    Parameters
    ----------
    N : int
        Number of mascon shells.
    xlim : float
        Maximum distance of shells from 0.

    Returns
    -------
    ris : list of floats
        Dark matter distances.

    """
    _, ris = get_Plummer_DM(N, xlim)
    return ris

def get_ReversedPlummer_DM(N,xlim):
    """
    Returns a reversed Plummer dark matter model, discretised in mascons.
    
    Parameters
    ----------
    N : int
        Number of mascon shells.
    xlim : float
        Maximum distance of shells from 0.
    
    Returns
    -------
    mis : list of floats
        Dark matter masses.
    ris : list of floats
        Dark matter distances.
    
    """
    mis, ris = get_Plummer_DM(N, xlim)
    
    mis = list(reversed(mis))
    
    return mis, ris

def get_ConstantDensity_DM(N,xlim):
    """
    Returns a constant density dark matter model, discretised in mascons.
    
    Parameters
    ----------
    N : int
        Number of mascon shells.
    xlim : float
        Maximum distance of shells from 0.
    
    Returns
    -------
    mis : list of floats
        Dark matter masses.
    ris : list of floats
        Dark matter distances.
    
    """
    ris = get_DM_distances(N, xlim)
    
    vols = 4*np.pi*(np.array(ris)**3)/3
    for i in range(1,len(vols)):
        vols[i] = vols[i] - vols[i-1]
    
    
    dens = np.array(N*[2e-13])
    
    mis = dens*vols
    
    return mis, ris
    

def get_Uniform_DM(N,xlim):
    """
    Returns a uniform dark matter model, discretised in mascons.
    The average value of the Plummer model is used.

    Parameters
    ----------
    N : int
        Number of mascon shells.
    xlim : float
        Maximum distance of shells from 0.

    Returns
    -------
    mis : list of floats
        Dark matter masses.
    ris : list of floats
        Dark matter distances.

    """
    mis, ris = get_Plummer_DM(N, xlim)
    
    return np.repeat(np.mean(mis),N), ris

def get_Sinusoidal_DM(N,xlim):
    """
    Returns a sinusoidal dark matter model, discretised in mascons.
    The average value of the Plummer model is used.

    Parameters
    ----------
    N : int
        Number of mascon shells.
    xlim : float
        Maximum distance of shells from 0.

    Returns
    -------
    mis : list of floats
        Dark matter masses.
    ris : list of floats
        Dark matter distances.

    """
    mis, ris = get_Plummer_DM(N, xlim)
    avg = np.repeat(np.mean(mis),N)
    mis = (1e-3/N)*np.sin(np.array(ris)/350 + 250) + avg*4/5
    
    return mis, ris

def get_Plummer_DM(N,xlim,rho0=1.69*10**(-10),r0=2474.01):
    """
    Returns the Plummer Dark matter model, discretised in mascons

    Parameters
    ----------
    N : int
        Number of mascon shells.
    xlim : float
        Maximum distance of shells from 0.
    rho0 : float
        Density parameter, optional.
    r0 : float
        Scale parameter, optional.

    Returns
    -------
    mis : list of floats
        Dark matter masses.
    ris : list of floats
        Dark matter distances.

    """
    
    M_0, D_0, T_0 = getBaseUnitConversions()
    rho0plum = rho0 * (D_0**3) / M_0
    
    return get_PlummerOrBahcall_DM(N, xlim, rho0plum, r0, True)

def get_BahcallWolf_DM(N,xlim,rho0=2.24*10**(-11),r0=2474.01):
    """
    Returns the BahcallWolf-cusp Dark matter model, discretised in mascons

    Parameters
    ----------
    N : int
        Number of mascon shells.
    xlim : float
        Maximum distance of shells from 0.
    rho0 : float
        Density parameter, optional.
    r0 : float
        Scale parameter, optional.

    Returns
    -------
    mis : list of floats
        Dark matter masses.
    ris : list of floats
        Dark matter distances.

    """
    
    M_0, D_0, T_0 = getBaseUnitConversions()
    rho0cusp =  rho0 * (D_0**3) / M_0
    
    return get_PlummerOrBahcall_DM(N, xlim, rho0cusp,r0, False)


def get_PlummerOrBahcall_DM(N,xlim,rho0,r0,PLUM=True):
    """
    Returns the Plummer or BahcallWolf Dark matter model, discretised in mascons

    Parameters
    ----------
    N : int
        Number of mascon shells.
    xlim : float
        Maximum distance of shells from 0.
    rho0 : float
        Density parameter
    r0 : float
        Scale parameter
    PLUM : boolean
        True if using the plummer model, false if using bahcall-wolf

    Returns
    -------
    mis : list of floats
        Dark matter masses.
    ris : list of floats
        Dark matter distances.

    """
    
    def enclosedMassPlum(a,rho0,r0):
        return (4 * a**3 * np.pi * r0**3 * rho0) / ( 3 * (a**2 + r0**2)**(3/2))
    
    def enclosedMassCusp(a,rho0,r0):
        return (4 * a**3 * np.pi * (a/r0)**(-7/4) * rho0) / (3 - (7/4))
    
    M_0, D_0, T_0 = getBaseUnitConversions()
    
    x_mid = np.linspace(0,xlim,2*(N+1)+1) # Midpoints
    ris = x_mid[1::2]
    
    if PLUM:
        y_mid = np.append(0,enclosedMassPlum(ris,rho0,r0))
    else:
        y_mid = np.append(0,enclosedMassCusp(ris,rho0,r0))
    
    mis = [t - s for s, t in zip(y_mid, y_mid[1:])]
    
    #Convert to a 'middle riemann sum'
    mis = mis[1:]
    newris = []
    for i in range(len(ris)-1):
        newris.append((ris[i]+ris[i+1])/2)
    
    
    #Divides [0,xlim] in N+1 parts and takes the borders
    #   e.g. [0,3000] -> ris = [750,1500,2250]
    
    return mis, newris

def AU_to_arcseconds(dist):
    """
    Converts AU to arcseconds

    Parameters
    ----------
    dist : float or np.array
        Distance in [AU] to be converted.

    Returns
    -------
    float or np.array
        The converted distance in arcseconds.

    """
    _, D_0, _ = getBaseUnitConversions()
    R = 2.5540153e+20
    return 2 * np.arctan(dist*D_0/(2*R)) * 206264.8

def arcseconds_to_AU(arcsec):
    """
    Converts arcseconds to AU

    Parameters
    ----------
    arcsec : float or np.array
        Distance in [arsec] to be converted.

    Returns
    -------
    float or np.array
        The converted distance in AU.

    """
    _, D_0, _ = getBaseUnitConversions()
    R = 2.5540153e+20
    
    return 2 * R * np.tan(arcsec/(2*206264.8)) / D_0  



def convertToCartesian(p,e,i,om,w,f):
    """
    Converts orbital parameters to cartesian coordinates/velocities

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
    GM = 1
    vx = -np.sqrt(GM/p) * (np.cos(om) * (np.sin(w+f) + e*np.sin(w)) + \
             np.cos(i) * np.sin(om) * (np.cos(w+f) + e*np.cos(w)))
    vy = -np.sqrt(GM/p) * (np.sin(om) * (np.sin(w+f) + e*np.sin(w)) - \
             np.cos(i) * np.cos(om) * (np.cos(w+f) + e*np.cos(w)))
    vz = np.sqrt(GM/p) * np.sin(i) * (np.cos(w+f) + e * np.cos(w))
    
    return rx,ry,rz,vx,vy,vz


@lru_cache(maxsize=1) #Cached for quick retrieval in the gradient descent
def cartesianConversionGradient(): 
    """
    Returns the gradient of cartesian coordinates/velocities over orbital parameters
    
    Returns
    -------
    derobsdx : list of hy.expression
        List of derivatives of cartesian observation wrt orbital parameters.

    """
    
    # Create the symbolic variables.
    p, e, i, om, w, f = hy.make_vars("p", "e", "i", "om", "w", "f")
    x    = np.array([p,e,i,om,w,f])
    
    GM = 1
    
    """
    Cartesian position and velocity conversion
    """
    r = p / (1 + e * hy.cos(f))
    
    rx = r * (hy.cos(om) * hy.cos(w + f) - hy.cos(i)*hy.sin(om)*hy.sin(w+f))
    ry = r * (hy.sin(om) * hy.cos(w + f) + hy.cos(i)*hy.cos(om)*hy.sin(w+f))
    rz = r * hy.sin(i) * hy.sin(w + f)
    
    vx = -hy.sqrt(GM/p) * (hy.cos(om) * (hy.sin(w+f) + e * hy.sin(w)) + \
              hy.cos(i) * hy.sin(om) * (hy.cos(w+f) + e * hy.cos(w)))
    vy = -hy.sqrt(GM/p) * (hy.sin(om) * (hy.sin(w+f) + e * hy.sin(w)) - \
              hy.cos(i) * hy.cos(om) * (hy.cos(w+f) + e * hy.cos(w)))
    vz = hy.sqrt(GM/p) * hy.sin(i) * (hy.cos(w+f) + e * hy.cos(w))

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
    Returns the initial conditions for the variational equations.

    Parameters
    ----------
    N : int
        Number of dark matter shells.

    Returns
    -------
    list of ints
        The initial conditions for the variational equations.

    """
    ic_var_phi = np.eye(6).reshape((36,)).tolist()
    ic_var_psi = np.zeros((6*N,)).tolist()
    
    return ic_var_phi + ic_var_psi


def buildTaylorIntegrator(PNCORR,N,include_variational_eqs=True,compact_mode = False,LOAD_PICKLE=False,SAVE_PICKLE=False,verbose=True):
    """
    Returns the heyoka taylor integrator with the given parameters
    
    Parameters
    ----------
    PNCORR : boolean
        True if using the 1PN correction, false if using Kepler mechanics.
    N : int
        Number of dark matter shells.
    include_variational_eqs : bool, optional
        True if you want to include variational equations (e.g. to reconstruct IC's and DM distribution). Slows the process down a bit. The default is True.
    compact_mode : bool, optional
        True if you to enable compact mode (faster setup, slower integration). The default is False.
    LOAD_PICKLE : bool, optional
        True if you want to load a previously saved version of ta. The default is False.
    verbose : bool, optional
        True if you want to print information on the process. The default is False.

    Returns
    -------
    ta : heyoka.core._taylor_adaptive_dbl
        The heyoka taylor integrator.
    
    """
    
    if LOAD_PICKLE and SAVE_PICKLE:
        raise RuntimeWarning('The functionalities of loading and saving overlap. Only enable one.')
    
    
    if LOAD_PICKLE:
        start_time = time.time()
        ta_file = open("ta_saved_N={}".format(N),'rb')
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
        
        M_0, D_0, T_0 = getBaseUnitConversions()
        
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
        
        #Initial conditions:
        IC = get_S2_IC()
        
        
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
            compact_mode = compact_mode
        )
        if verbose:
            print("--- %s seconds --- to build the Taylor integrator" % (time.time() - start_time))
        
        ## Pickle save
        if SAVE_PICKLE:
            ta_file = open("ta_saved_N={}".format(N),"wb")
            pickle.dump(ta,ta_file)
            ta_file.close()
        
        return ta
    


def simulateOrbits(PNCORRECTION,IC,mis,ris,t_grid):
    """
    Simulates orbits on a given time grid from given initial conditions and dark matter

    Parameters
    ----------
    PNCORRECTION : boolean
        True if using the 1PN correction, false if using Kepler mechanics.
    IC : list of floats
        Initial conditions of orbital parameters at the first observation point.
    mis : list of floats
        Dark matter shell masses in [MBH masses].
    ris : list of floats
        Dark matter shell distances from 0 in [AU].
    t_grid : list of floats
        Observation times in years. 

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
    
    t_grid = convertYearsTimegridToOurFormat(t_grid)
    
    ta = buildTaylorIntegrator(PNCORRECTION, N, include_variational_eqs=False,compact_mode=True,LOAD_PICKLE=False)
    
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
    Simulates orbits on a given time grid from given initial conditions and dark matter
    Returns cartesian coordinates/velocities
    
    Parameters
    ----------
    PNCORRECTION : boolean
        True if using the 1PN correction, false if using Kepler mechanics.
    IC : list of floats
        Initial conditions of orbital parameters at the first observation.
    mis : list of floats
        Dark matter shell masses in [MBH masses].
    ris : list of floats
        Dark matter shell distances from 0 in [AU].
    t_grid : list of floats
        Observation times in years.

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
    # t_grid = convertYearsTimegridToOurFormat(t_grid)
    p,e,i,om,w,f = simulateOrbits(PNCORRECTION, IC, mis, ris, t_grid)
    rx,ry,rz,vx,vy,vz = convertToCartesian(p, e, i, om, w, f)
    M_0, D_0, T_0 = getBaseUnitConversions()
    
    #Returns  observations (AU, meters/second)
    return rx,ry,rz, vx * D_0 / T_0, vy * D_0 / T_0, vz * D_0 / T_0


def corrector(ta, x0, DMm0, obs, t_obs, alpha, m, v, t,CARTESIANOBS=True,optimizer='ADAM'):
    """
    Performs 1 iteration of a gradient descent algorithm that simulates an orbit from given initial conditions and DM distribution. 
    The result is a new tentative x0 and DM0 that should result in a closer observation.
    Note: this assumes that the dark matter distance parameters are already set in ta.

    Parameters
    ----------
    ta : hy.taylor_adaptive
        System of equations.
    x0 : list of floats
        initial condition guess (p,e,i,om,w,f)
    DMm0 : list of floats
        initial dark matter mass guess (m1,m2,...,mN).
    obs : list of list of floats
        observation at time tj of (p,e,i,om,w,f) or (x,y,z,vx,vy,vz)
    t_obs : list of floats
        observation times tj
    alpha : float
        Learning rate for Adam.
    m : list of floats
        1st moment estimates.
    v : list of floats
        2nd moment estimates.
    t : int
        Iteration step number.
    CARTESIANOBS : boolean, optional
        True if using cartesian observations instead of orbital parameters. The default is True.
    optimizer : string, optional
        Optimizer with which to do the gradient descent. The default is Adam.

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
    
    orbparamvalues = np.asarray(out[4][:,[0,1,2,3,4,5]])
    
    Phis = np.asarray(out[4][:,6:6+36])
    
    Psis = np.asarray(out[4][:,6+36:])
    
    simulatedlist = orbparamvalues.copy()
    
    if CARTESIANOBS:
        simulatedlist = convertToCartesian(simulatedlist[:,0], simulatedlist[:,1], simulatedlist[:,2],\
                simulatedlist[:,3], simulatedlist[:,4], simulatedlist[:,5])
        
        if OBS3:
            simulatedlist = np.array(simulatedlist)
            simulatedlist =  simulatedlist[[0,1,-1],:]
    
        simulatedlist = np.transpose(simulatedlist)
    
    #Take difference of observation with simulation from initial guess
    difference = np.subtract(simulatedlist, obs)
    
    gradx0 = np.zeros((1,6))
    gradDM0 = np.zeros((1,N))
    
    
    
    varlist = ["p", "e", "i", "om", "w", "f"]
    derobsdx = cartesianConversionGradient()
    #Iterate over observations:
    for oj in range(len(obs)):
        
        Phi = Phis[oj].reshape((6,6))
        Psi = Psis[oj].reshape((6,N))
        
        if CARTESIANOBS:
            #Need to multiply by gradient of observed cartesian vs orbital parameters
            valuelist = orbparamvalues[oj]
            peixyzDict = dict(zip(varlist, valuelist))
            
            dobsdx = []
            if OBS3:
                #OBS3 -> take x,y and vz instead of all 6
                for i in [0,1,5]:
                    for j in range(6):
                        dobsdx.append(hy.eval(derobsdx[i*6+j],peixyzDict))
                dobsdx = np.array(dobsdx).reshape((3,6))
            else:
                for i in range(6):
                    for j in range(6):
                        dobsdx.append(hy.eval(derobsdx[i*6+j],peixyzDict))
                dobsdx = np.array(dobsdx).reshape((6,6))
            
        
        #Calculate gradient wrt initial conditions (phi) and dark matter masses (psi)
        if CARTESIANOBS:
            gradx0 = gradx0 + (2 * difference[oj] @ dobsdx @ Phi ).reshape(1,-1)[0]
            gradDM0 = gradDM0 + (2 * difference[oj] @ dobsdx @ Psi ).reshape(1,-1)[0]
            
        else:
            gradx0 = gradx0 + (2 * difference[oj] @ Phi ).reshape(1,-1)[0]
            gradDM0 = gradDM0 + (2 * difference[oj] @ Psi ).reshape(1,-1)[0]
    
    
    #Plot of gradient matrices:
    # print(dobsdx)
    # if t == 50 or t == 299:
    #     # plt.matshow(dobsdx,cmap='RdYlGn')
    #     plt.matshow(Psi,cmap='RdYlGn')
    #     ax = plt.gca()
    #     plt.colorbar()
    #     plt.clim(-1,1)
    #     ax.xaxis.set_ticks_position('bottom')
    #     # plt.xticks(range(6),['dp','de','di','dOm','dw','df'])
    #     plt.yticks(range(6),['dp','de','di','dOm','dw','df'])
        
    #     plt.xticks(range(20),['dm{}'.format(i+1) for i in range(20)])
    #     # plt.yticks(range(6),['dx','dy','dz','dvx','dvy','dvz'])
    #     # plt.xticks(range(6),['dp0','de0','di0','dOm0','dw0','df0'])
    #     for (i, j), z in np.ndenumerate(Psi):
    #         if z == 0:
    #             plt.text(j, i, '0', ha='center', va='center')
    #         else:
    #             plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
    #     # plt.title("Gradient of cartesian over orbital elements")
    #     plt.title("Gradient of orbital elements over mascon masses")
    #     plt.show()
        
    #     # plt.matshow(dobsdx,cmap='RdYlGn')
    #     plt.matshow(dobsdx @ Psi,cmap='RdYlGn')
    #     ax = plt.gca()
    #     plt.colorbar()
    #     plt.clim(-1,1)
    #     ax.xaxis.set_ticks_position('bottom')
    #     # plt.xticks(range(6),['dp','de','di','dOm','dw','df'])
    #     plt.xticks(range(20),['dm{}'.format(i+1) for i in range(20)])
    #     plt.yticks(range(3),['dx','dy','dvz'])
    #     # plt.xticks(range(6),['dp0','de0','di0','dOm0','dw0','df0'])
    #     for (i, j), z in np.ndenumerate(dobsdx @ Psi):
    #         if z == 0:
    #             plt.text(j, i, '0', ha='center', va='center')
    #         else:
    #             plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
    #     # plt.title("Gradient of cartesian over orbital elements")
    #     plt.title("Gradient of cartesian over mascon masses")
    #     plt.show()
        
        # print(gradDM0)
        
    grad = np.append(gradx0,gradDM0)
    
    # if t >= 199:
    #     optimizer = 'LINE'
    
    # if t % 400 == 0:
    #     plt.figure()
    #     mis, ris = get_BahcallWolf_DM(N, xlim=2100)
    #     # plt.scatter(ris,DMm0,label='Mascon shells',alpha=1 - (t/100),color='blue')
    #     plt.scatter(ris,mis,label='True')
    #     plt.scatter(ris,DMm0,label='Reconstructed',color='orange')
    #     # if CARTESIANOBS:
    #     #     normalizedGrad = -grad[6:] * alpha * 0.001
    #     # else:
    #     #     normalizedGrad = -grad[6:] * 0.01
            
    #     # if CARTESIANOBS:
    #     #     normalizedGrad = -gradDM0[0]**2 * alpha * 0.001
    #     # else:
    #     #     normalizedGrad = -gradDM0[0]**2 * 0.01
    #     print(gradplot)
    #     if CARTESIANOBS:
    #         normalizedGrad = -gradplot[0]**2
    #     else:
    #         normalizedGrad = -gradplot[0]**2 * 0.01
        
    #     for i in range(N):
    #         # if i == 0:
    #         #     plt.arrow(ris[i],DMm0[i],0,normalizedGrad[i],head_width=0.8, head_length=max(abs(normalizedGrad))/10,label='-Gradient')
    #         # else:
    #         #     plt.arrow(ris[i],DMm0[i],0,normalizedGrad[i],head_width=0.8, head_length=max(abs(normalizedGrad))/10)
    #         if i == 0:
    #             plt.errorbar(ris[i],DMm0[i],1/normalizedGrad[i],capsize=5,color='orange',label='1/Gradient²')
    #         else:
    #             plt.errorbar(ris[i],DMm0[i],1/normalizedGrad[i],capsize=5,color='orange')
    
        
    #     # plt.ylim(0,max(1.2*max(DMm0)))
    #     plt.ylim(0,0.00014)
    #     plt.legend()
        
        
    
        
    if optimizer == 'ADAM':
        #Adam optimizer:
            
        # factor for average gradient
        beta1 = 0.9
        # factor for average squared gradient
        beta2 = 0.999
        #Precision
        eps = 1e-8
        
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad**2
        mhat = m / (1.0 - beta1**(t+1))
        vhat = v / (1.0 - beta2**(t+1))
        xDM_new = np.append(x0,DMm0) - alpha * mhat / (np.sqrt(vhat) + eps)
        
        x_new = xDM_new[:6]
        DM_new = xDM_new[6:]
    
    
    elif optimizer == 'LINE':
        #Linesearch:
            
        alphas = np.append(0,np.logspace(-15,-5,20))
        if True:
            # print(alphas)
            mindiff = 100000
            # chosenindex = 0
            diffs = []
            fulldiffs = []
            chosenX = x0
            chosenDM = DMm0
            for i in range(len(alphas)):
                # print(i)
                delta = - alphas[i] * grad
                delta = delta.reshape(1,-1)[0]
                x_new = x0+delta[:6]
                # x_new = x0
                DM_new = DMm0 + delta[6:]
                
                #Reset the state
                ta.state[:] = np.append(x_new,np.array(variationalEqsInitialConditions(N)))
                ta.pars[:N] = DM_new
                ta.time = 0
                #Simulate ta from initial guess (t=0) until t_obs
                out = ta.propagate_grid(t_obs)
                
                orbparamvalues = np.asarray(out[4][:,[0,1,2,3,4,5]])
                
                simulatedlistL = orbparamvalues.copy()
                
                if CARTESIANOBS:
                    simulatedlistL = convertToCartesian(simulatedlistL[:,0], simulatedlistL[:,1], simulatedlistL[:,2],\
                            simulatedlistL[:,3], simulatedlistL[:,4], simulatedlistL[:,5])
                    
                    if OBS3:
                        simulatedlistL = np.array(simulatedlistL)
                        simulatedlistL =  simulatedlistL[[0,1,-1],:]
                
                    simulatedlistL = np.transpose(simulatedlistL)
                
                #Take difference of observation with simulation from initial guess
                difference = np.subtract(simulatedlistL, obs)
                totdiff = np.sum(abs(difference))
                diffs.append(totdiff)
                fulldiffs.append(difference)
                
                if totdiff < mindiff:
                    mindiff = totdiff
                    # chosenindex = i
                    chosenX = x0+delta[:6]
                    chosenDM = DMm0 + delta[6:]
                    
            # if t == 0:
            difference = np.subtract(simulatedlist, obs)
            # print('Orig diff:',difference)
            totdiff = np.sum(abs(difference))
            # print('Loss:',totdiff)
            # print('New diff',fulldiffs[chosenindex])
            # print('New loss:',mindiff)
            # print('Chosen alpha:', alphas[chosenindex])
            # print('Chosen DM:',chosenDM)
            plt.figure()
            plt.hlines(totdiff, alphas[0], alphas[-1],label='Previous loss',color='orange')
            plt.plot(alphas,diffs,label='New loss')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Learning rate')
            plt.ylabel('Loss')
            plt.legend()
            
            x_new = chosenX
            DM_new = chosenDM
            
    else: #optimizer == 'GRAD'
        # Basic gradient descent:
        delta = - alpha * grad
        delta = delta.reshape(1,-1)[0]
        x_new = x0 + delta[:6]
        DM_new = DMm0 + delta[6:]
    
    
    #Additional constraints:
    #DM mass can become negative, don't allow this
    for i in range(len(DM_new)):
        if DM_new[i] < 0:
            DM_new[i] = 0
    
    
    #Monotonous density constraint:
    # ris = np.array(get_DM_distances(N, 2100))
    
    # vols = 4*np.pi*(ris**3)/3
    # for i in range(1,len(vols)):
    #     vols[i] = vols[i] - vols[i-1]
    
    # for i in range(1,len(DM_new)):
    #     constr = vols[i]*DM_new[i-1]/vols[i-1]
    #     if DM_new[i] > constr:
    #         DM_new[i] = constr
    
    # dens = DM_new/vols
    
    #TODO: Don't allow initial conditions to go past confidence interval of measurements
    
     
    return ta, x_new, DM_new, m, v



def getGoodnessOfFit(ta, ic_guess, dm_guess, obslist,noisefactor=0):
    timegrid = obslist[:,0].copy()
    
    observationlist = np.delete(obslist, [0], axis=1)    

    t_grid = convertYearsTimegridToOurFormat(timegrid)
    
    N = len(dm_guess)
    NF = noisefactor
    
    Nfree = 3 * len(t_grid) - (N + 6)
    
    if Nfree < 1:
        raise RuntimeWarning("Trying to fit {} observations with {} parameters".format(3*len(t_grid),N+6))
    
    M_0, D_0, T_0 = getBaseUnitConversions()
    if NF != 0:
        sigmaPosition = 50 * NF # AU
        sigmaVelocity = 10 * NF # km/s -> m/s -> our velocity units
        weights = [1/(sigmaPosition**2), 1/(sigmaPosition**2), 1/(sigmaVelocity**2)]
    else:
        weights = [1,1,1]
        
    #1 last simulation of the final guess:
    ta.state[:] = np.append(ic_guess,np.array(variationalEqsInitialConditions(N)))
    ta.time = 0
    ta.pars[:N] = dm_guess
    out = ta.propagate_grid(t_grid)
    finalsim = np.asarray(out[4][:,[0,1,2,3,4,5]])
    finalsim = convertToCartesian(finalsim[:,0], finalsim[:,1], finalsim[:,2],\
            finalsim[:,3], finalsim[:,4], finalsim[:,5])
    finalsim = np.array(finalsim)
    finalsim =  finalsim[[0,1,-1],:]
    finalsim = np.transpose(finalsim)
    
    #Diffs: [AU] [AU] [AU/T_0]
    difxs = 1e6*(AU_to_arcseconds(finalsim[:,0])-AU_to_arcseconds(observationlist[:,0]))
    difys = 1e6*(AU_to_arcseconds(finalsim[:,1])-AU_to_arcseconds(observationlist[:,1]))
    difvzs = finalsim[:,-1]* D_0 / (T_0 * 1000)-observationlist[:,-1]* D_0 / (T_0 * 1000)
    gfit = 0
    for i in range(len(difxs)):
        gfit += weights[0]*(difxs[i]**2)
        gfit += weights[1]*(difys[i]**2)
        gfit += weights[2]*(difvzs[i]**2)
        
    # print('Goodness of fit =', gfit/Nfree)
    
    # return gfit/Nfree
    return gfit



def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    
    Prints a progress bar in the console

    Parameters
    ----------
    iteration : int
        current iteration.
    total : int
        total iterations.
    prefix : string, optional
        prefix string. The default is ''.
    suffix : string, optional
        suffix string. The default is ''.
    decimals : int, optional
        positive number of decimals in percent complete. The default is 1.
    length : int, optional
        character length of bar. The default is 100.
    fill : string, optional
        bar fill character. The default is '█'.
    printEnd : string, optional
        end character (e.g. "\r", "\r\n"). The default is "\r".

    Returns
    -------
    None.

    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def reconstructDistribution(obslist, ic_guess, dm_guess, CARTESIANOBS = True,OBS3 = True,noisefactor=0):
    """
    Reconstructs initial conditions and dark matter from given initial guesses and observations

    Parameters
    ----------
    obslist : list of list of floats
        the observations [[t0, y0, x0, vz0], ...]
    ic_guess : list of floats
        the initial guess for the initial conditions
    dm_guess : list of floats
        the initial guess for the dark matter masses
    CARTESIANOBS : boolean, optional
        True if using cartesian observations instead of orbital parameter observations. The default is True.
    OBS3 : boolean, optional
        True if only using 3 observed parameters (first, second and last = x,y and vz for cartesian). The default is True.
    noisefactor : float, optional
        Factor to reduce/increase the noise below/above the standard. The default is 0.

    Raises
    ------
    RuntimeError
        If the length of dark matter masses and distances do not match, an error is raised.

    Returns
    -------
    list of floats
        list of reconstructed dark matter masses.

    Reconstructs dark matter distribution starting from an initial guess
    """
    
    N = len(dm_guess)
    
    M_0, D_0, T_0 = getBaseUnitConversions()
    
    timegrid = obslist[:,0]
    
    observationlist = np.delete(obslist, [0], axis=1)    

    t_grid = convertYearsTimegridToOurFormat(timegrid)

    
    # initialize first and second moments for Adam
    m = np.array([0.0 for _ in range(6+N)])
    v = np.array([0.0 for _ in range(6+N)])
    
    #Choose your favorite optimizer:
    optimizer = 'ADAM'
    # optimizer = 'LINE'
    # optimizer = 'GRAD'
    
    #If batch size is < 1, all observations are used each iteration
    batch_size = 0
    
    BATCHED = False
    if batch_size > 0:
        BATCHED = True
    
    # step size 1e-5 for Adam, 1e-10 for Grad descent
    alpha = 1e-5
    
    #batch 1, 1e-6, 4000, Adam, works great
    #all obs, 1e-5, 3000, Adam, gets very close to 0, but 500 iters gets close enough
    iterations = 500
    #TODO: Decide on stop criterion
    #TODO: add a 'verbose'/'plotting' parameter
    
    ICiterations = np.array([ic_guess])
    DMiterations = np.array([dm_guess])
    # for i in range(len(iters)):
    #     losses.append(loss(ta,ICiterations[i],DMiterations[i],obslist))]
    
    # print(N,dm_guess)
    # ta = buildTaylorIntegrator(True, N,compact_mode=True,LOAD_PICKLE=False)
    ta = buildTaylorIntegrator(True, N,LOAD_PICKLE=True)
    #Set DM distances
    ta.pars[N:] = get_DM_distances(N, xlim=2100)
    
    # Number of mascons is ~linear with runtime
    
    losses = [getGoodnessOfFit(ta,ic_guess,dm_guess,obslist,noisefactor)]
    
    #Print progress of the iterations
    printProgressBar(0, iterations, prefix = 'Progress:', suffix = 'Complete', length = 50)
    start = time.time()
    
    for t in range(iterations):
        if t != 0 and iterations > 10 and (t+1) % round(iterations/10) == 0: 
            pointsPerSecond = round(t/(time.time()-start),2)
            secsRemaining = round((iterations - t) / (pointsPerSecond))
            minsRemaining = round((iterations - t) / (pointsPerSecond*60),2)
            if minsRemaining > 1:
                printProgressBar(t+1, iterations, prefix = 'Iterating:', \
                                 suffix = 'complete,{} mins remaining, {} iterations/s'.format(minsRemaining,pointsPerSecond), length = 30)
            else:
                printProgressBar(t+1, iterations, prefix = 'Iterating:', \
                                 suffix = 'complete,{} secs remaining, {} iterations/s'.format(secsRemaining,pointsPerSecond), length = 30)
                        
        if BATCHED:
            # Stochastic batched gradient descent:
            chosenbatchindices = np.sort(np.random.choice(observationlist.shape[0],size=batch_size,replace=False))
            
            ta, ic_guess, dm_guess,m,v = corrector(ta, ic_guess,dm_guess, \
                  observationlist[chosenbatchindices,:], t_grid[chosenbatchindices], alpha,m,v,t,CARTESIANOBS,optimizer)
        else:
            #Using all observations every iteration
            ta, ic_guess, dm_guess,m,v = corrector(ta, ic_guess,dm_guess, \
                  observationlist, t_grid, alpha,m,v,t,CARTESIANOBS,optimizer)
            
            
            
        #Don't allow initial conditions to change:
        # ta, _ ,dm_guess,m,v = corrector(ta, ic_guess,dm_guess, \
        #       observationlist, t_grid, alpha,m,v,t,CARTESIANOBS,optimizer)
        
        
        #Don't allow dark matter to change:
        # ta, ic_guess ,_,m,v = corrector(ta, ic_guess,dm_guess, \
        #       observationlist, t_grid, alpha,m,v,t,CARTESIANOBS,optimizer)
            
        ICiterations = np.append(ICiterations,ic_guess)
        DMiterations = np.append(DMiterations,dm_guess)
        
        loss = getGoodnessOfFit(ta,ic_guess,dm_guess,obslist,noisefactor)
        losses.append(loss)
        
        #Stop criterion depends on noisefactor:
        #NF == 0 -> Stop at loss < 1e-5
        # if noisefactor == 0 and loss < 1:
        #     print('Desired tolerance is reached')
        #     break
        
    
    #1 last simulation of the final guess:
    ta.state[:6] = ic_guess
    ta.time = 0
    ta.pars[:N] = dm_guess
    out = ta.propagate_grid(t_grid)
    finalsim = np.asarray(out[4][:,[0,1,2,3,4,5]])
    if CARTESIANOBS:
        finalsim = convertToCartesian(finalsim[:,0], finalsim[:,1], finalsim[:,2],\
                finalsim[:,3], finalsim[:,4], finalsim[:,5])
        if OBS3:
            finalsim = np.array(finalsim)
            finalsim =  finalsim[[0,1,-1],:]
        finalsim = np.transpose(finalsim)
    
    
    #1 initial simulation of the first guess:
    ta.state[:6] = ICiterations[:6]
    ta.time = 0
    ta.pars[:N] = DMiterations[:N]
    out = ta.propagate_grid(t_grid)
    initialsim = np.asarray(out[4][:,[0,1,2,3,4,5]])
    if CARTESIANOBS:
        initialsim = convertToCartesian(initialsim[:,0], initialsim[:,1], initialsim[:,2],\
                initialsim[:,3], initialsim[:,4], initialsim[:,5])
        if OBS3:
            initialsim = np.array(initialsim)
            initialsim =  initialsim[[0,1,-1],:]
        initialsim = np.transpose(initialsim)
    
    #Reshape
    ICiterations = ICiterations.reshape((iterations+1,6))  
    DMiterations = DMiterations.reshape((iterations+1,N))  
    print('Total time:', round((time.time()-start),2),'seconds')
    
    # print("")
    print('First guess for IC:',ICiterations[0])
    print('Reconstructed IC:  ',np.array(ic_guess))
    # print("")
    # print('First guess for DM:',DMiterations[0])
    # print('Reconstructed DM:  ',list(dm_guess))
    # print("")
        
    iters = np.arange(0,iterations+1,1)
    
    
    #Plot convergence of initial conditions:
    # absdiffs = np.sum(abs(np.subtract(ICiterations,(iterations+1)*[IC])),axis=1)
    # plt.figure()
    # plt.scatter(iters,absdiffs,color='blue',s=8)
    # plt.ylabel("Difference")
    # plt.xlabel("Number of iterations")
    # plt.title("Difference with true initial conditions")
    
    
    #Plot convergence of dark matter:
    # absdiffs = np.sum((np.subtract(DMiterations,(iterations+1)*[mis])),axis=1)
    # plt.figure()
    # plt.scatter(iters,absdiffs,color='blue',s=8)
    # plt.ylabel("Difference with true value")
    # plt.xlabel("Number of iterations")
    # plt.title("Difference with true DM distribution")
    
    
    
    print('Loss before training =', losses[0])
    print('Loss after training =', losses[-1])
    
    
    xdifs = 1e6*(AU_to_arcseconds(finalsim[:,0])-AU_to_arcseconds(observationlist[:,0]))
    ydifs = 1e6*(AU_to_arcseconds(finalsim[:,1])-AU_to_arcseconds(observationlist[:,1]))
    vzdifs = finalsim[:,-1]* D_0 / (T_0 * 1000)-observationlist[:,-1]* D_0 / (T_0 * 1000)
    
    print('Max X difference:',max(abs(xdifs)),'[µas]')
    print('Max Y difference:',max(abs(ydifs)),'[µas]')
    print('Max VZ difference:',max(abs(vzdifs)),'[km/s]')
            
        
        
    NF = noisefactor
        
    #Can only plot X,Y and VZ differences for cartesian observations
    if CARTESIANOBS:
        fig, ((ax11,ax12,ax13,ax14)) = plt.subplots(1,4)
        fig.set_size_inches(19,4)
        fig.set_tight_layout(True)
        
        #Plot losses:
        ax11.scatter(iters,losses,color='blue',s=8)
        ax11.set_yscale('log')
        ax11.set_ylabel("Loss")
        ax11.set_xlabel("Number of iterations")
        ax11.set_title("Gradient descent")
        
        # plt.figure()
        ax12.scatter(timegrid,1e6*(AU_to_arcseconds(initialsim[:,0])-AU_to_arcseconds(observationlist[:,0])),color='lightgrey',s=8,label='Initial difference')
        ax12.scatter(timegrid,1e6*(AU_to_arcseconds(finalsim[:,0])-AU_to_arcseconds(observationlist[:,0])),color='blue',s=8,label='Final difference')
        ax12.set_ylabel("Difference with observation [µas]")
        ax12.set_xlabel("Time [years]")
        ax12.set_title("X simulated - X observed")
        if NF != 0:
            ax12.plot(timegrid,len(timegrid)*[NF*50],'--',label='Precision',color='red')
            ax12.plot(timegrid,len(timegrid)*[NF*-50],'--',color='red')
            ax12.set_ylim(8*NF*-50,8*NF*50)
        ax12.legend()
        
        # plt.figure()
        ax13.scatter(timegrid,1e6*(AU_to_arcseconds(initialsim[:,1])-AU_to_arcseconds(observationlist[:,1])),color='lightgrey',s=8,label='Initial difference')
        ax13.scatter(timegrid,1e6*(AU_to_arcseconds(finalsim[:,1])-AU_to_arcseconds(observationlist[:,1])),color='blue',s=8,label='Final difference')
        ax13.set_ylabel("Difference with observation [µas]")
        ax13.set_xlabel("Time [years]")
        ax13.set_title("Y simulated - Y observed")
        if NF != 0:
            ax13.plot(timegrid,len(timegrid)*[NF*50],'--',label='Precision',color='red')
            ax13.plot(timegrid,len(timegrid)*[NF*-50],'--',color='red')
            ax13.set_ylim(8*NF*-50,8*NF*50)
        ax13.legend()
        
        if not OBS3:
            fig, ((ax11,ax12,ax13)) = plt.subplots(1,3)
            fig.set_size_inches(19,4)
            # plt.tight_layout(pad=2,w_pad=6)
            fig.set_tight_layout(True)
            # plt.figure()
            ax11.scatter(timegrid,1e6*(AU_to_arcseconds(initialsim[:,2])-AU_to_arcseconds(observationlist[:,2])),color='lightgrey',s=8,label='Initial difference')
            ax11.scatter(timegrid,1e6*(AU_to_arcseconds(finalsim[:,2])-AU_to_arcseconds(observationlist[:,2])),color='blue',s=8,label='Final difference')
            if NF != 0:
                ax11.plot(timegrid,len(timegrid)*[50],'--',label='Precision',color='red')
                ax11.plot(timegrid,len(timegrid)*[-50],'--',color='red')
            ax11.set_ylabel("Difference with observation [µas]")
            ax11.set_xlabel("Time [years]")
            ax11.set_title("Z simulated - Z observed")
            ax11.legend()
            
            # plt.figure()
            ax12.scatter(timegrid,initialsim[:,3]* D_0 / (T_0 * 1000)-observationlist[:,3]* D_0 / (T_0 * 1000),color='lightgrey',s=8,label='Initial difference')
            ax12.scatter(timegrid,finalsim[:,3]* D_0 / (T_0 * 1000)-observationlist[:,3]* D_0 / (T_0 * 1000),color='blue',s=8,label='Final difference')
            if NF != 0:
                ax12.plot(timegrid,len(timegrid)*[10],'--',label='Precision',color='red')
                ax12.plot(timegrid,len(timegrid)*[-10],'--',color='red')
            ax12.set_ylabel("Difference with observation [km/s]")
            ax12.set_xlabel("Time [years]")
            ax12.set_title("VX simulated - VX observed")
            ax12.legend()
            
            # plt.figure()
            ax13.scatter(timegrid,initialsim[:,4]* D_0 / (T_0 * 1000)-observationlist[:,4]* D_0 / (T_0 * 1000),color='lightgrey',s=8,label='Initial difference')
            ax13.scatter(timegrid,finalsim[:,4]* D_0 / (T_0 * 1000)-observationlist[:,4]* D_0 / (T_0 * 1000),color='blue',s=8,label='Final difference')
            if NF != 0:
                ax13.plot(timegrid,len(timegrid)*[10],'--',label='Precision',color='red')
                ax13.plot(timegrid,len(timegrid)*[-10],'--',color='red')
            ax13.set_ylabel("Difference with observation [km/s]")
            ax13.set_xlabel("Time [years]")
            ax13.set_title("VY simulated - VY observed")
            ax13.legend()
        
        # plt.figure()
        ax14.scatter(timegrid,initialsim[:,-1]* D_0 / (T_0 * 1000)-observationlist[:,-1]* D_0 / (T_0 * 1000),color='lightgrey',s=8,label='Initial difference')
        ax14.scatter(timegrid,finalsim[:,-1]* D_0 / (T_0 * 1000)-observationlist[:,-1]* D_0 / (T_0 * 1000),color='blue',s=8,label='Final difference')
        ax14.set_ylabel("Difference with observation [km/s]")
        ax14.set_xlabel("Time [years]")
        ax14.set_title("VZ simulated - VZ observed")
        if NF != 0:
            ax14.plot(timegrid,len(timegrid)*[NF*10],'--',label='Precision',color='red')
            ax14.plot(timegrid,len(timegrid)*[NF*-10],'--',color='red')
            ax14.set_ylim(8*NF*-10,8*NF*10)
        ax14.legend()
    
    
    
    
    return ic_guess, dm_guess




def reconstructDistributionFromTrueMasses(PNCORRECTION,mis,ris, obstimes, ic_guess, dm_guess, CARTESIANOBS = True,OBS3 = True,
                                          noisefactor = 1,seed = 0):
    """
    Reconstructs initial conditions and dark matter from a given true DM distribution

    Parameters
    ----------
    PNCORRECTION : boolean
        True if using 1PN correction
    mis : list of floats
        True masses of dark matter shells in MBH masses
    ris : list of floats
        distances of dark matter shells in AU.
    obstimes : list of floats
        Observation times in years.
    ic_guess : list of floats
        the initial guess for the initial conditions
    dm_guess : list of floats
        the initial guess for the dark matter masses
    noisefactor : float, optional
        Factor to reduce/increase the noise below/above the standard. The default is 1.
    seed : int, optional
        Random seed for reproducibility. The default is 0.
    CARTESIANOBS : boolean, optional
        True if using cartesian observations instead of orbital parameter observations. The default is True.
    OBS3 : boolean, optional
        True if only using 3 observed parameters (first, second and last = x,y and vz for cartesian). The default is True.

    Raises
    ------
    RuntimeError
        If the length of dark matter masses and distances do not match, an error is raised.

    Returns
    -------
    list of floats
        list of reconstructed dark matter masses.

    Reconstructs dark matter distribution starting from an initial guess
    """
    
    if len(mis) != len(ris):
        raise RuntimeError("Lengths of DM masses and distances does not match")
        
    if noisefactor == 0:
        ADD_NOISE = False
    else:
        ADD_NOISE = True
        
    np.set_printoptions(precision=5)
    
    IC = get_S2_IC()
    
    rx,ry,rz,vx,vy,vz = simulateOrbitsCartesian(True, IC, mis, ris, obstimes)
    rx, ry, vz = convertXYVZtoArcsec(rx, ry, vz)
    
    observationlist = np.column_stack((obstimes, ry,rx, vz))
    
    
    observationlist = addNoiseToObservations(observationlist,ADD_NOISE,seed,noisefactor)
    
    
    #observationlist =[[t1 x1 y1 ... vz1], [t2 x2 y2 ... vz2],...[]]
    return reconstructDistribution(observationlist, ic_guess, dm_guess,CARTESIANOBS,OBS3,noisefactor)

    
def reconstructFromFile(filename,ic_guess,dm_guess,noisefactor = 1,seed = 0):
    """
    Reconstructs initial conditions and dark matter from a given filename

    Parameters
    ----------
    filename : string
        Path of datafile (.txt).
    ic_guess : list of floats
        the initial guess for the initial conditions
    dm_guess : list of floats
        the initial guess for the dark matter masses
    noisefactor : float, optional
        Factor to reduce/increase the noise below/above the standard. The default is 1.
    seed : int, optional
        Random seed for reproducibility. The default is 0.
    
    Returns
    -------
    reconic, reconmis : list of floats, list of floats
        Reconstructed initial conditions and dark matter masses.

    """
    if noisefactor == 0:
        ADD_NOISE = False
    else:
        ADD_NOISE = True
    
    observations = np.loadtxt(filename)
    
    #Observations are given in time [yr], Y [arcsec], X [arcsec], VZ [km/s]
    observationlist = addNoiseToObservations(observations,ADD_NOISE,seed,noisefactor)
    #
    
    return reconstructDistribution(observationlist, ic_guess, dm_guess,CARTESIANOBS=True,OBS3=True,noisefactor=noisefactor)
    
    

def addNoiseToObservations(observations,ADD_NOISE = True,seed = 0,noisefactor = 1):
    """
    Reconstructs initial conditions and dark matter from a given filename

    Parameters
    ----------
    filename : string
        Path of datafile (.txt).
    ADD_NOISE : boolean, optional
        True if adding artificial noise to the data. The default is True.
    seed : int, optional
        Random seed for reproducibility. The default is 0.
    noisefactor : float, optional
        Factor to reduce/increase the noise below/above the standard. The default is 1.

    Returns
    -------
    reconic, reconmis : list of floats, list of floats
        Reconstructed initial conditions and dark matter masses.

    """
    #Important: input order of observations is Y, X, VZ, output order is X,Y,VZ
    
    #Input: arcsec
    #Output: AU
    
    M_0, D_0, T_0 = getBaseUnitConversions()
    
    timegrid = observations[:,0]
    
    
    rYs = observations[:,1]
    rXs = observations[:,2]
    vZs = observations[:,3]
    
    #Add Gaussian noise
    if ADD_NOISE:
        np.random.seed(seed)
        
        #Positional precision is 50 microarcseconds
        noiseLevelPos = 50 * 1e-6 * noisefactor
        #Velocity precision is 10 km/s
        noiseLevelVel = 10 * noisefactor
        
        
        noisePos = np.random.normal(0,noiseLevelPos,len(rXs))
        noiseVel = np.random.normal(0,noiseLevelVel,len(rXs))
        rYs = rYs + noisePos
        rXs = rXs + noisePos
        vZs = vZs + noiseVel
    
        
    rYs = arcseconds_to_AU(rYs)
    rXs = arcseconds_to_AU(rXs)
    vZs = vZs * 1000 * T_0 / D_0 
    
    
    observationlist = np.column_stack((timegrid, rXs, rYs, vZs))
    
    return observationlist




def getObservations(filename):
    
    M_0, D_0, T_0 = getBaseUnitConversions()
    
    observations = np.loadtxt(filename)
    
    timegrid = observations[:,0]
    
    rYs = observations[:,1]
    rXs = observations[:,2]
    vZs = observations[:,3]
    
        
    rYs = arcseconds_to_AU(rYs)
    rXs = arcseconds_to_AU(rXs)
    vZs = vZs * 1000 * T_0 / D_0 

    
    observationlist = np.column_stack((timegrid, rXs, rYs, vZs))
    
    return observationlist
        

        
# def loss(ta,ic_guess,dm_guess,obslist):
    
#     timegrid = obslist[:,0].copy()
    
#     observationlist = np.delete(obslist, [0], axis=1)    

#     t_grid = convertYearsTimegridToOurFormat(timegrid)
    
#     N = len(dm_guess)
#     #1 last simulation of the final guess:
#     ta.state[:] = np.append(ic_guess,np.array(variationalEqsInitialConditions(N)))
#     ta.time = 0
#     ta.pars[:N] = dm_guess
#     ta.pars[N:] = get_DM_distances(N, 2100)
#     out = ta.propagate_grid(t_grid)
#     finalsim = np.asarray(out[4][:,[0,1,2,3,4,5]])
#     finalsim = convertToCartesian(finalsim[:,0], finalsim[:,1], finalsim[:,2],\
#             finalsim[:,3], finalsim[:,4], finalsim[:,5])
#     finalsim = np.array(finalsim)
#     finalsim =  finalsim[[0,1,-1],:]
#     finalsim = np.transpose(finalsim)
    
#     #Diffs: [AU] [AU] [AU/T_0]
#     difxs = finalsim[:,0]-observationlist[:,0]
#     difys = finalsim[:,1]-observationlist[:,1]
#     difvzs = finalsim[:,-1]-observationlist[:,-1]
#     loss = 0
#     for i in range(len(difxs)):
#         loss += (difxs[i]**2)
#         loss += (difys[i]**2)
#         loss += (difvzs[i]**2)
    
#     return loss


class ReconstructDM:
    
    def __init__(self,ta,ic_guess,obslist):
        self.ta = ta
        self.ic_guess = ic_guess
        self.obslist = obslist
    
    def fitness(self,x):
        return (getGoodnessOfFit(self.ta,self.ic_guess,x,self.obslist),)
    
    def get_bounds(self):
        N = 5
        lb = [0] * N
        ub = [0.0005] * N
    
        return (lb, ub)
    
    

def getBestBahcallFit(filename,N): 
    
    def obsloss(ta,ic_guess,x,obslist):
        xlim = 2100
        
        rho0 = x[0]
        
        dm_guess,_ = get_BahcallWolf_DM(N, xlim,rho0)
        
        return getGoodnessOfFit(ta,ic_guess,dm_guess,obslist)
    
    
    class ReconstructBahcall:
        
        def __init__(self,filename,N):
            self.ta = buildTaylorIntegrator(True, N,LOAD_PICKLE=True)
            self.ic_guess = get_S2_IC()
            self.obslist = getObservations(filename)

            
        def fitness(self,x):
            return (obsloss(self.ta,self.ic_guess,x,self.obslist),)
        
        def get_bounds(self):
            lb = [1e-12]
            ub = [1e-10]
        
            return (lb, ub)
        
    
    pg.set_global_rng_seed(0)
    
    udp = ReconstructBahcall(filename,N)
    prob = pg.problem(udp)
    
    #If you get bad fits, feel free to increase the popside or amount of iterations
    popsize = 200
    pop = pg.population(prob,popsize)
    
    uda = pg.cmaes(1,force_bounds=True,memory=True)
    
    algo = pg.algorithm(uda)
    
    for i in range(1000):
        pop = algo.evolve(pop)
        
        # print(pop.champion_x,pop.champion_f)
    
    print('Best Bahcall fit')
    print(list(pop.champion_x),pop.champion_f)
    
    
    # plt.figure()
    reconmis, ris = get_BahcallWolf_DM(5,2100,pop.champion_x[0])
    # lb = udp.get_bounds()[0][0]
    # ub = udp.get_bounds()[1][0]
    # lbmis, ris = get_BahcallWolf_DM(5,2100,lb)
    # ubmis, ris = get_BahcallWolf_DM(5,2100,ub)
    # # plt.scatter(ris,dm,label='True')
    # plt.scatter(ris,reconmis,label='Best fit')
    # plt.plot(ris,lbmis,label='Lower bound')
    # plt.plot(ris,ubmis,label='Upper bound')
    # plt.legend()
    
    return reconmis


def getBestPlummerFit(filename,N): 
    
    def obsloss(ta,ic_guess,x,obslist):
        
        rho0 = x[0]
        r0 = x[1]
        
        dm_guess,_ = get_Plummer_DM(N, xlim,rho0,r0)
        
        return getGoodnessOfFit(ta,ic_guess,dm_guess,obslist)
    
    
    class ReconstructBahcall:
        
        def __init__(self,filename,N):
            self.ta = buildTaylorIntegrator(True, N,LOAD_PICKLE=True)
            self.ic_guess = get_S2_IC()
            self.obslist = getObservations(filename)

            
        def fitness(self,x):
            return (obsloss(self.ta,self.ic_guess,x,self.obslist),)
        
        def get_bounds(self):            
            lb = [1e-11, 1e-12]
            ub = [1e-9, 5000] 
        
            return (lb, ub)
        
    
    pg.set_global_rng_seed(0)
    
    xlim = 2100
    
    udp = ReconstructBahcall(filename,N)
    prob = pg.problem(udp)
    
    #If you get bad fits, feel free to increase the popside or amount of iterations
    popsize = 30
    pop = pg.population(prob,popsize)
    
    uda = pg.cmaes(1,force_bounds=True,memory=True)
    
    algo = pg.algorithm(uda)
    
    for i in range(200):
        pop = algo.evolve(pop)
        # print(pop.champion_x,pop.champion_f)
    
    print('Best Plummer fit')
    print(list(pop.champion_x),pop.champion_f)
    
    
    # plt.figure()
    reconmis, ris = get_Plummer_DM(N,xlim,pop.champion_x[0],pop.champion_x[1])
    # lb = udp.get_bounds()[0][0]
    # ub = udp.get_bounds()[1][0]
    # lbmis, ris = get_BahcallWolf_DM(5,2100,lb)
    # ubmis, ris = get_BahcallWolf_DM(5,2100,ub)
    # # plt.scatter(ris,dm,label='True')
    # plt.scatter(ris,reconmis,label='Best fit')
    # plt.plot(ris,lbmis,label='Lower bound')
    # plt.plot(ris,ubmis,label='Upper bound')
    # plt.legend()
    
    return reconmis
    




def lossesForDifferentNoiseProfiles(dm_guess,noisefactor):
    N = len(dm_guess)

    ta = buildTaylorIntegrator(True, N,LOAD_PICKLE=True)
    ic_guess = get_S2_IC()
        
    filename = 'Datasets/BahcallWolf_N={}.txt'.format(N)
    # filename = 'Datasets/Plummer_N={}.txt'.format(N)
    
    observations = np.loadtxt(filename)
    losses = []
    for i in range(100):
        #Observations are given in time [yr], Y [arcsec], X [arcsec], VZ [km/s]
        obslist = addNoiseToObservations(observations,ADD_NOISE=True,seed=i,noisefactor=noisefactor)
        
        losses.append(getGoodnessOfFit(ta,ic_guess,dm_guess,obslist,noisefactor))
    
    return losses


def lossLandscape(N=5,noisefactor=1e-1,nbrOfDistributions=10000):
    xlim = 2100
    
    ta = buildTaylorIntegrator(True, N,LOAD_PICKLE=True)
    ic_guess = get_S2_IC()
    
    mis,ris = get_BahcallWolf_DM(N, xlim)
    
    M_0, D_0, T_0 = getBaseUnitConversions()
        
    IC = get_S2_IC()
    
    obstimes = getObservationTimes()
    
    rx,ry,rz,vx,vy,vz = simulateOrbitsCartesian(True, IC, mis, ris, obstimes)
    rx, ry, vz = convertXYVZtoArcsec(rx, ry, vz)
    
    observationlist = np.column_stack((obstimes, ry,rx, vz))
    
    obslist = addNoiseToObservations(observationlist,ADD_NOISE=True,noisefactor=noisefactor)
    
    # import itertools
    # possdmvalues = list(np.linspace(0,0.0006,10))
    # print(possdmvalues)
    # # a = [[1,2,3],[4,5,6],[7,8,9,10]]
    # a = N * [possdmvalues]
    # print(a)
    # dms = list(itertools.product(*a))
    # # print(dms)
    # print(len(dms))
        
    dms = []
    for i in range(nbrOfDistributions):
        # dm_guess_new = Nf*[0]
        np.random.seed(i)
        noise = np.random.normal(0,0.0002,N)
        dm_new = mis+noise
        dms.append(dm_new.clip(min=0))
    
    # plt.figure()
    # for i in range(len(dms)):
    #     plt.plot(dms[i])
    # plt.plot(mis,label='True',linewidth=3)
    # plt.legend()
    
    trueloss = getGoodnessOfFit(ta, ic_guess, mis, obslist,noisefactor)
    print('True loss:',trueloss)
    
    
    start = time.time()
    
    losses = []
    for i in range(len(dms)):
        losses.append(getGoodnessOfFit(ta, ic_guess, dms[i], obslist,noisefactor))
        
    print(round(time.time()-start),'seconds elapsed')
    
    
    # plt.figure()
    # B = plt.boxplot(losses)
    # plt.ylabel('Loss')
    # for item in B['whiskers']:
    #     print(item.get_ydata() )
    # firstQuartile = B['whiskers'][0].get_ydata()[0]
    # thirdQuartile = B['whiskers'][1].get_ydata()[0]
    
        
    
    # cmin = 0
    cmin = trueloss
    # cmin = min(losses)
    # cmax = firstQuartile
    # sigma = 0.43097
    sigma = 0.01*trueloss
    cmax = trueloss + 5*sigma
    
    
    # plt.figure()
    # for i in range(len(dms)):
    #     if losses[i] < cmax:
    #         # plt.scatter(ris,dms[i],color='blue',alpha=1-(np.log(losses[i])/np.log(max(losses))))
    #         plt.scatter(ris,dms[i],c=N*[losses[i]],
    #                     cmap='rainbow',vmin=cmin,vmax=cmax)
    #         # plt.plot(ris,dms[i],color=3*[losses[i]/max(losses)])
    #         # plt.plot(ris,dms[i],color=colors[i])
    #         # print(N*[losses[i]])
    
    # plt.ylabel('Mass [MBH masses]')
    # plt.xlabel('Distance from MBH [AU]')
    # rp = 119.52867
    # ra = 1948.96214
    # plt.axvline(rp,linestyle='--',label='rp and ra',color='black')
    # plt.axvline(ra,linestyle='--',color='black')
    
    
    # cbar = plt.colorbar()
    # cbar.set_label("Loss",fontsize=12)
    # plt.clim(cmin,cmax)
    
    # plt.scatter(ris,mis,c=N*[0],label='True',cmap='rainbow',vmin=cmin,vmax=cmax)
    # plt.legend()
    
    print('Max loss:',max(losses))
    print('Min loss:',min(losses))
    
    nbrOfBands = 5
    # percentages = np.linspace(1,2,nbrOfBands)
    # lossbands = trueloss*percentages
    
    # lossbands = np.linspace(cmin,cmax,nbrOfBands+1)
    # print(lossbands)
    # lossbands = (lossbands[1:] + lossbands[:-1]) / 2
    
    # lossbands = np.linspace(cmin+sigma,cmax,nbrOfBands)
    lossbands = np.linspace(cmin,cmax-sigma,nbrOfBands)
    # lossbands = [6.72,6.79,6.86,6.93,6.99]
    # lossbands = (lossbands[1:] + lossbands[:-1]) / 2
    # print(lossbands)
    crange = np.linspace(0,1,nbrOfBands+1)
    crange = (crange[1:] + crange[:-1]) / 2
    
    
    plt.figure()
    for i in reversed(range(len(lossbands))):
        losslim = lossbands[i]
        
        newmises = []
        for j in range(len(losses)):
            if losses[j] < losslim:
                newmises.append(dms[j])
        
        newmises = np.array(newmises)
        if len(newmises) > 0:
            mins = np.amin(newmises,axis=0)
            maxs = np.amax(newmises,axis=0)
        else:
            mins = mis
            maxs = mis
            
        cmap = plt.cm.get_cmap('rainbow')


        rgba = cmap(crange[i])
        # plt.plot(ris,mins,color=rgba,label='loss={}({}sig)'.format(round(losslim,2),i+1))
        plt.plot(ris,mins,color=rgba,label='loss<={}({}%)'.format(round(losslim,2),i))
        # plt.plot(ris,mins,color=rgba,label='loss={}'.format(round(losslim,2)))
        plt.plot(ris,maxs,color=rgba)
    
    # cbar = plt.colorbar()
    # cbar.set_label("Loss",fontsize=12)
    # plt.clim(cmin,cmax)
    
    plt.ylabel('Mass [MBH masses]')
    plt.xlabel('Distance from MBH [AU]')
    rp = 119.52867
    ra = 1948.96214
    # plt.axvline(rp,linestyle='--',label='rp and ra',color='black')
    plt.axvline(rp,linestyle='--',color='black')
    plt.axvline(ra,linestyle='--',color='black')
    
    plt.scatter(ris,mis,label='True loss={}'.format(round(trueloss,2)))
    
    plt.legend()
    
    
    
    
            
            
            
    
    
    
    
#Deprecated: fitting masses to reconstructed distribution 
# def getBestBahcallMassFit(dm): 
    
#     def lossMassesBahcall(x,dm2):
#         rho0 = x[0]
#         #r0 is fixed (only 1 degree of freedom)
#         dm1, _ = get_BahcallWolf_DM(5,2100,rho0)
#         return sum(abs(np.array(dm1)-np.array(dm2)))
    
#     class ReconstructBahcall:
#         def __init__(self,dm):
#             self.dm = dm
            
#         def fitness(self,x):
#             return [lossMassesBahcall(x,self.dm)]
        
#         def get_bounds(self):
#             lb = [1e-12]
#             ub = [1e-10]
        
#             return (lb, ub)
        
    
#     pg.set_global_rng_seed(0)
    
#     udp = ReconstructBahcall(dm)
#     prob = pg.problem(udp)
#     popsize = 1000
#     pop = pg.population(prob,popsize)
    
#     uda = pg.cmaes(1,force_bounds=True,memory=True)
    
#     algo = pg.algorithm(uda)
    
#     for i in range(1000):
#         pop = algo.evolve(pop)
        
#         # print(pop.champion_x,pop.champion_f)
    
#     print('Best Bahcall fit')
#     print(list(pop.champion_x),pop.champion_f)
    
#     plt.figure()
#     reconmis, ris = get_BahcallWolf_DM(5,2100,pop.champion_x[0])
#     lb = udp.get_bounds()[0][0]
#     ub = udp.get_bounds()[1][0]
#     lbmis, ris = get_BahcallWolf_DM(5,2100,lb)
#     ubmis, ris = get_BahcallWolf_DM(5,2100,ub)
#     plt.scatter(ris,dm,label='True')
#     plt.scatter(ris,reconmis,label='Best fit')
#     plt.plot(ris,lbmis,label='Lower bound')
#     plt.plot(ris,ubmis,label='Upper bound')
#     plt.legend()
    
#     return reconmis


    
# def getBestPlummerMassFit(dm): 
    
#     def lossMassesPlummer(x,dm2):
#         rho0 = x[0]
#         r0 = x[1]
#         dm1, _ = get_Plummer_DM(5,2100,rho0,r0)
#         # dm2, _ = get_Plummer_DM(5, 2100)
#         return sum(abs(np.array(dm1)-np.array(dm2)))
    
#     class ReconstructPlummer:
#         def __init__(self,dm):
#             self.dm = dm
            
#         def fitness(self,x):
#             return [lossMassesPlummer(x,self.dm)]
        
#         def get_bounds(self):
#             lb = [1e-11, 1e-12]
#             ub = [1e-9, 5000] 
#             return (lb, ub)
    
#     pg.set_global_rng_seed(0)
        
#     udp = ReconstructPlummer(dm)
#     prob = pg.problem(udp)
#     popsize = 100
#     pop = pg.population(prob,popsize)
    
    
    
#     uda = pg.cmaes(1,force_bounds=True,memory=True)
    
#     algo = pg.algorithm(uda)
    
#     for i in range(1000):
#         pop = algo.evolve(pop)
        
#         # print(pop.champion_x,pop.champion_f)
    
#     print('Best Plummer fit')
#     print(list(pop.champion_x),pop.champion_f)
    
#     plt.figure()
#     reconmis, ris = get_Plummer_DM(5,2100,pop.champion_x[0],pop.champion_x[1])
    
#     lb = udp.get_bounds()[0][0]
#     ub = udp.get_bounds()[1][0]
#     lbmis, ris = get_Plummer_DM(5,2100,lb)
#     ubmis, ris = get_Plummer_DM(5,2100,ub)
    
#     plt.scatter(ris,dm,label='True')
#     plt.scatter(ris,reconmis,label='Best fit')
#     plt.plot(ris,lbmis,label='Lower bound')
#     plt.plot(ris,ubmis,label='Upper bound')
#     plt.legend()


#     return reconmis
    



# @Deprecated: Calculated the variance, but is not interpretable
# def getModelUncertainty(x0, DM, t_obs, noisefactor = 1):
#     """
#     Returns the model uncertainty for the given dark matter masses and initial conditions

#     Parameters
#     ----------
#     x0 : list of floats
#         the initial conditions.
#     DM : list of floats
#         the dark matter masses.
#     t_obs : list of floats
#         the observation times.
#     noisefactor : float
#         a multiplicative factor to increase/reduce the noise

#     Returns
#     -------
#     variance_x0 : list of floats
#         The variance of the initial conditions.
#     variance_DM : list of floats
#         The variance of the dark matter masses.

#     """
#     N = len(DM)
    
#     #Reset the state
#     ta = buildTaylorIntegrator(True, N, LOAD_PICKLE=True)
#     ta.state[:] = np.append(x0,np.array(variationalEqsInitialConditions(N)))
#     ta.pars[:N] = DM
#     ta.pars[N:] = get_DM_distances(N, xlim=2100)
#     ta.time = 0
#     #Simulate ta from initial guess (t=0) until t_obs
#     out = ta.propagate_grid(t_obs)
    
#     orbparamvalues = np.asarray(out[4][:,[0,1,2,3,4,5]])
    
#     Phis = np.asarray(out[4][:,6:6+36])
    
#     Psis = np.asarray(out[4][:,6+36:])
    
    
#     M_0, D_0, T_0 = getBaseUnitConversions()
#     sigmaPosition = arcseconds_to_AU(50 * 1e-6 * noisefactor) # AU
#     sigmaVelocity = 10 * 1000 * T_0 / D_0 * noisefactor  # km/s -> m/s -> our velocity units
    
#     weights = [1/sigmaPosition**2, 1/sigmaPosition**2, 1/sigmaVelocity**2]
    
#     # M_0, D_0, T_0 = getBaseUnitConversions()
#     # sigmaPosition = 50 * noisefactor # AU
#     # sigmaVelocity = 10 * noisefactor # km/s -> m/s -> our velocity units
#     # weights = [1/(sigmaPosition**2), 1/(sigmaPosition**2), 1/(sigmaVelocity**2)]
    
    
#     variance_x0 = np.zeros((1,6))
#     variance_DM = np.zeros((1,N))
    
#     varlist = ["p", "e", "i", "om", "w", "f"]
#     derobsdx = cartesianConversionGradient()
#     #Iterate over observations:
#     for oj in range(len(t_obs)):
        
#         Phi = Phis[oj].reshape((6,6))
#         Psi = Psis[oj].reshape((6,N))
        
#         #Need to multiply by gradient of observed cartesian vs orbital parameters
#         valuelist = orbparamvalues[oj]
#         peixyzDict = dict(zip(varlist, valuelist))
        
#         dobsdx = []
#         #OBS3 -> take x,y and vz instead of all 6
#         for i in [0,1,5]:
#             for j in range(6):
#                 dobsdx.append(hy.eval(derobsdx[i*6+j],peixyzDict))
#         dobsdx = np.array(dobsdx).reshape((3,6))
        
#         #Square the matrices element-wise
#         # dobsdx = np.square(dobsdx)
#         # Phi = np.square(Phi)
#         # Psi = np.square(Psi)
        
#         # varx0 = dobsdx @ Phi
#         # varDM = dobsdx @ Psi
#         varx0 = np.square(dobsdx @ Phi)
#         varDM = np.square(dobsdx @ Psi)
        
        
#         #Diffs: [AU] [AU] [AU/T_0]
#         # if oj == 24:
#         #     print(varDM[0])
#         # varDM[0] = 1e6*(AU_to_arcseconds(varDM[0]))
#         # varDM[1] = 1e6*(AU_to_arcseconds(varDM[1]))
#         # varDM[2] = varDM[2] * D_0 / (T_0 * 1000)
        
#         # print(varDM[0])
#         # if oj == 24:
#         #     print(varDM[0])
        
#         #Calculate gradient wrt initial conditions (phi) and dark matter masses (psi)
#         variance_x0 = variance_x0 + (weights @ varx0).reshape(1,-1)[0]
#         variance_DM = variance_DM + (weights @ varDM).reshape(1,-1)[0]
        
#         #Calculate gradient wrt initial conditions (phi) and dark matter masses (psi)
#         # variance_x0 = variance_x0 + (weights @ dobsdx @ Phi).reshape(1,-1)[0]
#         # variance_DM = variance_DM + (weights @ dobsdx @ Psi).reshape(1,-1)[0]
        
#         # if oj == 24:
#         #     print(weights)
#         #     print(dobsdx @ Psi)
#     # print(variance_x0)
#     # print(variance_DM)
    
#     # variance_x0 = np.square(variance_x0)
#     # variance_DM = np.square(variance_DM)
    
    
#     # return np.sqrt(1/variance_x0[0]), np.sqrt(1/variance_DM[0])
#     return 1/np.sqrt(variance_x0[0]), 1/np.sqrt(variance_DM[0])

#@Deprecated: Changes each mass until Chi^2 changes by 1
# def getParameterUncertainty(ta,ic_guess,dm_guess,obslist):
#     origGfit = getGoodnessOfFit(ta, ic_guess, dm_guess, obslist.copy())
    
#     N = len(dm_guess)
#     paramUncertaintiesMax = N*[0]
#     paramUncertaintiesMin = N*[0]
    
#     for n in range(N):
#         alphas = np.logspace(-2,0,20)
        
#         # print(1+alphas)
#         gfitMaxs = []
#         gfitMins = []
#         bestMaxFound = False
#         bestMinFound = False
        
#         for i in range(len(alphas)):
#             x_new = ic_guess.copy()
#             DM_new = dm_guess.copy() 
#             DM_new[n] = DM_new[n] * (1+alphas[i])
#             print(n, alphas[i])
#             gfitMax = getGoodnessOfFit(ta, x_new, DM_new, obslist.copy())
            
#             x_new = ic_guess.copy()
#             DM_new = dm_guess.copy() 
#             DM_new[n] = DM_new[n] * (1-alphas[i])
#             gfitMin = getGoodnessOfFit(ta, x_new, DM_new, obslist.copy())
            
#             if gfitMax > origGfit + 1 or i == len(alphas) - 1:
#                 if not bestMaxFound:
#                     paramUncertaintiesMax[n] = dm_guess[n] * (1+alphas[i]) - dm_guess[n]
                    
#                     print(1+alphas[i], 'gfit:',gfitMax)
#                     bestMaxFound = True
                
#             if gfitMin > origGfit + 1 or i == len(alphas) - 1:
#                 if not bestMinFound:
#                     paramUncertaintiesMin[n] = dm_guess[n] * (1-alphas[i]) - dm_guess[n]
                    
#                     print(1+alphas[i], 'gfit:',gfitMin)
#                     bestMinFound = True
                
#             gfitMaxs.append(gfitMax)
#             gfitMins.append(gfitMin)
    
    
#     paramUncertaintiesMax = np.abs(paramUncertaintiesMax)
#     paramUncertaintiesMin = np.abs(paramUncertaintiesMin)
    
    
#     # plt.figure()
#     # plt.plot(alphas,np.array(gfitMaxs)-1,label='Mascon 1')
#     # plt.hlines(1,alphas[0],alphas[-1])
#     # plt.xscale('log')
#     # plt.yscale('log')
#     # plt.xlabel('Parameter difference max')
#     # plt.ylabel('Goodness of fit')
#     # plt.legend()
    
#     # plt.figure()
#     # plt.plot(alphas,np.array(gfitMins)-1,label='Mascon 1')
#     # plt.hlines(1,alphas[0],alphas[-1])
#     # plt.xscale('log')
#     # plt.yscale('log')
#     # plt.xlabel('Parameter difference min')
#     # plt.ylabel('Goodness of fit')
#     # plt.legend()
    
#     uncertainties = np.array([paramUncertaintiesMin,paramUncertaintiesMax]).reshape((2,N))
#     # print(uncertainties)
#     # uncertainties = [paramUncertaintiesMax[0],paramUncertaintiesMin[0]]
#     # print(uncertainties)
#     ris = get_DM_distances(N, xlim=2100)
#     plt.figure()
#     mis,ris = get_BahcallWolf_DM(N, xlim=2100)
#     plt.scatter(ris,mis,label='True')
#     plt.scatter(ris,dm_guess,label='Reconstructed',color='orange')
#     plt.errorbar(ris,dm_guess,uncertainties,capsize=5,label='1 sigma',fmt='none',color='orange')
#     plt.legend()
#     plt.title('Reduced Chi^2 sensitivity to individual mascons')
    
#     return uncertainties
    
    
    
