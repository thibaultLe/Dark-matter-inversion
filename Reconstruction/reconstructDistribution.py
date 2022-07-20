# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 16:21:03 2022

@author: Thibault
"""

import heyoka as hy
import numpy as np
from matplotlib.pylab import plt
import time
import pickle

def convertToCartesian(lp,le,li,lom,lw,lf):
    lr = lp / (1 + le * np.cos(lf))
    
    # Position 
    rx = lr * (np.cos(lom) * np.cos(lw + lf) - np.cos(li)*np.sin(lom)*np.sin(lw+lf))
    ry = lr * (np.sin(lom) * np.cos(lw + lf) + np.cos(li)*np.cos(lom)*np.sin(lw+lf))
    rz = lr * np.sin(li) * np.sin(lw + lf)
    
    # Velocity 
    vx = -np.sqrt(1/lp) * (np.cos(lom) * (np.sin(lw+lf) + le*np.sin(lw)) + \
             np.cos(li) * np.sin(lom) * (np.cos(lw+lf) + le*np.cos(lw)))
    vy = -np.sqrt(1/lp) * (np.sin(lom) * (np.sin(lw+lf) + le*np.sin(lw)) - \
             np.cos(li) * np.cos(lom) * (np.cos(lw+lf) + le*np.cos(lw)))
    vz = np.sqrt(1/lp) * np.sin(li) * (np.cos(lw+lf) + le*np.cos(lw))
    
    return rx,ry,rz,vx,vy,vz



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


"""
Reconstructs dark matter distribution starting from an initial guess

@param: PNCORRECTION: True if using 1PN correction
@param: mis: initial guesses of masses of dark matter shells in MBH masses
@param: ris: distances of dark matter shells in AU

@return: mis distribution
"""
def reconstructDistribution(PNCORRECTION,mis,ris, CARTESIANOBS = True,OBS3 = True):
    """
    

    Parameters
    ----------
    PNCORRECTION : boolean
        True if using 1PN correction
    mis : list of floats
        initial guesses of masses of dark matter shells in MBH masses
    ris : list of floats
        distances of dark matter shells in AU.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    
    if len(mis) != len(ris):
        raise RuntimeError("Lengths of DM masses and distances does not match")
    
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
    k = 0.1
    #Amount of dark matter shells
    N = len(mis)
    
    
    
    #alpha in arcseconds
    alpha_mpe = 0.1249527719 
    #R in parsec
    R_mpe = 8277.09055007
    e_mpe = 0.884429099282  
    a_mpe = 2 * R_mpe * np.tan(alpha_mpe * np.pi / (2*648000)) * 3.08567758149e+16 / D_0
    p_mpe = a_mpe * (1-e_mpe**2) 
    # T_period = np.sqrt(a_mpe**3)*2*np.pi
    # print(T_period)
    # T_0mpe =  2010.3561125977762 * 365.25 * 24 * 60**2 /T_0
    
    #Initial conditions:
    IC= [p_mpe, e_mpe, -134.700204975 / 180 * np.pi, 228.191510132 / 180 * np.pi, \
      66.2689390128 / 180 * np.pi, 1]
        
    
    
    
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
    
    
    # Cartesian position and velocity conversion
    """
    rx = r * (hy.cos(om) * hy.cos(w + f) - hy.cos(i)*hy.sin(om)*hy.sin(w+f))
    ry = r * (hy.sin(om) * hy.cos(w + f) + hy.cos(i)*hy.cos(om)*hy.sin(w+f))
    rz = r * hy.sin(i) * hy.sin(w + f)
    
    vx = -hy.sqrt(GM/p) * (hy.cos(om) * (hy.sin(w+f) + e * hy.sin(w)) + \
              hy.cos(i) * hy.sin(om) * (hy.cos(w+f) + e * hy.cos(w)))
    vy = -hy.sqrt(GM/p) * (hy.sin(om) * (hy.sin(w+f) + e * hy.sin(w)) - \
              hy.cos(i) * hy.cos(om) * (hy.cos(w+f) + e * hy.cos(w)))
    vz = hy.sqrt(GM/p) * hy.sin(i) * (hy.cos(w+f) + e * hy.cos(w))
    
            
    """
    Variational equations
    """
    x = np.array([p,e,i,om,w,f])
    func = np.array([dpdt,dedt,didt,domdt,dwdt,dfdt])
    cart = np.array([rx,ry,rz,vx,vy,vz])
    
    
    
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
    
    
    
    dyn = []
    for state, rhs in zip(x,func):
        dyn.append((state, rhs))
    #Phi
    for state, rhs in zip(phi.reshape((36,)),dphidt.reshape((36,))):
        dyn.append((state, rhs))
    #Psi
    for state, rhs in zip(psi.reshape((6*N,)),dpsidt.reshape((6*N,))):
        dyn.append((state, rhs))
    # Initial conditions on the variational equations
    ic_var_phi = np.eye(6).reshape((36,)).tolist()
    ic_var_psi = np.zeros((6*N,)).tolist()
    
    
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
    
    
    #Derivative of cartesian observation wrt orbital parameters
    derobsdx = []
    #rows
    for i in range(6):
        #columns
        for j in range(6):
            derobsdx.append(hy.diff(cart[i],x[j]))
    
    
    
    def corrector(ta, x0, DMm0, DMr, obs, t_obs, alpha, beta1, beta2, eps, m, v, t):
        """
        

        Parameters
        ----------
        ta : hy.taylor_adaptive
            System of equations.
        x0 : list of floats
            initial condition guess (p,e,i,om,w,f)
        obs : list of list of floats
            observation at time tj of (p,e,i,om,w,f)
        t_obs : list of floats
            time tj

        Returns
        -------
        ta : hy.taylor_adaptive
            System of equations.
        x0_new
            The corrected initial conditions (p,e,i,om,w,f).

        """
        
        
        """
        Performs and logs a step of a corrector algorithm that takes a numerical integration from x0 -> T -> xf. The result
        is a new tentative x0 that should result in a closer observation
        """
        
        #Reset the state
        ta.state[:] = np.concatenate((x0,np.array(ic_var_phi),np.array(ic_var_psi)))
        ta.pars[:N] = DMm0
        ta.pars[N:] = DMr
        ta.time = 0
        #Simulate ta from initial guess until t_obs
        # ta.propagate_until(t_obs)
        out = ta.propagate_grid(t_obs)
        
        orbparamvalues = np.asarray(out[4][:,[0,1,2,3,4,5]]).copy()
        
        simulatedlist = orbparamvalues.copy()
        
        
        if CARTESIANOBS:
            simulatedlist = convertToCartesian(simulatedlist[:,0], simulatedlist[:,1], simulatedlist[:,2],\
                    simulatedlist[:,3], simulatedlist[:,4], simulatedlist[:,5])
            
            if OBS3:
                simulatedlist = np.array(simulatedlist)
                simulatedlist =  simulatedlist[[0,1,-1],:].transpose()
        
        
        if t == 0 or t==iterations-1:
            print('Simulation: ',simulatedlist)
            print('Observation:',obs)
        
        
        #Take difference of observation with simulation from initial guess
        difference = np.subtract(simulatedlist, obs)
        
        Phi = ta.state[6:6+36].reshape((6,6))
        
        Psi = ta.state[6+36:].reshape((6,N))
        
        
        gradx0 = np.zeros((1,6))
        gradDM0 = np.zeros((1,N))
        
        #Iterate over observations:
        for oj in range(len(obs)):
        #TODO: add stochasticity (random sample instead of full list)
            if CARTESIANOBS:
                #Need to multiply by gradient of observed cartesian vs orbital parameters
                varlist = ["p", "e", "i", "om", "w", "f"]
                valuelist = orbparamvalues[oj]
                    
                peixyzDict = dict(zip(varlist, valuelist))
                dobsdx = []
                for i in range(6):
                    for j in range(6):
                        dobsdx.append(hy.eval(derobsdx[i*6+j],peixyzDict))
                dobsdx = np.array(dobsdx).reshape((6,6))
                
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
        
        # x_new = xDM_new[:6]
        x_new = x0
        DM_new = xDM_new[6:]
        # DM_new = DMm0
        
        
        #Basic gradient descent:
        # delta = - alpha * grad
        # delta = delta.reshape(1,-1)[0]
        # x_new = x0+delta[:6]
        # DM_new = DMm0 + delta[6:]
        
        
        #DM sometimes becomes negative, don't allow this
        for i in range(len(DM_new)):
            if DM_new[i] < 0:
                DM_new[i] = 0
        
         
        return ta, x_new, DM_new, simulatedlist, m, v
        
    
    
    
    
    np.set_printoptions(precision=5)
    
        
    #last observation time
    # last_time = 2.032859999999999900e+03
    last_time = 293097.9510676383
    
        
    #Time of observation
    # t_obs = last_time
    # t_obslist = [last_time,last_time*1.1,last_time*1.2]
    
    t_obslist =  np.append(0,(np.linspace(0,16.056,228) * 365.25 * 24 * 60**2 /T_0 ) + 84187.772)
    
    
    
    #Setup for fake reconstruction:
    ta.state[:6] = IC
    ta.time = 0
    ta.pars[:N] = mis
    ta.pars[N:] = ris
    # ta.propagate_until(t_obs)
    out = ta.propagate_grid(t_obslist)
    
    
    
    observationlist = np.asarray(out[4][:,[0,1,2,3,4,5]]).copy()
    # print('obslist:',observationlist)
    
    
    if CARTESIANOBS:
        observationlist = convertToCartesian(observationlist[:,0], observationlist[:,1], observationlist[:,2],\
                observationlist[:,3], observationlist[:,4], observationlist[:,5])
        
        if OBS3:
            observationlist = np.array(observationlist)
            observationlist =  observationlist[[0,1,-1],:].transpose()
    
    #[[x1 y1 vz1], [x2 y2 vz2],...[]]
    # print(observationlist)
            
    # observation = ta.state[:6].copy()
    # if CARTESIANOBS:
    #     observation = convertToCartesian(observation[0], observation[1], observation[2],\
    #                                   observation[3], observation[4], observation[5])
    #     if OBS3:
    #         observation =  observation[0:2]+(observation[5],)
    
    
    # comparedData = np.loadtxt('1PN.txt')
    # comparedYs = comparedData[:,1]
    # comparedXs = comparedData[:,2]
    # comparedVZs = comparedData[:,3]
    
    # timegrid = comparedData[:,0]
    # #Use an offset so that t=0 corresponds to the first observation
    # timeoffset = timegrid[0]
    # timegrid = timegrid - timeoffset
    
    
    
    # initialize first and second moments
    m = np.array([0.0 for _ in range(len(IC)+N)])
    v = np.array([0.0 for _ in range(len(IC)+N)])
    
    # step size
    alpha = 1e-6
    
    #TODO: tune these
    # factor for average gradient
    beta1 = 0.9
    # factor for average squared gradient
    beta2 = 0.999
    #Precision
    eps = 1e-8
    
    
    #Initial guesses:
    ic_guess = IC.copy()
    # ic_guess = [p_mpe, e_mpe, -134.700204975 / 180 * np.pi, 228.191510132 / 180 * np.pi, \
    #   66.2689390128 / 180 * np.pi,1.1]
    # ic_guess = np.multiply(IC, len(IC)*[1.0001])
    
    DM_guess = N*[0]
    
    ICiterations = np.array([ic_guess])
    DMiterations = np.array([DM_guess])
    obsiterations = np.array([])
    
    iterations = 1000
    
    for t in range(iterations):
        if t != 0 and iterations > 5 and t % round(iterations/5) == 0: 
            print('Iteration',t,'done')
        
        ta, ic_guess,DM_guess,sim,m,v = corrector(ta, ic_guess,DM_guess, ris, observationlist, t_obslist, alpha,beta1,beta2,eps,m,v,t)
        
        ICiterations = np.append(ICiterations,ic_guess)
        DMiterations = np.append(DMiterations,DM_guess)
        obsiterations = np.append(obsiterations,sim)
        
    
    #1 last simulation of the final guess:
    ta.state[:6] = ic_guess
    ta.time = 0
    ta.pars[:N] = DM_guess
    ta.pars[N:] = ris
    out = ta.propagate_grid(t_obslist)
    finalsim = np.asarray(out[4][:,[0,1,2,3,4,5]]).copy()
    if CARTESIANOBS:
        finalsim = convertToCartesian(finalsim[:,0], finalsim[:,1], finalsim[:,2],\
                finalsim[:,3], finalsim[:,4], finalsim[:,5])
        if OBS3:
            finalsim = np.array(finalsim)
            finalsim =  finalsim[[0,1,-1],:].transpose()
    obsiterations = np.append(obsiterations,finalsim)
    
    ICiterations = ICiterations.reshape((iterations+1,6))  
    DMiterations = DMiterations.reshape((iterations+1,N))  
    obsiterations = obsiterations.reshape((iterations+1,len(observationlist),len(observationlist[0])))    
    
    
    print("")
    print('First guess for IC:',ICiterations[0])
    print('Reconstructed IC:  ',np.array(ic_guess))
    print("True IC:           ",np.array(IC))
    print("")
    print('First guess for DM:',DMiterations[0])
    print('Reconstructed DM:  ',DM_guess)
    print("True DM:           ",np.array(mis))
    print("")
    print('First simulation:',obsiterations[0])
    print('Last simulation: ',obsiterations[-1])
    print("True observation:",np.array(observationlist))
    
        
    iters = np.arange(0,iterations+1,1)
    
    
    
    #Plot convergence of initial conditions:
    absdiffs = np.sum(abs(np.subtract(ICiterations,(iterations+1)*[IC])),axis=1)
    plt.figure()
    plt.scatter(iters,absdiffs,color='blue',s=8)
    plt.ylabel("Difference")
    plt.xlabel("Amount of iterations")
    plt.title("Difference with true initial conditions")
    
    
    #Plot convergence of dark matter:
    absdiffs = np.sum((np.subtract(DMiterations,(iterations+1)*[mis])),axis=1)
    plt.figure()
    plt.scatter(iters,absdiffs,color='blue',s=8)
    plt.ylabel("Difference with true value")
    plt.xlabel("Amount of iterations")
    plt.title("Difference with true DM distribution")
    
    
    # absdiffsForF = (np.subtract(ICiterations[:,5],(iterations+1)*[IC[5]]))
    # plt.figure()
    # plt.scatter(iters,absdiffsForF,color='blue',s=8)
    # plt.ylabel("Difference with true value")
    # plt.xlabel("Amount of iterations")
    # plt.title("Gradient descent for finding initial f")
    
    
    #Convergence of observation:
    # absdiffs = [[difx obs 1, dify obs1, difz obs1] , [difx obs 2, dify obs2, difz obs2] 
    absdiffs = np.sum(abs(np.subtract(obsiterations,np.array((iterations+1)*[observationlist]))),axis=1)
    
    absdiffsTotal = np.sum(absdiffs,axis = 1)
    plt.figure()
    plt.scatter(iters,absdiffsTotal,color='blue',s=8)
    plt.ylabel("Difference with observation")
    plt.xlabel("Amount of iterations")
    plt.title("Gradient descent to match observation")
    
    if CARTESIANOBS and OBS3:
        #difference of x,y and vz observation:
        plt.figure()
        plt.scatter(t_obslist,AU_to_arcseconds(obsiterations[:][-1][:,0])-AU_to_arcseconds(observationlist[:,0]),color='blue',s=8,label='Difference')
        plt.plot(t_obslist,len(t_obslist)*[50],'--',label='Precision',color='red')
        plt.plot(t_obslist,len(t_obslist)*[-50],'--',color='red')
        plt.ylabel("Difference with observation")
        plt.xlabel("Time")
        plt.title("X simulated - X observed")
        plt.legend()
        
        plt.figure()
        plt.scatter(t_obslist,AU_to_arcseconds(obsiterations[:][-1][:,1])-AU_to_arcseconds(observationlist[:,1]),color='blue',s=8,label='Difference')
        plt.plot(t_obslist,len(t_obslist)*[50],'--',label='Precision',color='red')
        plt.plot(t_obslist,len(t_obslist)*[-50],'--',color='red')
        plt.ylabel("Difference with observation")
        plt.xlabel("Time")
        plt.title("Y simulated - Y observed")
        plt.legend()
        
        
        
        plt.figure()
        plt.scatter(t_obslist,obsiterations[:][-1][:,-1]* D_0 / (T_0 * 1000)-observationlist[:,-1]* D_0 / (T_0 * 1000),color='blue',s=8,label='Difference')
        plt.plot(t_obslist,len(t_obslist)*[10],'--',label='Precision',color='red')
        plt.plot(t_obslist,len(t_obslist)*[-10],'--',color='red')
        plt.ylabel("Difference with observation")
        plt.xlabel("Time")
        plt.title("VZ simulated - VZ observed")
        plt.legend()
    
    
    
    #Returns  observations (AU, meters/second)
    return DM_guess

ris =  [144.14414414414415, 288.2882882882883, 432.4324324324324, 576.5765765765766, 720.7207207207207, 864.8648648648648, 1009.009009009009, 1153.1531531531532, 1297.2972972972973, 1441.4414414414414, 1585.5855855855855, 1729.7297297297296, 1873.8738738738737, 2018.018018018018, 2162.162162162162, 2306.3063063063064, 2450.4504504504503, 2594.5945945945946, 2738.7387387387384, 2882.8828828828828]
mis = [8.265361613774615e-07, 5.686349100247396e-06, 1.4927307091440782e-05, 2.7672594520147764e-05, 4.2787249490168856e-05, 5.9043390760264746e-05, 7.527496444156235e-05, 9.049525234007644e-05, 0.00010396460998861513, 0.00011520962910339889, 0.00012400425819432555, 0.0001303272388261693, 0.00013430953743146022, 0.00013618220727268707, 0.00013623112544037478, 0.00013476151923237666, 0.00013207265668939558, 0.00012844157430656538, 0.00012411404664116991, 0.00011930089158700878]

# ris = [400]
# mis = [1e-2]

# ris = [1]
# mis=[0]

# reconstructDistribution(True,[1e-3],[400])
reconmis = reconstructDistribution(True,mis,ris,CARTESIANOBS = True,OBS3 = True)




rp = 119.52867
ra = 1948.96214

plt.figure()
plt.scatter(ris,reconmis,label='Reconstructed')
plt.scatter(ris,mis,label='Plummer model')
plt.axvline(-rp,linestyle='--',label='rp and ra',color='black')
plt.axvline(ra,linestyle='--',color='black')
plt.xlabel("Distance from MBH [AU]")
plt.ylabel("Mass [MBH masses]")
plt.title('Reconstructed dark matter distribution')
plt.legend()


