# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 16:21:03 2022

@author: Thibault
"""

import heyoka as hy
import numpy as np
from matplotlib.pylab import plt
import time
import orbitModule


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
        
    N = len(mis)
    
    ta = orbitModule.buildTaylorIntegrator(PNCORRECTION, N)
    derobsdx = orbitModule.cartesianConversionGradient()
    
    
    def corrector(ta, x0, DMm0, obs, t_obs, alpha, beta1, beta2, eps, m, v, t):
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
        ta.state[:] = np.append(x0,np.array(orbitModule.variationalEqsInitialConditions(N)))
        ta.pars[:N] = DMm0
        ta.time = 0
        #Simulate ta from initial guess (t=0) until t_obs
        out = ta.propagate_grid(t_obs)
        
        orbparamvalues = np.asarray(out[4][:,[0,1,2,3,4,5]]).copy()
        
        simulatedlist = orbparamvalues.copy()
        
        
        if CARTESIANOBS:
            simulatedlist = orbitModule.convertToCartesian(simulatedlist[:,0], simulatedlist[:,1], simulatedlist[:,2],\
                    simulatedlist[:,3], simulatedlist[:,4], simulatedlist[:,5])
            
            if OBS3:
                simulatedlist = np.array(simulatedlist)
                simulatedlist =  simulatedlist[[0,1,-1],:].transpose()
        
        
        # if t == 0 or t==iterations-1:
        #     print('Simulation: ',simulatedlist)
        #     print('Observation:',obs)
        
        
        #Take difference of observation with simulation from initial guess
        difference = np.subtract(simulatedlist, obs)
        
        Phi = ta.state[6:6+36].reshape((6,6))
        
        Psi = ta.state[6+36:].reshape((6,N))
        
        
        gradx0 = np.zeros((1,6))
        gradDM0 = np.zeros((1,N))
        
        varlist = ["p", "e", "i", "om", "w", "f"]
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
    
    M_0, D_0, T_0 = orbitModule.getBaseUnitConversions()
    
    #Time of observation
    # last_time = 2.032859999999999900e+03
    # last_time = 293097.9510676383
    # t_obs = last_time
    # t_obslist = [last_time,last_time*1.1,last_time*1.2]
    
    t_obslist =  np.append(0,(np.linspace(0,16.056,228) * 365.25 * 24 * 60**2 /T_0 ) + 84187.772)
    
    
    
    # #alpha in arcseconds
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
        
    #Setup for fake reconstruction:
    ta.state[:6] = IC
    ta.time = 0
    ta.pars[:N] = mis
    ta.pars[N:] = ris
    out = ta.propagate_grid(t_obslist)
    
    
    
    observationlist = np.asarray(out[4][:,[0,1,2,3,4,5]]).copy()
    
    
    if CARTESIANOBS:
        observationlist = orbitModule.convertToCartesian(observationlist[:,0], observationlist[:,1], observationlist[:,2],\
                observationlist[:,3], observationlist[:,4], observationlist[:,5])
        
        if OBS3:
            observationlist = np.array(observationlist)
            observationlist =  observationlist[[0,1,-1],:].transpose()
    
    #observationlist =[[x1 y1 vz1], [x2 y2 vz2],...[]]
    # print(observationlist)
            
    
    
    # initialize first and second moments
    m = np.array([0.0 for _ in range(len(IC)+N)])
    v = np.array([0.0 for _ in range(len(IC)+N)])
    
    # step size
    alpha = 1e-5
    
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
    
    
    iterations = 400
    
    ICiterations = np.array([ic_guess])
    DMiterations = np.array([DM_guess])
    obsiterations = np.array([])
    for t in range(iterations):
        if t != 0 and iterations > 5 and t % round(iterations/5) == 0: 
            print('Iteration',t,'done')
        
        ta, ic_guess,DM_guess,sim,m,v = corrector(ta, ic_guess,DM_guess, observationlist, t_obslist, alpha,beta1,beta2,eps,m,v,t)
        
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
        finalsim = orbitModule.convertToCartesian(finalsim[:,0], finalsim[:,1], finalsim[:,2],\
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
    # print('First simulation:',obsiterations[0])
    # print('Last simulation: ',obsiterations[-1])
    # print("True observation:",np.array(observationlist))
    
        
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
        plt.scatter(t_obslist,orbitModule.AU_to_arcseconds(obsiterations[:][-1][:,0])-orbitModule.AU_to_arcseconds(observationlist[:,0]),color='blue',s=8,label='Difference')
        plt.plot(t_obslist,len(t_obslist)*[50],'--',label='Precision',color='red')
        plt.plot(t_obslist,len(t_obslist)*[-50],'--',color='red')
        plt.ylabel("Difference with observation")
        plt.xlabel("Time")
        plt.title("X simulated - X observed")
        plt.legend()
        
        plt.figure()
        plt.scatter(t_obslist,orbitModule.AU_to_arcseconds(obsiterations[:][-1][:,1])-orbitModule.AU_to_arcseconds(observationlist[:,1]),color='blue',s=8,label='Difference')
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
mis = np.array([8.265361613774615e-07, 5.686349100247396e-06, 1.4927307091440782e-05, 2.7672594520147764e-05, 4.2787249490168856e-05, 5.9043390760264746e-05, 7.527496444156235e-05, 9.049525234007644e-05, 0.00010396460998861513, 0.00011520962910339889, 0.00012400425819432555, 0.0001303272388261693, 0.00013430953743146022, 0.00013618220727268707, 0.00013623112544037478, 0.00013476151923237666, 0.00013207265668939558, 0.00012844157430656538, 0.00012411404664116991, 0.00011930089158700878])

# ris = [400]
# mis = [1e-2]

# ris = [1]
# mis=[0]

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


