# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 16:21:03 2022

@author: Thibault
"""

import numpy as np
from matplotlib.pylab import plt
import time
import orbitModule


def reconstructDistribution(PNCORRECTION,mis,ris, CARTESIANOBS = True,OBS3 = False):
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
    
    
    np.set_printoptions(precision=5)
    
    M_0, D_0, T_0 = orbitModule.getBaseUnitConversions()
    
    #Time of observation
    # last_time = 2.032859999999999900e+03
    # last_time = 293097.9510676383
    # t_obslist = [last_time,last_time*1.1,last_time*1.2]
    
    t_obslist =  np.append(0,(np.linspace(0,16.056,228) * 365.25 * 24 * 60**2 /T_0 ) + 84187.772)
    
        
    IC = orbitModule.get_S2_IC()
        
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
        
        ta, ic_guess,DM_guess,sim,m,v = orbitModule.corrector(ta, ic_guess,DM_guess, \
              observationlist, t_obslist, alpha,beta1,beta2,eps,m,v,t,CARTESIANOBS)
        
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
    
    #Reshape
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
mis = [8.265361613774615e-07, 5.686349100247396e-06, 1.4927307091440782e-05, 2.7672594520147764e-05, 4.2787249490168856e-05, 5.9043390760264746e-05, 7.527496444156235e-05, 9.049525234007644e-05, 0.00010396460998861513, 0.00011520962910339889, 0.00012400425819432555, 0.0001303272388261693, 0.00013430953743146022, 0.00013618220727268707, 0.00013623112544037478, 0.00013476151923237666, 0.00013207265668939558, 0.00012844157430656538, 0.00012411404664116991, 0.00011930089158700878]

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


