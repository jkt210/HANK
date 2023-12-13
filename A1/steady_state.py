import time
import numpy as np

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst
from consav.misc import elapsed

import root_finding

def prepare_hh_ss(model):
    """ prepare the household block to solve for steady state """

    par = model.par
    ss = model.ss

    ############
    # 1. grids #
    ############
    
    # a. beta
    par.beta_grid[:] = np.hstack( ( np.linspace(par.beta_mean-par.beta_delta,par.beta_mean+par.beta_delta,par.Nbeta), np.linspace(par.beta_mean-par.beta_delta,par.beta_mean+par.beta_delta,par.Nbeta)))
    par.eta0_grid[:] = np.hstack((np.ones(par.Nbeta),np.zeros(par.Nbeta)))
    par.eta1_grid[:] = np.hstack((np.zeros(par.Nbeta),np.ones(par.Nbeta)))

    # b. a
    par.a_grid[:] = equilogspace(0.0,ss.w1*par.a_max,par.Na)
    
    # c. z
    par.z_grid[:],z_trans,z_ergodic,_,_ = log_rouwenhorst(par.rho_z,par.sigma_psi,par.Nz)

    #############################################
    # 2. transition matrix initial distribution #
    #############################################
    
    ss.z_trans[:,:,:] = z_trans
    ss.Dbeg[:3,:,0] = z_ergodic*2/3*1/3
    ss.Dbeg[:3,:,1:] = 0.0
    ss.Dbeg[3:,:,0] = z_ergodic*1/3*1/3
    ss.Dbeg[3:,:,1:] = 0.0

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    # a. raw value
    y = (ss.w0*par.phi0+ss.w1*par.phi1)*par.z_grid 
    c = m = (1+ss.r)*par.a_grid[np.newaxis,:] + y[:,np.newaxis]
    v_a = (1+ss.r)*c**(-par.sigma)

    # b. expectation
    ss.vbeg_a[:] = ss.z_trans@v_a

def obj_ss(K_ss,model,do_print=False):
    """ objective when solving for steady state capital """

    par = model.par
    ss = model.ss

    # a. production
    ss.Gamma = 1.0
    ss.A = ss.K = K_ss
    ss.L0 = ss.phi0*2/3
    ss.L1 = ss.phi1*1/3
    ss.Y = (ss.Gamma*(ss.K**par.alpha)*(ss.L0**((1-par.alpha)/2))*(ss.L1**((1-par.alpha)/2)))  

    # b. implied prices
    ss.rK = par.alpha*ss.Gamma*(ss.K**(par.alpha-1.0))*(ss.L0**((1-par.alpha)/2))*(ss.L1**((1-par.alpha)/2))
    ss.r = ss.rK - par.delta
    ss.w0 = ((1.0-par.alpha)/2)*ss.Gamma*(ss.K**par.alpha)*(ss.L0**((-1.0-par.alpha)/2))*(ss.L1**((1-par.alpha)/2))
    ss.w1 = ((1.0-par.alpha)/2)*ss.Gamma*(ss.K**par.alpha)*(ss.L1**((-1.0-par.alpha)/2))*(ss.L0**((1-par.alpha)/2))

    # c. household behavior
    if do_print:

        print(f'guess {ss.K = :.4f}')    
        print(f'implied {ss.r = :.4f}')
        print(f'implied {ss.w0 = :.4f}')
        print(f'implied {ss.w1 = :.4f}')

    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    if do_print: print(f'implied {ss.A_hh = :.4f}')

    ss.phi0 = par.phi0_ss
    ss.phi1 = par.phi1_ss

    # d. market clearing
    ss.I = par.delta*ss.K
    ss.clearing_A = ss.A-ss.A_hh
    ss.L0 = ss.L0-ss.L0_hh
    ss.L1 = ss.L1-ss.L1_hh
    ss.clearing_Y = ss.Y-ss.C_hh-ss.I

    return ss.clearing_A # target to hit
    
def find_ss(model,method='direct',do_print=False,K_min=1.0,K_max=10.0,NK=10):
    """ find steady state using the direct or indirect method """

    t0 = time.time()

    if method == 'direct':
        find_ss_direct(model,do_print=do_print,K_min=K_min,K_max=K_max,NK=NK)
    elif method == 'indirect':
        find_ss_indirect(model,do_print=do_print)
    else:
        raise NotImplementedError

    if do_print: print(f'found steady state in {elapsed(t0)}')

def find_ss_direct(model,do_print=False,K_min=1.0,K_max=10.0,NK=10):
    """ find steady state using direct method """

    # a. broad search
    if do_print: print(f'### step 1: broad search ###\n')

    K_ss_vec = np.linspace(K_min,K_max,NK) # trial values
    clearing_A = np.zeros(K_ss_vec.size) # asset market errors

    for i,K_ss in enumerate(K_ss_vec):
        
        try:
            clearing_A[i] = obj_ss(K_ss,model,do_print=do_print)
        except Exception as e:
            clearing_A[i] = np.nan
            print(f'{e}')
            
        if do_print: print(f'clearing_A = {clearing_A[i]:12.8f}\n')
            
    # b. determine search bracket
    if do_print: print(f'### step 2: determine search bracket ###\n')

    K_max = np.min(K_ss_vec[clearing_A < 0])
    K_min = np.max(K_ss_vec[clearing_A > 0])

    if do_print: print(f'K in [{K_min:12.8f},{K_max:12.8f}]\n')

    # c. search
    if do_print: print(f'### step 3: search ###\n')

    root_finding.brentq(
        obj_ss,K_min,K_max,args=(model,),do_print=do_print,
        varname='K_ss',funcname='A_hh-K'
    )

def find_ss_indirect(model,do_print=False):
    """ find steady state using indirect method """

    par = model.par
    ss = model.ss

    ss.phi0 = par.phi0
    ss.phi1 = par.phi1

    # a. exogenous and targets
    ss.L_0 = 2/3*par.phi0
    ss.L_1 = 1/3*par.phi1
    ss.r = par.r_ss_target
    ss.w0 = par.w0_ss_target
    ss.w1 = par.w1_ss_target

    # b. stock and capital stock from household behavior
    model.solve_hh_ss(do_print=do_print) # give us ss.a and ss.c (steady state policy functions)
    model.simulate_hh_ss(do_print=do_print) # give us ss.D (steady state distribution)
    if do_print: print('')

    ss.A = ss.K = ss.A_hh
    
    # c. back technology and depreciation rate
    # ss.Gamma = ss.w / ((1-par.alpha)*(ss.K/ss.L)**par.alpha)
    ss.Gamma = 1.0
    ss.rK = par.alpha*ss.Gamma*(ss.K**(par.alpha-1))*(ss.L0**((1-par.alpha)/2))*(ss.L1**((1-par.alpha)/2))
    par.delta = ss.rK - ss.r

    # d. produktion and investment
    ss.Y = (ss.Gamma*(ss.K**par.alpha)*(ss.L0**((1-par.alpha)/2))*(ss.L1**((1-par.alpha)/2)))  
    ss.I = par.delta*ss.K

    # e. market clearing
    ss.clearing_A = ss.A-ss.A_hh
    ss.clearing_L0 = ss.L0-ss.L0_hh
    ss.clearing_L1 = ss.L1-ss.L1_hh
    ss.clearing_Y = ss.Y-ss.C_hh-ss.I

    # f. print
    if do_print:

        print(f'Implied K = {ss.K:6.3f}')
        print(f'Implied Y = {ss.Y:6.3f}')
        print(f'Implied Gamma = {ss.Gamma:6.3f}')
        print(f'Implied delta = {par.delta:6.3f}') # check is positive
        print(f'Implied K/Y = {ss.K/ss.Y:6.3f}') 
        print(f'Discrepancy in A = {ss.clearing_A:12.8f}') # = 0 by construction
        print(f'Discrepancy in L0 = {ss.clearing_L0:12.8f}') # = 0 by construction
        print(f'Discrepancy in L1 = {ss.clearing_L1:12.8f}') # = 0 by construction
        print(f'Discrepancy in Y = {ss.clearing_Y:12.8f}') # != 0 due to numerical error 
