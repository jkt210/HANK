import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit(parallel=True)        
def solve_hh_backwards(par,z_trans,r,w0,w1,phi0,phi1,vbeg_a_plus,vbeg_a,a,c,u,l0,l1,ss=False):
    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """

    for i_fix in nb.prange(par.Nfix):

        # a. solve step
        for i_z in nb.prange(par.Nz):
        
            ## i. labor supply
            l0[i_fix,i_z,:] = phi0*par.eta0_grid[i_fix]*par.z_grid[i_z]
            l1[i_fix,i_z,:] = phi1*par.eta1_grid[i_fix]*par.z_grid[i_z]

            ## ii. cash-on-hand
            m = (1+r)*par.a_grid + par.z_grid[i_z]*(w0*l0[i_fix,i_z,:]) + par.z_grid[i_z]*(w1*l1[i_fix,i_z,:])

            if ss:

                a[i_fix,i_z,:] = 0.0

            else:

                # iii. EGM
                c_endo = (par.beta_grid[i_fix]*vbeg_a_plus[i_fix,i_z])**(-1/par.sigma)
                m_endo = c_endo + par.a_grid # current consumption + end-of-period assets
                
                # iv. interpolation to fixed grid
                interp_1d_vec(m_endo,par.a_grid,m,a[i_fix,i_z])
                a[i_fix,i_z,:] = np.fmax(a[i_fix,i_z,:],0.0) # enforce borrowing constraint
            
            c[i_fix,i_z] = m-a[i_fix,i_z]
            u[i_fix,i_z] = c[i_fix,i_z]**(1.0-par.sigma)/(1.0-par.sigma)-par.nu

        # b. expectation step
        v_a = (1+r)*c[i_fix]**(-par.sigma)
        vbeg_a[i_fix] = z_trans[i_fix]@v_a