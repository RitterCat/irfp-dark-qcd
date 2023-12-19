import constants as cn
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

pc_Nd = {}

for Nd in [2,3,4]:
    with open(f'precoeffs/pc_{Nd}', 'rb') as fp:
        pc = pkl.load(fp)
        pc_Nd[Nd] = pc
        
pc_SUSY_Nd = {}

for Nd in [2,3,4]:
    with open(f'precoeffs/pc_SUSY_{Nd}', 'rb') as fp:
        pc = pkl.load(fp)
        pc_SUSY_Nd[Nd] = pc
        
class SubModel:
    
    def __init__(self, n_fermion, n_scalar, Nd):
        self.n_fermion = n_fermion
        self.n_scalar = n_scalar
        self.Nd = Nd
        
        precoeffs = pc_Nd[self.Nd]
        
        self.coeffs = [sum((n_fermion*Cf).flat) + sum((n_scalar*Cs).flat) + Cg for Cf, Cs, Cg in zip(*precoeffs)]
        
    @property
    def fixed_point(self):
        A, B, C, D, E, F = self.coeffs
        
        alpha_s_star = (A*E - C*D)/(C*F - B*E)
        alpha_d_star = (B*D - A*F)/(C*F - B*E)

        return np.array([alpha_s_star, alpha_d_star])
        
    def betas(self, alphas):
        A, B, C, D, E, F = self.coeffs
        alpha_s, alpha_d = alphas
        
        beta_c = A*alpha_s**2 + B*alpha_s**3 + C*alpha_s**2*alpha_d
        beta_d = D*alpha_d**2 + E*alpha_d**3 + F*alpha_d**2*alpha_s
        
        return np.array([beta_c, beta_d])
        
    def coupling_evolution(self, alphas_initial, mu_initial, mu_final):
        log_mu_initial = np.log(mu_initial)
        log_mu_final = np.log(mu_final)
        return solve_ivp(lambda mu, alphas: self.betas(alphas), [log_mu_initial, log_mu_final], alphas_initial)
        
#     def dark_coupling_evolution(self, alpha_d_initial, mu_initial, mu_final):
#         log_mu_initial = np.log(mu_initial)
#         log_mu_final = np.log(mu_final)
#         return solve_ivp(lambda mu, alpha_d: self.betas((0, alpha_d))[1], [log_mu_initial, log_mu_final], [alpha_d_initial])
    
class ModelClass:
    
    def __init__(self, n_fermion, n_scalar, n_dq, Nd):
        self.n_fermion = n_fermion
        self.n_scalar = n_scalar
        self.n_dq = n_dq
        self.Nd = Nd
        
        self.CJT_bound = (np.pi/3)/((self.Nd**2 - 1)/(2*self.Nd))
        
        self.n_fermion_EFT6 = np.zeros((5,5), int)
        self.n_fermion_EFT6[1][0] += 6 # there are 6 light quarks above mt
        self.n_fermion_EFT6[0][1] += self.n_dq # there are n_dq light dark quarks
        self.n_scalar_EFT6 = np.zeros((5,5), int) # there are no light scalars
        
        self.n_fermion_EFT5 = np.zeros((5,5), int)
        self.n_fermion_EFT5[1][0] += 5 # there are 5 light quarks below mt
        self.n_fermion_EFT5[0][1] += self.n_dq # there are n_dq light dark quarks
        self.n_scalar_EFT5 = np.zeros((5,5), int) # there are no light scalars
        
        self.UV = SubModel(self.n_fermion, self.n_scalar, self.Nd)
        self.EFT6 = SubModel(self.n_fermion_EFT6, self.n_scalar_EFT6, self.Nd)
        self.EFT5 = SubModel(self.n_fermion_EFT5, self.n_scalar_EFT5, self.Nd)
        
    def _fn_to_solve_for_M(self, M, alphas_UV):
        
        EFT5_evol_s = self.EFT5.coupling_evolution((cn.alpha_s_MZ, 0), cn.MZ, cn.mt)
        alpha_s_mt = EFT5_evol_s.y[0][-1]
        EFT6_evol_s = self.EFT6.coupling_evolution((alpha_s_mt, 0), cn.mt, M)
        UV_evol = self.UV.coupling_evolution(alphas_UV, cn.mu_UV, M)
        
        alpha_s_EFT = EFT6_evol_s.y[0][-1]
        alpha_s = UV_evol.y[0][-1]
        
        return alpha_s - alpha_s_EFT        
    
    def Lambda_dQCD_M(self, alphas_UV):
        f = lambda M: self._fn_to_solve_for_M(M, alphas_UV)
        
        fmin = f(cn.mt)
        fmax = f(cn.mu_UV)
        
        if fmin > 0:
            M = cn.mt
        elif fmax < 0:
            M = cn.mu_UV
        else:
            M = brentq(f, cn.mt, cn.mu_UV)
            
        alpha_d_0 = self.UV.coupling_evolution(alphas_UV, cn.mu_UV, M).y[1][-1]
        
        fn_for_IR_evol_d = lambda mu: self.EFT6.coupling_evolution((0, alpha_d_0), M, mu).y[1][-1] - self.CJT_bound
        
        fn_min = fn_for_IR_evol_d(1e-100)
        fn_max = fn_for_IR_evol_d(M)
        
        if fn_min < 0:
            Lambda_dQCD = 1e-100
        elif fn_max > 0:
            Lambda_dQCD = 0 # choose a better value, but indicate that one cannot solve for an IRFP (alpha_d is non-perturbative at the decoupling scale)
        else:
            Lambda_dQCD = brentq(fn_for_IR_evol_d, 1e-100, M)
        
        return (Lambda_dQCD, M)
        
    def _alphas_UV(self, LdQCD, M):
        EFT5_evol_s = self.EFT5.coupling_evolution((cn.alpha_s_MZ, 0), cn.MZ, cn.mt)
        alpha_s_mt = EFT5_evol_s.y[0][-1]
        EFT6_evol_s = self.EFT6.coupling_evolution((alpha_s_mt, 0), cn.mt, M)
        EFT_evol_d = self.EFT6.coupling_evolution((0, self.CJT_bound), LdQCD, M)
        
        alpha_s_0 = EFT6_evol_s.y[0][-1]
        alpha_d_0 = EFT_evol_d.y[1][-1]
        
        UV_evol = self.UV.coupling_evolution((alpha_s_0, alpha_d_0), M, cn.mu_UV)
        
        return UV_evol
        
    def _M_bounds_finite_aUV(self, LdQCD):
        f = lambda M: self._alphas_UV(LdQCD, M).t[-1] - np.log(cn.mu_UV) + 1e-5
        
        fmin = f(cn.mt)
        
        if fmin > 0:
            return (cn.mt, cn.mu_UV)
        
        M_min_0 = brentq(f, cn.mt, cn.mu_UV)
        
        for i in range(5):
            M_min = M_min_0*(1 + 10**(-5+i))
            if f(M_min) == 1e-5:
                return (M_min, cn.mu_UV)
            
        return (M_min, cn.mu_UV)
                
        
    def _M_bounds_pert_aUV(self, LdQCD):
        M_min_0, M_max_0 = self._M_bounds_finite_aUV(LdQCD)
        
        f = lambda M: max(self._alphas_UV(LdQCD, M).y.T[-1]) - 0.3
        
        fmin = f(M_min_0)
        
        if fmin < 0:
            return (M_min_0, M_max_0)
        
        M_min = brentq(f, M_min_0, M_max_0)
        
        return(M_min, M_max_0)        
        
    @staticmethod
    def _M_values(M_min, M_max):
        n_points = 99
        M_values = np.ones(n_points)*M_min + np.logspace(np.log10(M_min)-3, np.log10(M_max - M_min), n_points)
        np.insert(M_values, 0, M_min)
        return M_values
        
    def alphas_UV_contour(self, LdQCD):
        M_min, M_max = self._M_bounds_pert_aUV(LdQCD)
        
        M_values = self._M_values(M_min, M_max)
        
        aUV_contour = np.array([self._alphas_UV(LdQCD, M).y.T[-1] for M in M_values])
        
        return aUV_contour
       
    @property
    def alphas_UV_low(self):
        if not hasattr(self, '_alphas_UV_low'):
            self._alphas_UV_low = self.alphas_UV_contour(cn.LdQCD_low)
        return self._alphas_UV_low
       
    @property
    def alphas_UV_high(self):
        if not hasattr(self, '_alphas_UV_high'):
            self._alphas_UV_high = self.alphas_UV_contour(cn.LdQCD_high)
        return self._alphas_UV_high
        
    @property
    def epsilon_v(self):
        if not hasattr(self, '_epsilon_v'):
            c_low = self.alphas_UV_low
            c_high = self.alphas_UV_high

            if np.argmax(c_low[0]) == np.argmax(c_high[0]):
                c_total = np.concatenate((c_low, np.flip(c_high, axis=0)))
            else:
                c_total = np.concatenate((c_low, np.flip(c_high, axis=0), 0.3*np.ones((1,2))))

            valid_region = Polygon(c_total)

            self._epsilon_v = valid_region.area/0.3**2
        
        return self._epsilon_v
    
#     def inv_betas(self, inv_alphas):
#         A, B, C, D, E, F = self.coeffs
#         inv_alpha_s, inv_alpha_d = inv_alphas
        
#         inv_beta_c = -A - B/inv_alpha_s - C/inv_alpha_d
#         inv_beta_d = -D - E/inv_alpha_d - F/inv_alpha_s
        
#         return np.array([inv_beta_c, inv_beta_d])
    
#     def inv_coupling_evolution(self, inv_alphas_initial, mu_initial, mu_final):
#         log_mu_initial = np.log(mu_initial)
#         log_mu_final = np.log(mu_final)
#         return solve_ivp(lambda mu, inv_alphas: self.inv_betas(inv_alphas), [log_mu_initial, log_mu_final], inv_alphas_initial)

class SUSYSubModel:
    
    def __init__(self, n_multiplet, Nd):
        self.n_multiplet = n_multiplet
        self.Nd = Nd
        
        precoeffs = pc_SUSY_Nd[self.Nd]
        
        self.coeffs = [sum((n_multiplet*Cm).flat) + Cg for Cm, Cg in zip(*precoeffs)]
        
    @property
    def fixed_point(self):
        A, B, C, D, E, F = self.coeffs
        
        alpha_s_star = (A*E - C*D)/(C*F - B*E)
        alpha_d_star = (B*D - A*F)/(C*F - B*E)

        return np.array([alpha_s_star, alpha_d_star])
        
    def betas(self, alphas):
        A, B, C, D, E, F = self.coeffs
        alpha_s, alpha_d = alphas
        
        beta_c = A*alpha_s**2 + B*alpha_s**3 + C*alpha_s**2*alpha_d
        beta_d = D*alpha_d**2 + E*alpha_d**3 + F*alpha_d**2*alpha_s
        
        return np.array([beta_c, beta_d])
        
    def coupling_evolution(self, alphas_initial, mu_initial, mu_final):
        log_mu_initial = np.log(mu_initial)
        log_mu_final = np.log(mu_final)
        return solve_ivp(lambda mu, alphas: self.betas(alphas), [log_mu_initial, log_mu_final], alphas_initial)
        
#     def dark_coupling_evolution(self, alpha_d_initial, mu_initial, mu_final):
#         log_mu_initial = np.log(mu_initial)
#         log_mu_final = np.log(mu_final)
#         return solve_ivp(lambda mu, alpha_d: self.betas((0, alpha_d))[1], [log_mu_initial, log_mu_final], [alpha_d_initial])

class SUSYModelClass:
    
    def __init__(self, n_multiplet, n_dq, Nd):
        self.n_multiplet = 2*n_multiplet # For each field we introduce (i.e. Dirac fermion/complex scalar) we need two chiral multiplets
        self.n_dq = n_dq
        self.Nd = Nd
        
        self.CJT_bound = (np.pi/3)/((self.Nd**2 - 1)/(2*self.Nd))
        
        ### Lambda_UV ###
        
        self.UV = SUSYSubModel(self.n_multiplet, Nd)
        
        ### M (the mass scale of the heavy new physics) ###
        
        self.n_multiplet_SUSY = np.zeros((5,5), int)
        self.n_multiplet_SUSY[1][0] += 12 # there are 12 light chiral supermultiplets (2 per quark, which is a Dirac fermion)
                                          # we are above the SUSY breaking scale so the masses of quarks and squarks will be the same ?
        self.n_multiplet_SUSY[0][1] += 2*self.n_dq # there are 2 chiral supermultiplets per dark quark (Dirac fermion), also containing dark squarks
        
        self.SUSY = SUSYSubModel(self.n_multiplet_SUSY, Nd)
        
        ### MSUSY ###
        
        self.n_fermion_EFT6 = np.zeros((5,5), int)
        self.n_fermion_EFT6[1][0] += 6 # there are 6 light quarks for energy scales above the top mass
        self.n_fermion_EFT6[0][1] += self.n_dq # there are n_dq light dark quarks
        self.n_scalar_EFT6 = np.zeros((5,5), int) # there are no light scalars
        
        self.EFT6 = SubModel(self.n_fermion_EFT6, self.n_scalar_EFT6, Nd)
        
        ### mt ###
        
        self.n_fermion_EFT5 = np.zeros((5,5), int)
        self.n_fermion_EFT5[1][0] += 5 # there are 5 light quarks for energy scales below the top mass
        self.n_fermion_EFT5[0][1] += self.n_dq # there are n_dq light dark quarks
        self.n_scalar_EFT5 = np.zeros((5,5), int) 
        
        self.EFT5 = SubModel(self.n_fermion_EFT5, self.n_scalar_EFT5, Nd)
        
    def _fn_to_solve_for_M(self, M, alphas_UV):
        
        EFT5_evol_s = self.EFT5.coupling_evolution((cn.alpha_s_MZ, 0), cn.MZ, cn.mt)
        alpha_s_mt = EFT5_evol_s.y[0][-1]
        EFT6_evol_s = self.EFT6.coupling_evolution((alpha_s_mt, 0), cn.mt, cn.MSUSY)
        alpha_s_MSUSY = EFT6_evol_s.y[0][-1]
        SUSY_evol_s = self.SUSY.coupling_evolution((alpha_s_MSUSY, 0), cn.MSUSY, M)
        
        UV_evol = self.UV.coupling_evolution(alphas_UV, cn.mu_UV, M)
        
        alpha_s_EFT = SUSY_evol_s.y[0][-1]
        alpha_s = UV_evol.y[0][-1]
        
        return alpha_s - alpha_s_EFT
    
    def Lambda_dQCD_M(self, alphas_UV):
        f = lambda M: self._fn_to_solve_for_M(M, alphas_UV)
        
        fmin = f(cn.MSUSY)
        fmax = f(cn.mu_UV)
        
        if fmin > 0:
            M = cn.MSUSY
        elif fmax < 0:
            M = cn.mu_UV
        else:
            M = brentq(f, cn.MZ, cn.mu_UV)
            
        alpha_d_0 = self.UV.coupling_evolution(alphas_UV, cn.mu_UV, M).y[1][-1]
        
        SUSY_evol_d = self.SUSY.coupling_evolution((0, alpha_d_0), M, cn.MSUSY)
        alpha_d_MSUSY = SUSY_evol_d.y[1][-1]
        
        fn_for_IR_evol_d = lambda mu: self.EFT6.coupling_evolution((0, alpha_d_MSUSY), cn.MSUSY, mu).y[1][-1] - self.CJT_bound
        
        fn_min = fn_for_IR_evol_d(1e-100)
        fn_max = fn_for_IR_evol_d(M)
        
        if fn_min < 0:
            Lambda_dQCD = 1e-100
        elif fn_max > 0:
            Lambda_dQCD = 0 # choose a better value, but indicate that one cannot solve for an IRFP (alpha_d is non-perturbative at the decoupling scale)
        else:
            Lambda_dQCD = brentq(fn_for_IR_evol_d, 1e-100, M)
        
        return (Lambda_dQCD, M)
    
    def _alphas_UV(self, LdQCD, M):
        EFT5_evol_s = self.EFT5.coupling_evolution((cn.alpha_s_mt, 0), cn.MZ, cn.mt)
        alpha_s_mt = EFT5_evol_s.y[0][-1]
        EFT6_evol_s = self.EFT6.coupling_evolution((cn.alpha_s_mt, 0), cn.mt, cn.MSUSY)
        alpha_s_MSUSY = EFT6_evol_s.y[0][-1]
        EFT5_evol_d = self.EFT5.coupling_evolution((0, self.CJT_bound), LdQCD, cn.MSUSY)
        alpha_d_MSUSY = EFT5_evol_d.y[1][-1]
        SUSY_evol = self.SUSY.coupling_evolution((alpha_s_MSUSY, alpha_d_MSUSY), cn.MSUSY, M)
        
        alpha_s_0 = SUSY_evol.y[0][-1]
        alpha_d_0 = SUSY_evol.y[1][-1]
        
        UV_evol = self.UV.coupling_evolution((alpha_s_0, alpha_d_0), M, cn.mu_UV)
        
        return UV_evol
        
    def _M_bounds_finite_aUV(self, LdQCD):
        f = lambda M: self._alphas_UV(LdQCD, M).t[-1] - np.log(cn.mu_UV) + 1e-5
        
        fmin = f(cn.MSUSY)
        
        if fmin > 0:
            return (cn.MSUSY, cn.mu_UV)
        
        M_min_0 = brentq(f, cn.MSUSY, cn.mu_UV)
        
        for i in range(5):
            M_min = M_min_0*(1 + 10**(-5+i))
            if f(M_min) == 1e-5:
                return (M_min, cn.mu_UV)
            
        return (M_min, cn.mu_UV)
                
    def _M_bounds_pert_aUV(self, LdQCD):
        M_min_0, M_max_0 = self._M_bounds_finite_aUV(LdQCD)
        
        f = lambda M: max(self._alphas_UV(LdQCD, M).y.T[-1]) - 0.3
        
        fmin = f(M_min_0)
        
        if fmin < 0:
            return (M_min_0, M_max_0)
        
        M_min = brentq(f, M_min_0, M_max_0)
        
        return(M_min, M_max_0)        
        
    @staticmethod
    def _M_values(M_min, M_max):
        n_points = 99
        M_values = np.ones(n_points)*M_min + np.logspace(np.log10(M_min)-3, np.log10(M_max - M_min), n_points)
        np.insert(M_values, 0, M_min)
        return M_values
        
    def alphas_UV_contour(self, LdQCD):
        M_min, M_max = self._M_bounds_pert_aUV(LdQCD)
        
        M_values = self._M_values(M_min, M_max)
        
        aUV_contour = np.array([self._alphas_UV(LdQCD, M).y.T[-1] for M in M_values])
        
        return aUV_contour
       
    @property
    def alphas_UV_low(self):
        if not hasattr(self, '_alphas_UV_low'):
            self._alphas_UV_low = self.alphas_UV_contour(cn.LdQCD_low)
        return self._alphas_UV_low
       
    @property
    def alphas_UV_high(self):
        if not hasattr(self, '_alphas_UV_high'):
            self._alphas_UV_high = self.alphas_UV_contour(cn.LdQCD_high)
        return self._alphas_UV_high
        
    @property
    def epsilon_v(self):
        if not hasattr(self, '_epsilon_v'):
            c_low = self.alphas_UV_low
            c_high = self.alphas_UV_high

            if np.argmax(c_low[0]) == np.argmax(c_high[0]):
                c_total = np.concatenate((c_low, np.flip(c_high, axis=0)))
            else:
                c_total = np.concatenate((c_low, np.flip(c_high, axis=0), 0.3*np.ones((1,2))))

            valid_region = Polygon(c_total)

            self._epsilon_v = valid_region.area/0.3**2
        
        return self._epsilon_v