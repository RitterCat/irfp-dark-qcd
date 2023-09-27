MZ = 91.2
mt = 172.76
mu_UV = 1e19
alpha_s_MZ = 0.11729
Nc = 3

LdQCD_low = 0.2
LdQCD_high = 5

SUN_irreps = {
    2: ((1, 0), (2, 1/2), (3, 2), (4, 5), (5, 10)),
    3: ((1, 0), (3, 1/2), (6, 5/2), (8, 3), (10, 15/2)),
    4: ((1, 0), (4, 1/2), (6, 1), (10, 3), (15, 4))
}

### CODE STORAGE ###
# This is the code that was used to calculate the pre-coefficients that are now stored on file

# def get_precoeffs(Nd):
#     vis_irreps = cn.SUN_irreps[3]
#     dark_irreps = cn.SUN_irreps[Nd]
    
#     Ag = -(11/3)*cn.Nc/(2*np.pi)
#     Bg = -(34/3)*cn.Nc**2/(8*np.pi**2)
#     Cg = 0
#     Dg = -(11/3)*Nd/(2*np.pi)
#     Eg = -(34/3)*Nd**2/(8*np.pi**2)
#     Fg = 0
    
#     gluon_coeffs = [Ag, Bg, Cg, Dg, Eg, Fg]
    
#     fermion_precoeffs = [np.zeros((5,5)) for _ in range(6)]
#     scalar_precoeffs = [np.zeros((5,5)) for _ in range(6)]
    
#     Af, Bf, Cf, Df, Ef, Ff = fermion_precoeffs
#     As, Bs, Cs, Ds, Es, Fs = scalar_precoeffs
    
#     for vis_idx in range(5):
#         for dark_idx in range(5):
#             d1, T1 = vis_irreps[vis_idx]
#             C1 = (cn.Nc**2 - 1)*T1/d1
#             d2, T2 = dark_irreps[dark_idx]
#             C2 = (Nd**2 - 1)*T2/d2
            
#             Af[vis_idx][dark_idx] += (2/3)*T1*2*d2/(2*np.pi)
#             Bf[vis_idx][dark_idx] += ((10/3)*cn.Nc + 2*C1)*T1*2*d2/(8*np.pi**2)
#             Cf[vis_idx][dark_idx] += 2*C2*T1*2*d2/(8*np.pi**2)
#             Df[vis_idx][dark_idx] += (2/3)*T2*2*d1/(2*np.pi)
#             Ef[vis_idx][dark_idx] += ((10/3)*Nd + 2*C2)*T2*2*d1/(8*np.pi**2)
#             Ff[vis_idx][dark_idx] += 2*C1*T2*2*d1/(8*np.pi**2)
            
#             As[vis_idx][dark_idx] += (1/3)*T1*d2/(2*np.pi)
#             Bs[vis_idx][dark_idx] += ((2/3)*cn.Nc + 4*C1)*T1*d2/(8*np.pi**2)
#             Cs[vis_idx][dark_idx] += 4*C2*T1*d2/(8*np.pi**2)
#             Ds[vis_idx][dark_idx] += (1/3)*T2*d1/(2*np.pi)
#             Es[vis_idx][dark_idx] += ((2/3)*Nd + 4*C2)*T2*d1/(8*np.pi**2)
#             Fs[vis_idx][dark_idx] += 4*C1*T2*d1/(8*np.pi**2)
            
#     return (fermion_precoeffs, scalar_precoeffs, gluon_coeffs)

# for Nd in [2,3,4]:
#     pc = get_precoeffs(Nd)
#     with open(f'precoeffs/pc_{Nd}', 'wb') as fp:
#         pkl.dump(pc, fp)

# def get_SUSY_precoeffs(Nd):
#     vis_irreps = cn.SUN_irreps[3]
#     dark_irreps = cn.SUN_irreps[Nd]

#     Ag = -3*cn.Nc/(2*np.pi)
#     Bg = -6*cn.Nc**2/(8*np.pi**2)
#     Cg = 0
#     Dg = -3*Nd/(2*np.pi)
#     Eg = -6*Nd**2/(8*np.pi**2)
#     Fg = 0
    
#     gluon_coeffs = [Ag, Bg, Cg, Dg, Eg, Fg]
    
#     multiplet_precoeffs = [np.zeros((5,5)) for _ in range(6)]
    
#     Am, Bm, Cm, Dm, Em, Fm = multiplet_precoeffs
    
#     for vis_idx in range(5):
#         for dark_idx in range(5):
#             d1, T1 = vis_irreps[vis_idx]
#             C1 = (cn.Nc**2 - 1)*T1/d1
#             d2, T2 = dark_irreps[dark_idx]
#             C2 = (Nd**2 - 1)*T2/d2
            
#             Am[vis_idx][dark_idx] += T1*d2/(2*np.pi)
#             Bm[vis_idx][dark_idx] += (2*cn.Nc + 4*C1)*T1*d2/(8*np.pi**2)
#             Cm[vis_idx][dark_idx] += 4*C2*T1*d2/(8*np.pi**2)
#             Dm[vis_idx][dark_idx] += T2*d1/(2*np.pi)
#             Em[vis_idx][dark_idx] += (2*Nd + 4*C2)*T2*d1/(8*np.pi**2)
#             Fm[vis_idx][dark_idx] += 4*C1*T2*d1/(8*np.pi**2)
            
#     return (multiplet_precoeffs, gluon_coeffs)

# for Nd in [2,3,4]:
#     pc_SUSY = get_SUSY_precoeffs(Nd)
#     with open(f'precoeffs/pc_SUSY_{Nd}', 'wb') as fp:
#         pkl.dump(pc_SUSY, fp)