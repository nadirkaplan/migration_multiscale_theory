import math as m
import numpy as np
import matplotlib.pyplot as plt
import User_functions_cellmigration2 as usf
from random import random
import pdb


'''
Backward Euler to update fraction/probablity of closed bond
'''
def closed_bond_prob(alpha, epsi, zeta, ks, kc, fa, fcr, fc, fs, f, rho_n, dt):
    #print('f', f)
    if fa > fcr:
        kon = alpha  + epsi*f  + zeta * np.exp((fa-fcr)/fs)  # + f 
    else:
        kon = alpha +  epsi*f
    koff = ks * np.exp(f/fs) + kc * np.exp(-f/fc)
    rho = (rho_n + kon*dt)/(1 + kon*dt + koff*dt)
    return rho


'''
Retrograde velocity calculation
'''
def Retrograde_velocity(F_st, F_stall, v0):
    vf = v0*(1-F_st/F_stall)
    #if vf < float(0.): vf = 0.
    #if vf > v0:  vf=v0
    return vf


'''
Update displacements of the substrate (xs) and actin bundle (xc)
'''
def update_disps(xs0, xc0, F_stall, F_p, beta, K_sub, v0, gama, dt):
    m11 = F_stall/v0 + beta*dt 
    m12 = - beta*dt 
    m21 = - beta*dt
    m22 = gama + K_sub*dt + beta*dt
    matrix = np.array([[m11, m12], [m21, m22]])
    r1 = dt*F_stall + F_stall*xc0/v0 + dt*F_p
    r2 = gama*xs0
    rside = np.array([r1, r2])
    xcs = np.linalg.solve(matrix, rside)
    return xcs


'''
calculate nucleus velocity based on traction force input and nucleus parameters
'''
def nucleus_velocity(pst_vx, pst_vy, Ftr, Force_ck, R_nuc, eta_nuc):
    Fx = (Ftr-Force_ck)*pst_vx;   Fy = (Ftr-Force_ck)*pst_vy;
    Fx_sum = np.sum(Fx);  Fy_sum = np.sum(Fy); 
    coef = 6*np.pi*R_nuc*eta_nuc ;
    Vnuc_x = Fx_sum/coef;  Vnuc_y = Fy_sum/coef;  
    return (Vnuc_x, Vnuc_y)


'''
calculate microtubule compressive force
<R> = [Rx, Ry] = [nm_pts_vx, nm_pts_vy] --- Direction vector
<R+> =[Ry, -Rx] = [nm_pts_vy, -nm_pts_vx] --- Orthogonal vector
'''
def microtubule_force(R_len, k_ck, Rcell):  
    alph = 0;
    nlen = np.size(R_len);  Force_ck=np.zeros(nlen);
    energ_ck = np.zeros(nlen);
    for i in range(nlen):
        stretch_change = (Rcell-R_len[i]);
        #Force_ck[i]= k_ck*stretch_change
        if stretch_change > 0:
            Force_ck[i]= k_ck*stretch_change;
        else:
            stretch = R_len[i]/Rcell;
            Force_ck[i]= -(k_ck*Rcell*(stretch-1) + alph*Rcell*(np.exp(usf.macualay(stretch-3))-1))
        energ_ck[i]= 0.50*k_ck*stretch_change**2.0
    ck_energy = np.sum(energ_ck)
    return (Force_ck, ck_energy)


'''
calculate membrane force and then get protrusion force based on equilibrium
<R> = [Rx, Ry] = [nm_pts_vx, nm_pts_vy] --- Direction vector
<R+> =[Ry, -Rx] = [nm_pts_vy, -nm_pts_vx] --- Orthogonal vector
'''
def membrane_protrusion_fv(vector_edge, k_mem, edge_L0, Force_ck,  eta_mem, vn, nm_pts_vx, nm_pts_vy):
    nlen = np.shape(vector_edge)[0];  protrusion_force=np.zeros([nlen,2]); lavg = np.zeros([nlen]);
    energ_mb = np.zeros([nlen]);
    for i in range(nlen):
        ext_disp1 = (vector_edge[i,0]-edge_L0); 
        ext_disp2 = (vector_edge[i,3]-edge_L0);
        if ext_disp1<0: ext_disp1=0.0; 
        if ext_disp2<0: ext_disp2=0.0; 
        protrusion_force[i,0]= k_mem*(ext_disp1*vector_edge[i,1]+ext_disp2*vector_edge[i,4]); 
        protrusion_force[i,1]= k_mem*(ext_disp1*vector_edge[i,2]+ext_disp2*vector_edge[i,5]);
        lavg[i] = (vector_edge[i,0] + vector_edge[i,3])/2;
        energ_mb[i] = 0.50*k_mem*ext_disp2**2.0
    # membrane force in polar and tagent directions 
    Fmemb_nm = protrusion_force[:,0]*nm_pts_vx+protrusion_force[:,1]*nm_pts_vy 
    Fmemb_tg = protrusion_force[:,0]*nm_pts_vy-protrusion_force[:,1]*nm_pts_vx 
    # protrusion forces in nucleus-membrane direction
    F_pro = eta_mem*vn*lavg-Fmemb_nm-Force_ck;    
    vt = Fmemb_tg/(eta_mem*lavg);
    memb_energy = np.sum(energ_mb)
    return (F_pro, vt, Fmemb_nm, Fmemb_tg, memb_energy) 


def Protrusion_force_correction(Force_pro, Force_ck, Fstall, Fsub):
    nlen = np.shape(Force_ck)[0];  
    for i in range(nlen):
        Fst = Fsub[i]-Force_pro[i]
        if Force_pro[i]>0 and Fst<0:
           Force_ck[i]=Force_ck[i]+(Force_pro[i]-Fsub[i]); Force_pro[i]=Fsub[i]
        elif Force_pro[i]<0 and Fst>Fstall[i]:
           Force_ck[i]=Fstall[i]-Fst; Force_pro[i]=Fsub[i] - Fstall[i];
    return (Force_pro, Force_ck)


'''
Area conservation forces
'''
def Area_conservation_force(Aa, A0, K_Area, dvd_vec_all):
    if Aa>A0:
        F_Area = 0.
    else:
        F_Area = K_Area*(A0 - Aa) #/A0
    F_Area_vec = F_Area*dvd_vec_all
    #print('F_Area', F_Area)
    #print('dvd_vec_all', dvd_vec_all)
    return F_Area_vec

