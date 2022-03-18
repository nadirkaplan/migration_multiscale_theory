import math as m
import numpy as np
import matplotlib.pyplot as plt
import User_functions_cellmigration1 as usf
import Traction as T
from random import random
from celluloid import Camera
from datetime import date
import pdb

'''
Uniform = 0 --> apolar chemical signaling,
duro = 1 --> gradient substrates. 
For gradient substrates 'Grad' must be set to a nonzero value (0.5 is used in main text). 
'''

def migration_simulator(uniform, duro, K_sub, gama_sub, Grad, nnum):

    dt = 0.000008 ;  #  step size

    Nsteps = int(3000000) ;  #  time steps
    Step_inv = int(10000);   
    gama_id = np.amax(gama_sub);
    ksub_id = np.amax(K_sub);
    
    Grad_id = np.amax(Grad);

    fname = 'Biphasic'+'_Uni'+str(uniform)+'_Ksub'+str(round(ksub_id*100)/100)+'_Gama'+str(round(gama_id*100)/100)
    
    Rcell = float(5.0)  
    R_nuc = float(2.0)
    shift_dist = float(0 ); 

    # define parameters of chemical modeling
    gama_r = float(0.3) # antagonistic effect of Rho to Rac
    gama_rho = float(0.3) # antagonistic effect of Rac to Rho 
    kb_plus = float(0.6)  # 2.4E-1 # Kp+  baseline Rac1 activation coefficient
    kb_minus = float(0.9); # 6.0E-1  # Kb-    baseline Rac1 inactivation coefficient
    kpolar_minus = float(0.0); #  # Kp-
    kapbb_plus = float(0.6);  #  # kap_p+  2.8E-1 # baseline Rho activation rate 
    kapbb_minus = float(0.9); #  # kap_b-  6.0E-1  # baseline Rho inactivation rate
    kap_polar_minus = float(0.0); #   # kap_p-
    beta_r = float(0.3); # 
    beta_rho = float(0.3); # 
    M_plus =  float(0.04); # float(0.1)  # Rac1 membrane association dissociation rates 
    M_minus = float(0.04); # float(0.05 )
    gg_val = float(0.0);  # float(0.08)
    mu_plus = float(0.04); # float(0.1)  # RhoA membrane association dissociation rates 
    mu_minus = float(0.04);  # float(0.05 )
    hh_val = float(0.0);  # float(0.08); 
    D = float(0.010);
    eta_memb = float(10)  # #membrane viscoelastic coefficient    pa*s = pn*s/um^2 =1e-6 pn*s/nm^2
    eta_nuc = float(800);  # pa*s = pn*s/um^2 =1e-6 pn*s/nm^2


    # define parameters of mechanics modeling
    alpha = float(5.0);
    kc = float(120.)
    ks = float(0.25)
    N_M = float(100 )  
    fs = float(1 )
    fm = float(2  )  #/fs
    epsilon = float(0.0 )  #/fs
    zeta = float(0.0)
    fc = float(0.50)      #*fs
    fcr = float(1.5)
    v0 = float(0.120)
    V_p = float(0.120)  #average polymerization rate


    N_C0 = float(100 ) 
    Ksub_ub = float(100)
    Gsub_ub = float(10)
    Ksub_lb = float(0.01)
    Gsub_lb = float(0.001)
    K_clutch = float(2.)
    fb = float(200.)
    K_memb = float(2.0) #membrane stiffness  pn/um  
    K_ck = float(20.0) #Microtubules stiffness pn/um   *Rcell
    Rac_cyto_tol = 1e-9

    # Dimensionless scale
    XX = 2*np.pi*Rcell;     #um
    TT = XX/v0; #s
    Km0 = 2.0;      #pN/um
    FF = Km0*XX;    #pN 

    #Dimesionalize all para
    gama_r_bar = gama_r*TT ;
    gama_rho_bar = gama_rho*TT ;
    beta_r_bar = beta_r*TT ;
    beta_rho_bar = beta_rho*TT ;

    kb_plus_bar = kb_plus*TT ;
    kb_minus_bar = kb_minus*TT ;
    kpolar_minus_bar = kpolar_minus*TT ;
    kap_polar_minus_bar = kap_polar_minus*TT ;
    kapbb_plus_bar = kapbb_plus*TT ;
    kapbb_minus_bar = kapbb_minus*TT ;

    M_plus_bar = M_plus*TT ;
    M_minus_bar = M_minus*TT ;
    mu_plus_bar = mu_plus*TT ;
    mu_minus_bar = mu_minus*TT ;
    D_bar = D/(XX*v0) ;

    Kon_0bar = alpha*TT ;
    epsilon_bar = epsilon*FF*TT;
    zeta_bar = zeta*TT ;
    kc_bar = kc*TT ;
    ks_bar = ks*TT ;
    fc_bar = fc/FF ;
    fs_bar = fs/FF ;
    fm_bar = fm/FF ;
    fcr_bar = fcr/FF ;
    v0_bar = v0*TT/XX ;
    K_clutch_bar = K_clutch/Km0*1e3 ;
    K_memb_bar = K_memb/Km0 ;

    Ksub_ub_bar = Ksub_ub/Km0*1e3 ;
    Gsub_ub_bar = Gsub_ub/(Km0*TT)*1e3 ;
    Ksub_lb_bar = Ksub_lb/Km0*1e3 ;
    Gsub_lb_bar = Gsub_lb/(Km0*TT)*1e3 ;

    K_ck_bar = K_ck/Km0;
    eta_memb_bar = eta_memb/Km0*v0 ;
    eta_nuc_bar = eta_nuc/Km0*v0 ;

    Rcell_bar = Rcell/XX ;
    R_nuc_bar = R_nuc/XX ; 
    vp_bar = V_p*TT/XX ;

    K_sub_bar = K_sub/Km0*1e3 ;
    gama_sub_bar = gama_sub/(Km0*TT)*1e3 ;
    Grad_bar = Grad/Km0*1e3 ;

    Ini_Raca = float(0.3)
    Ini_Raci = float(0.2)
    Ini_Rhoa = float(0.3)
    Ini_Rhoi = float(0.2)
    rac_0  = Ini_Raca/nnum
    rho_0  = Ini_Rhoa/nnum
    polar_num = int(nnum/4)
    #conv_iter = np.zeros([Nsteps, 2]);

    coords_scale = np.linspace(0.0, 1.0, nnum+1)

    xcell = Rcell_bar*np.cos(2*np.pi*coords_scale[0:nnum])
    ycell = Rcell_bar*np.sin(2*np.pi*coords_scale[0:nnum])
    edge_L0 = np.sqrt(2.*(Rcell_bar**2.)*(1 - np.cos(2*np.pi*coords_scale[1])))
    eglen_max = edge_L0;

    ''' 
    Pre-define values of chemo-singals 
    ''' 
    Raca_mb = np.zeros([nnum, Nsteps]) 
    Rhoa_mb = np.zeros([nnum, Nsteps]) 
    Raci_mb = np.zeros([nnum, Nsteps]) 
    Rhoi_mb = np.zeros([nnum, Nsteps]) 
    Rac_cyto = np.zeros([Nsteps]) 
    Rho_cyto = np.zeros([Nsteps]) 
    F_pro = np.zeros([nnum]); 
    R_len = Rcell_bar*np.ones([nnum]); 

    cycid=1;
    Ini_RacRho = usf.Initialize_RacRho(Ini_Raca, Ini_Raci, Ini_Rhoa, Ini_Rhoi, polar_num, nnum, uniform, cycid);
    Rac_a=Ini_RacRho[0];  Rac_i=Ini_RacRho[1];
    Rho_a=Ini_RacRho[2];  Rho_i=Ini_RacRho[3];

    N_C = N_C0 * np.ones(nnum) 
    nc = N_C ; 
    nm = N_M * np.ones(nnum) 
    Rc =   1- (np.sum(Rac_a) + np.sum(Rac_i))   #assign value 
    Rhoc = 1- (np.sum(Rho_a) + np.sum(Rho_i))   #assign value
    Raca_mb[:,0] = Rac_a
    Rhoa_mb[:,0] = Rho_a
    Raci_mb[:,0] = Rac_i
    Rhoi_mb[:,0] = Rho_i
    Rac_cyto[0] = Rc
    Rho_cyto[0] = Rhoc
    Avg_Rac1_mb = np.mean(Rac_a)
    Avg_Rhoa_mb = np.mean(Rho_a)


    #Pre-define mechanical responses
    #Vnuc_all = np.zeros([Nsteps,2]);
    xnuc_all = np.zeros([Nsteps]);  #Nucleus velocity and position vectors
    xnuc_all[0] = xnuc_all[0];
    ynuc_all = np.zeros([Nsteps]);
    xcen_all = np.zeros([Nsteps]);  #Centroid velocity and position vectors
    ycen_all = np.zeros([Nsteps]);

    Area = np.zeros([Nsteps]);  #Cell areas
    Ro_all = np.zeros([nnum, Nsteps])  #bonded clutch density
    N_clutch = np.zeros([nnum, Nsteps])  #Molecular clutch total numbers
    N_myosin = np.zeros([nnum, Nsteps]) #Myosin motor numbers
    xc_all = np.zeros([nnum, Nsteps]) #displacements of actin
    xs_all = np.zeros([nnum, Nsteps]) #displacements of substrate
    f_all = np.zeros([nnum, Nsteps]) #molecular clutch forces
    F_sub_all = np.zeros([nnum, Nsteps]) #Substrate forces
    Vr_all = np.zeros([nnum, Nsteps]) #Retrograde velocity
    Vs_all = np.zeros([nnum, Nsteps]) #Membrane spreading velocity
    Vp_all = np.zeros([nnum, Nsteps]) #Actin polymerization velocity

    xcoord_all = np.zeros([nnum+1, Nsteps]); xcoord_all[:nnum,0] = xcell; 
    ycoord_all = np.zeros([nnum+1, Nsteps]); ycoord_all[:nnum,0] = ycell;
    xcoord_all[nnum,0] = xcell[0];          ycoord_all[nnum,0] = ycell[0];
    nuc_mem_vx = xcoord_all[:nnum,0] - xnuc_all[0];  # direction vectors of every membrane pts 
    nuc_mem_vy = ycoord_all[:nnum,0] - ynuc_all[0];  # direction vectors from nucleus to membrane
    [nm_pts_vx, nm_pts_vy, R_len, Aa] = usf.Normalize_vector(nuc_mem_vx, nuc_mem_vy) #normalized position vectors
    Area[0] = Aa;     A0 = Aa;     
    edge_len_all = np.zeros([nnum, Nsteps]);
    edge_ang_all = np.zeros([nnum, Nsteps]);   # vertex angles of every time step

    ang_diff_all = np.zeros([nnum]);
    edge_ang_avg = np.zeros([nnum, int(Nsteps/Step_inv)+1]);  # average edge angles for every 'step_inv' steps
    avgang_diff_all = np.zeros([nnum, int(Nsteps/Step_inv)+1]);  #average angle differences between adjacent 'step_inv' steps

    cr_all = np.zeros([nnum, int(Nsteps/Step_inv)+1]);
    dr_all = np.zeros([nnum, int(Nsteps/Step_inv)+1]);
    F_pro_all= np.zeros([nnum, Nsteps]);

    F_ck_all= np.zeros([nnum, Nsteps]);
    F_st_all= np.zeros([nnum, Nsteps]);
    strain_energy= np.zeros([3, Nsteps]);
    Fstall=np.zeros([nnum]);


    '''
    Calculate and assemble global derivative matrix
    '''
    vec_edges=usf.edge_vectors(xcell, ycell);  #get edge length and direction vectors

    edge_ang_all[:,0]=usf.edge_angles(vec_edges, nnum);
    Angle0 = edge_ang_all[0,0];
    edge_len = usf.edge_length(xcell, ycell)
    edge_len_all[:,0] = edge_len[0]

    tm = float(0.0)
    time = np.zeros([Nsteps] )
    RacRho_ini = np.zeros([4*nnum + 2] )
    RacRho_last= np.zeros([4*nnum + 2] )
    RacRho_current= np.zeros([4*nnum + 2])
    RacRho_inputs= np.zeros([4*nnum + 2] )
    Step_inv_no = 0;      steady_step_num = np.zeros([1000]);  
    xcoord=xcoord_all[:, 0];       ycoord=ycoord_all[:, 0];

    for i in range(Nsteps-1):
        if i%10000<1e-3: print('Nsteps=', i)
        if i%Step_inv<1e-3 and i>0:   # get angle differences between different vertices every Step_inv=1e4 steps
            step_bn=Step_inv_no*Step_inv;   Step_inv_no += 1;   step_en=Step_inv_no*Step_inv-1;
            for nn in range(nnum):
                edge_ang_avg[nn, Step_inv_no] = np.average(edge_ang_all[nn,step_bn:step_en]);
            avgang_diff_all[:,Step_inv_no] = (edge_ang_avg[:,Step_inv_no]-edge_ang_avg[:,Step_inv_no-1]);
        
        if i%Step_inv<1e-3 and Step_inv_no>1:   # check conditions in Eq.s1 and s2 to update A_r and A_rho
            avgang_diff0 = np.zeros(nnum);      # maximum angle changes every 'Step_inv' steps
            for ij in range(nnum): 
                avgang_diff0[ij] = np.amax(np.absolute(avgang_diff_all[ij,0:Step_inv_no]));   
            avgang_diff = (edge_ang_avg[:,Step_inv_no]-edge_ang_avg[:,Step_inv_no-1]);
            polar_or_contract = (edge_ang_avg[:,Step_inv_no] - Angle0);  # + in contraction; - in protrusion

        tm += dt
        time[i+1]= tm
    
        if i<1: Rac_cyto_diff = 1;
        else:   Rac_cyto_diff = abs(Rac_cyto[i] - Rac_cyto[i-1]); #Rac_cyto[i-2]);

        if Rac_cyto_diff < Rac_cyto_tol:   # reintialize Rac1 and RhoA once reached steady state     
            Ini_RacRho = usf.Initialize_RacRho(Ini_Raca, Ini_Raci, Ini_Rhoa, Ini_Rhoi, polar_num, nnum, uniform, cycid);
            Raca_mb[:,i]=Ini_RacRho[0];  Raci_mb[:,i]=Ini_RacRho[1];
            Rhoa_mb[:,i]=Ini_RacRho[2];  Rhoi_mb[:,i]=Ini_RacRho[3];
            Rac_cyto[i] = 1- (np.sum(Raca_mb[:,i]) + np.sum(Raci_mb[:,i]))   #assign value 
            Rho_cyto[i] = 1- (np.sum(Rhoa_mb[:,i]) + np.sum(Rhoi_mb[:,i]))   #assign value
            steady_step_num[cycid] = i;   cycid += 1;
       
        RacRho_last = usf.Assemble_RacRho(Raca_mb[:,i], Raci_mb[:,i], Rhoa_mb[:,i],  \
                                  Rhoi_mb[:,i], Rac_cyto[i], Rho_cyto[i], nnum)
        rac_an = Raca_mb[:,i];   rac_in = Raci_mb[:,i];
        rho_an = Rhoa_mb[:,i];   rho_in = Rhoi_mb[:,i];
        Rcn = Rac_cyto[i];       Rhocn = Rho_cyto[i];
        RacRho_current = RacRho_last ;  # assign last step values as the current step trial values 
        rac_a = rac_an;   rac_i = rac_in;
        rho_a = rho_an;   rho_i = rho_in;
        Rc = Rcn;         Rhoc = Rhocn;

    
        '''
        Bio-chemical model  ----  update Rac and Rho signaling  
        '''
        tol = 1E-9
        maxiter = 20 
        eps = 1E3*tol 
        iter = int(0) 
        while eps > tol and iter < maxiter:
            iter += 1
            RacRho_ini = usf.Assemble_RacRho(rac_an, rac_in, rho_an, rho_in, Rcn, Rhocn, nnum);
            K_plus = np.zeros([nnum,3]);    K_minus = np.zeros([nnum]);
            kappa_p = np.zeros([nnum,3]);   kappa_m = np.zeros([nnum]);
            for j in range(nnum):
                rac_aj = rac_a[j];     rac_ij = rac_i[j];
                rho_aj = rho_a[j];     rho_ij = rho_i[j];
                Rj_len = R_len[j];     ncj = nc[j];    nmj = nm[j];
                vsj = Vs_all[j, i];    angj = edge_ang_all[j,i];
                ang_diffj = ang_diff_all[j];
                if Step_inv_no<=1:
                    cr = 1.0; dr = 1.0;
                    cr_all[j,Step_inv_no]=cr;   dr_all[j,Step_inv_no]=dr; 
                elif i%Step_inv<1e-3 and Step_inv_no>1:
                    avgang_diff_tol = np.absolute(avgang_diff);
                    avgang_diff0_tol = np.absolute(avgang_diff0);
                    # when (1) angle change rate due to protrusion/contraction becomes low & (2) angle change exceed threshold
                    if avgang_diff_tol[j]<0.3*avgang_diff0_tol[j] and polar_or_contract[j]>0.3*Angle0:
                        coeff_cr=np.absolute(2*ang_diffj)/Angle0;   cr = np.exp(coeff_cr);
                    elif avgang_diff_tol[j]<0.3*avgang_diff0_tol[j] and polar_or_contract[j]<-0.3*Angle0:
                        coeff_dr=np.absolute(2*ang_diffj)/Angle0;   dr = np.exp(coeff_dr);
                    else:
                        cr = 1.0; dr = 1.0;
                    cr_all[j,Step_inv_no]=cr;   dr_all[j,Step_inv_no]=dr; 
                cr = cr_all[j,Step_inv_no];     dr = dr_all[j,Step_inv_no]
                K_plus[j,:]  = usf.Rac1_activation(kb_plus_bar, gama_r_bar, beta_r_bar, rho_aj, rac_aj, rac_0, rho_0, cr, norder = 3.0)
                K_minus[j] = usf.Rac1_inactivation(kb_minus_bar, kpolar_minus_bar, angj, Angle0)
                kappa_p[j,:] = usf.RhoA_activation(kapbb_plus_bar, gama_rho_bar, beta_rho_bar, rac_aj, rho_aj, rac_0, rho_0, dr, norder = 3.0)
                kappa_m[j] = usf.RhoA_inactivation(kapbb_minus_bar, kap_polar_minus_bar, angj, Angle0)
            matrix_res = usf.global_matrix_residue(rho_a, rac_a, rho_i, rac_i, rho_an, rac_an,  \
                     rho_in, rac_in, K_plus, K_minus, kappa_p, kappa_m, M_plus_bar, M_minus_bar, \
                     mu_plus_bar, mu_minus_bar, D_bar, edge_len, Rc, Rhoc, Rcn, Rhocn, nnum, dt ) 
            Delta_RacRho = - np.linalg.solve(matrix_res[0], matrix_res[1])
            eps = np.linalg.norm(Delta_RacRho)
            RacRho_current += Delta_RacRho
            RacRho_inputs = usf.Disassemble_RacRho(RacRho_current, nnum)
            rac_a = RacRho_inputs[0];   rac_i = RacRho_inputs[1];
            rho_a = RacRho_inputs[2];   rho_i = RacRho_inputs[3];
            Rc = RacRho_inputs[4];      Rhoc = RacRho_inputs[5];
        
        RacRho_pts = usf.Disassemble_RacRho(RacRho_current, nnum) 
        Raca_mb[:,i+1] = RacRho_pts[0];    Raci_mb[:,i+1] = RacRho_pts[1];
        Rhoa_mb[:,i+1] = RacRho_pts[2];    Rhoi_mb[:,i+1] = RacRho_pts[3];
        Rac_cyto[i+1] = RacRho_pts[4];     Rho_cyto[i+1] = RacRho_pts[5];
        if iter >= maxiter-1: print('Cannot converge at step =',i); break

        #update/calculate N_clutch and N_myosin based on chemical signaling
        Avg_Rac1_mb = np.mean(RacRho_pts[0]);  
        Avg_Rhoa_mb = np.mean(RacRho_pts[2]);  
        nc = RacRho_pts[0]/Avg_Rac1_mb*N_C;   
        nm = RacRho_pts[2]/Avg_Rhoa_mb*N_M;
        vp = RacRho_pts[0]/Avg_Rac1_mb*vp_bar;   
        nc = np.minimum(nc, 3*N_C);     vp = np.minimum(vp, 3*vp_bar);      nm = np.minimum(nm, 3*N_M);   
        nc = np.maximum(nc, 0.3*N_C);   vp = np.maximum(vp, 0.3*vp_bar);    nm = np.maximum(nm, 0.3*N_M);
        N_clutch[:, i+1] = nc;          Vp_all[:,i+1] = vp;                 N_myosin[:, i+1] = nm;
    
    
        '''
        Now introduce mechanics  --- update displacements
        '''
        disp_tol = 1e-3; iteration =0;
        xcoord=xcoord_all[:, i];       ycoord=ycoord_all[:, i];
        xcoord0=xcoord_all[:, i];      ycoord0=ycoord_all[:, i];
        fbond = np.zeros([nnum]);      F_pro = F_pro_all[:,i];
        xc0_pts = xc_all[:,i];         xs0_pts = xs_all[:,i];
        Fst_val = F_st_all[:,i];       Fsub_val = F_sub_all[:,i];  

        while disp_tol>1e-5 and iteration < 3: 
            iteration += 1;
            for j in range(nnum):
                Ro_0 = Ro_all[j,i];  xc0 = xc_all[j,i];  xs0 = xs_all[j,i];
                f = f_all[j, i];
                xcj = xcoord_all[j, i];  ycj = ycoord_all[j, i];  
                Ro = T.closed_bond_prob(Kon_0bar, ks_bar, kc_bar, fcr_bar, fc_bar, fs_bar, f, Ro_0, dt);
                if Ro < 1e-6 or f > fb:
                    f =0.; Ro = 0.; 
                    xc = 0.; xs = 0.; 
                else:
                    beta = Ro*nc[j]*K_clutch_bar;      F_stall = nm[j]*fm_bar;       F_p = F_pro[j]; 
                    k_gama = usf.sub_stiff_visco(xcj, ycj, Rcell_bar, K_sub_bar, Grad_bar, gama_sub_bar, \
                                             Ksub_ub_bar, Gsub_ub_bar, Ksub_lb_bar, Gsub_lb_bar, duro);
                    xc_xs=T.update_disps(xs0, xc0, F_stall, F_p, beta, k_gama[0],  v0_bar, k_gama[1], dt);
                    f = K_clutch_bar*(xc_xs[0] - xc_xs[1]); 
                    xc = xc_xs[0]; xs = xc_xs[1]; 
                Ro_all[j, i+1] = Ro;  xc_all[j, i+1] = xc;  xs_all[j, i+1] = xs; 
                Fsub = Ro*nc[j]*f;  
                F_st = Fst_val[j];  
                vf =  T.Retrograde_velocity(F_st, F_stall, v0_bar);
                fbond[j] = f;   Fsub_val[j] = Fsub;
                Fstall[j]= F_stall;
                Vr_all[j, i+1] = vf;         Vs_all[j, i+1] = Vp_all[j,i+1] - vf;
                    
            #update membrane (xcoord_all) shape and outward-direction vector (vec)
            Vs=Vs_all[:,i+1];   
            xycoord=usf.coord_direction_update(Vs, xcoord_all[:, i], ycoord_all[:, i], nm_pts_vx, nm_pts_vy, dt, nnum);
            xcoord = xycoord[0];    ycoord = xycoord[1]; 
            Force_ck = T.microtubule_force(R_len, K_ck_bar, Rcell_bar);  # calculate cytoskeleton forces
            F_ck_all[:,i+1]= Force_ck[0]; 
            vec_edges=usf.edge_vectors(xcoord[:nnum], ycoord[:nnum]);
            edge_ang=usf.edge_angles(vec_edges, nnum); edge_ang_all[:,i+1] = edge_ang;
            ang_diff_all= usf.angle_diff(edge_ang, nnum);
        

            Force_Vm=T.membrane_protrusion_fv(vec_edges, K_memb_bar, edge_L0, Force_ck[0],  eta_memb_bar, Vs, nm_pts_vx, nm_pts_vy);
            F_pro=Force_Vm[0];      F_pro_all[:,i+1]= F_pro;          
            Vm = Force_Vm[1];
            xycoord=usf.coord_direction_update(Vm, xcoord, ycoord, nm_pts_vy, -nm_pts_vx, dt, nnum);
            xcoord = xycoord[0];   ycoord = xycoord[1];
    
            #update nucleus velocity based on F_sub;
            Fst_val = (Fsub_val[:nnum] - F_pro);
            V_nuc = T.nucleus_velocity(nm_pts_vx, nm_pts_vy, Fst_val, Force_ck[0], R_nuc_bar, eta_nuc_bar);
            xnuc_all[i+1] = xnuc_all[i] + dt*V_nuc[0] ;
            ynuc_all[i+1] = ynuc_all[i] + dt*V_nuc[1] ;

            nuc_mem_vx = xcoord[:nnum] - xnuc_all[i+1];  # direction vectors of every membrane pts 
            nuc_mem_vy = ycoord[:nnum] - ynuc_all[i+1];  # direction vectors from nucleus to membrane
            [nm_pts_vx, nm_pts_vy, R_len, Aa] = usf.Normalize_vector(nuc_mem_vx, nuc_mem_vy) #normalized position vectors
            Area[i+1] = Aa;   
            disp = np.sqrt((xcoord-xcoord0)**2 + (ycoord-ycoord0)**2) ;
            xcoord0 = xycoord[0];   ycoord0 = xycoord[1];
            disp_tol = abs(np.sum(disp)); 

        strain_energy[0, i+1] = Force_ck[1];
        strain_energy[1, i+1] = Force_Vm[4];
        strain_energy[2, i+1] = Force_ck[1] + Force_Vm[4];
        centroid = usf.centroid_cell(xycoord[0], xycoord[1], xnuc_all[i+1], ynuc_all[i+1], nnum);
        xcen_all[i+1] = centroid[0];   ycen_all[i+1] = centroid[1];   
        xcoord_all[:, i+1] = xycoord[0];    ycoord_all[:, i+1] = xycoord[1];
        edge_len = usf.edge_length(xcoord_all[0:nnum, i+1], ycoord_all[0:nnum, i+1]);
        if np.amax(edge_len[0])>eglen_max:    eglen_max = np.amax(edge_len[0]);
        f_all[:, i+1] = fbond;   #F_pro_all[:, i+1] = F_pro;

        F_st_all[:, i+1] = Fst_val;
        F_sub_all[:, i+1] = Fsub_val;
        #conv_iter[i,0] = tm;
        #conv_iter[i,1] = iteration;

    delta_x = np.zeros([cycid]);  #get nucleus displacements of every Rac-Rho cycle 
    for ii in range(cycid):   
        step_id1 = int(steady_step_num[ii]); step_id2 = int(steady_step_num[ii+1]);
        dist_xnuc = xnuc_all[step_id2] - xnuc_all[step_id1];
        delta_x[ii] = dist_xnuc;
    
    pindx = np.where(delta_x[0:cycid-1]>0);  #have to run long time simulation to get valid solutions
    nindx = np.where(delta_x[0:cycid-1]<0);
    pdisp = np.sum(delta_x[pindx]);
    ndisp = np.sum(delta_x[nindx]);
    abs_disp = np.sum(abs(delta_x[0:cycid-1]));
    positive_indx = pdisp/abs_disp;
    negative_indx = ndisp/abs_disp;
    #
    
    Area_avg = np.mean(Area[int(Nsteps*0.8):Nsteps]);

    x_delta = xnuc_all[1:Nsteps] - xnuc_all[0:Nsteps-1];
    y_delta = ynuc_all[1:Nsteps] - ynuc_all[0:Nsteps-1];
    len_delta = np.sqrt((x_delta**2 + y_delta**2));
    traj_length = np.sum(len_delta)
    avg_vel = traj_length/time[Nsteps-1]
    
    #sum_avg_eng = np.mean(strain_energy[2,int(Nsteps*0.9):Nsteps]);   
    #print('sum of strain energies',sum_avg_eng) # note that primary force scale used is F=Nm*fm in paper, while F=Km*XX is used. Conversion required

    write_to_file_ary = np.array([np.amax(K_sub), np.amax(gama_sub), avg_vel]);
    #write_to_file_ary = np.array([np.amax(K_sub), np.amax(gama_sub), np.amax(Grad),  avg_vel,  positive_indx, negative_indx]);
    file1 = open("solution.txt", "a");  
    np.savetxt(file1, [write_to_file_ary],  fmt='%.6e', delimiter='\t');
    file1.close()  #.transpose()      .transpose(), newline=" " \n

    savegifname = fname + '.gif'


    plt.rcParams['font.size'] = '16'
    fig = plt.figure(10);    
    title_name = r'$K_{sub}=$' + str(round(ksub_id*100)/100) + r' $pN/nm$'
    plt.title(title_name)     #Cell migration on substrate with 
    camera = Camera(fig)
    for ii in range(0, Nsteps, 4000):   plt.plot(xcoord_all[:, ii], ycoord_all[:, ii],'b-',lw=2.5);       draw_circle = plt.Circle((xnuc_all[ii], ynuc_all[ii]), 2.0/XX, fill=True, color='0.5');   plt.gcf().gca().add_artist(draw_circle);    tim = 't/T='+str(np.round(100*time[ii])/100);  plt.xlim([-0.4, 1.6]);  plt.text(0.5, -0.6, tim, fontsize=24);    plt.gca().set_aspect('equal');    plt.axis('off');       camera.snap();  
        
    animation = camera.animate(interval = 100, repeat = True,repeat_delay = 500)
    animation.save(savegifname) #,writer='Pillow', fps=2

    plt.show()

    '''
    #plt.style.use('seaborn-pastel')
    today = date.today()
    dd = '01' #today.strftime("%y_%m_%d")

    filetarray=fname+'_tarray_'+dd+'.dat'
    filemembX=fname+'_Xmemb_'+dd+'.dat'
    filemembY=fname+'_Ymemb_'+dd+'.dat'
    filenucX=fname+'_Xnuc_'+dd+'.dat'
    filenucY=fname+'_Ynuc_'+dd+'.dat'
    #filecenX=fname+'_Xcen_'+dd+'.dat'
    #filecenY=fname+'_Ycen_'+dd+'.dat'


    np.savetxt(filetarray, time.transpose(),  fmt='%.6e', delimiter='\t');
    np.savetxt(filenucX, xnuc_all.transpose(),  fmt='%.6e', delimiter='\t');
    np.savetxt(filenucY, ynuc_all.transpose(),  fmt='%.6e', delimiter='\t');
    #np.savetxt(filecenX, xcen_all.transpose(),  fmt='%.6e', delimiter='\t');
    #np.savetxt(filecenY, ycen_all.transpose(),  fmt='%.6e', delimiter='\t');
    '''
    
    #pdb.set_trace()
    
    return fname    




uniform = int(5); 
duro = 0;

nnum = int(16) ;

Grad = float(0.00)

ksub_ary = np.array([3.0]);   #np.array([0.01, 0.1, 1, 3, 5, 7, 9, 11, 13, 15, 20, 30, 50, 100])*16/nnum;
gama_ary = np.array([0.1]);

ki = np.size(ksub_ary); gi = np.size(gama_ary); 

for ii in range(ki):
    K_sub = ksub_ary[ii];
    for jj in range(gi):
        gama_sub = gama_ary[jj];
        flog = migration_simulator(uniform, duro, K_sub, gama_sub, Grad, nnum)
