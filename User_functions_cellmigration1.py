'''
Define functions for the cell migration discrete modeling 
'''
import math as m
import numpy as np
from random import random
import pdb

'''
Calculate edge angle difference w.r.t. close two angles
'''
def angle_diff(edge_angles, nnum):
    delta_angles = np.zeros(nnum);
    for jj in range(nnum-1):
        delta_angles[jj] = edge_angles[jj] - (edge_angles[jj-1]+edge_angles[jj+1])/2.
    delta_angles[nnum-1]=edge_angles[nnum-1] - (edge_angles[nnum-2]+edge_angles[0])/2.
    return delta_angles

'''
assemble solutions of Rac and Rho to the global vector
'''
def Assemble_RacRho(Rac_a, Rac_i, Rho_a, Rho_i, Rc, Rhoc, Npts):
    RacRho = np.zeros([4*Npts+2])
    for i in range(Npts):
        RacRho[4*i] = Rac_a[i]
        RacRho[4*i+1] = Rac_i[i]
        RacRho[4*i+2] = Rho_a[i]
        RacRho[4*i+3] = Rho_i[i]
    RacRho[4*Npts] = Rc
    RacRho[4*Npts+1] = Rhoc
    return (RacRho)



'''
calculate centroid of the cell
'''
def centroid_cell(x_coord, y_coord, x_nuc, y_nuc, nnum):
    tri_croid = np.zeros([nnum,2]);
    tri_area = np.zeros([nnum]);
    tri_pt3 = np.array([x_nuc, y_nuc]);
    for i in range(nnum):
        tri_pt1 = np.array( [x_coord[i], y_coord[i]]);
        tri_pt2 = np.array( [x_coord[i+1], y_coord[i+1]]);
        tri_area[i] = abs(tri_pt1[0]*(tri_pt2[1] - tri_pt3[1]) \
                + tri_pt2[0]*(tri_pt3[1] - tri_pt1[1]) \
                + tri_pt3[0]*(tri_pt1[1] - tri_pt2[1]))/2. ;
        tri_croid[i, 0] = (tri_pt1[0] + tri_pt2[0] + tri_pt3[0])/3.;
        tri_croid[i, 1] = (tri_pt1[1] + tri_pt2[1] + tri_pt3[1])/3.;
    area_sum = np.sum(tri_area);
    tri_mx = tri_area * tri_croid[:,0];
    tri_my = tri_area * tri_croid[:,1];
    moment_x = np.sum(tri_mx);
    moment_y = np.sum(tri_my);
    cent_x = moment_x/area_sum;
    cent_y = moment_y/area_sum;
    return (cent_x, cent_y)
    
            
  
'''
Update membrane coordinates 
'''
def coord_direction_update(vs, xcoord, ycoord, vec_x, vec_y, dt, nnum):
    coord_updated = np.zeros([nnum+1,2]);
    Delta_Radius = vs*dt;   
    Delta_dx = vec_x*Delta_Radius;    Delta_dy = vec_y*Delta_Radius;  
    coord_updated[:nnum,0] = xcoord[:nnum] + Delta_dx;       
    coord_updated[:nnum,1] = ycoord[:nnum] + Delta_dy;
    coord_updated[nnum,0] = coord_updated[0, 0];  coord_updated[nnum,1] = coord_updated[0, 1];
    return (coord_updated[:,0], coord_updated[:,1])


'''
disassemble the global vector to solutions of Rac and Rho 
'''
def Disassemble_RacRho(RacRho, Npts):
    Rac_a = np.zeros([Npts])
    Rac_i = np.zeros([Npts])
    Rho_a = np.zeros([Npts])
    Rho_i = np.zeros([Npts])
    Rc = 0.0;   Rhoc = 0.0;
    for i in range(Npts):
        Rac_a[i] = RacRho[4*i] 
        Rac_i[i] = RacRho[4*i+1] 
        Rho_a[i] = RacRho[4*i+2] 
        Rho_i[i] = RacRho[4*i+3] 
    Rc = RacRho[4*Npts] 
    Rhoc = RacRho[4*Npts+1]
    return (Rac_a, Rac_i, Rho_a, Rho_i, Rc, Rhoc)


'''
Compute diffusive flux from every pt --- J_{i-1} - J_{i}
'''
def diffusive_flux_J(edge_len, RacRho, D, dt ):  #xcoord, ycoord
    nlen = len(RacRho)
    LL = edge_len[0]
    l_l = edge_len[1]
    if nlen >= 3:
        avg_RacRho = RacRho/l_l
        avg_RacRho_mod = np.concatenate([avg_RacRho[1:nlen], avg_RacRho[0:1]])
        diff_RacRho = avg_RacRho_mod - avg_RacRho
        rate_RacRho = -D*diff_RacRho/LL
    else: rate_RacRho = np.zeros([nlen])
    rate_RacRho_mod = np.concatenate([rate_RacRho[nlen-1:nlen], rate_RacRho[0:nlen-1]])
    return -(rate_RacRho_mod - rate_RacRho)*dt


'''
Compute Jacobian matrix of the diffusion flux term 
'''
def derivative_J(edge_len, D, idx, jdx, Npts, dt):  #xcoord, ycoord
    LL = edge_len[0]
    l_l = edge_len[1]    
    if Npts > 3:
        if idx == int(0):
            Li = LL[idx]; Li_m1 = LL[Npts-1]; Li_p1 = LL[idx+1];
            lli = l_l[idx]; lli_m1 = l_l[Npts-1]; lli_p1 = l_l[idx+1];
        elif idx == int(Npts - 1):
            Li = LL[idx]; Li_m1 = LL[idx-1]; Li_p1 = LL[0];
            lli = l_l[idx]; lli_m1 = l_l[idx-1]; lli_p1 = l_l[0];
        else:
            Li = LL[idx]; Li_m1 = LL[idx-1]; Li_p1 = LL[idx+1];
            lli = l_l[idx]; lli_m1 = l_l[idx-1]; lli_p1 = l_l[idx+1];
            #
        if idx == jdx:
            kvalue = -D/lli * (1/Li_m1 + 1/Li)
        elif jdx == idx + 1:
            kvalue = D/(Li * lli_p1)
        elif jdx == idx - 1:
            kvalue = D/(Li_m1 * lli_m1)
        else: kvalue = 0.0
        ksub_matrix = -kvalue*np.identity(4)*dt
    else: ksub_matrix = np.zeros([4,4])
    return ksub_matrix

'''
Assemble derivative matrix   
Npi --- row number of the submatrix in the global one
Npj --- column number of the submatrix in the global one
'''
def derivative_matrix(kmatrix_node,  Npi, Npj, Npts):
    K_matrix_total = np.zeros(4*Npts, 4*Npts)
    K_matrix_total[4*Npi:4*Npi+4, 4*Npj:4*Npj+4] = kmatrix_node
    return K_matrix_total



'''
devided angle vector between two membrane vectors
'''
def devided_angle_vector(vec_edges, nm_pts_vx, nm_pts_vy):   
    nlen = np.shape(vec_edges)[0];  dvd_vec_all = np.zeros([nlen,2]); 
    for j in range(nlen):
        dvd_vec = np.zeros([2]);  vec_dvd = np.zeros([2]);
        vec_edge1 = vec_edges[j, 1:3];  vec_edge2 = vec_edges[j, 4:6];
        nm_vecx = nm_pts_vx[j];         nm_vecy = nm_pts_vy[j]; 
        if abs(vec_edge1[0]-vec_edge2[0])<1e-6:
           dvd_vec[0] = 1;   dvd_vec[1] = 0;
        elif abs(vec_edge1[1]-vec_edge2[1])<1e-6:
           dvd_vec[0] = 0;   dvd_vec[1] = 1;
        else:
           y_coef = ((vec_edge2[1]-vec_edge1[1])/(vec_edge1[0]-vec_edge2[0])); 
           dvd_vec[1] = np.sqrt(1/(y_coef**2 + 1));
           dvd_vec[0] = y_coef * dvd_vec[1];
        outward_check = dvd_vec[0]*nm_vecx + dvd_vec[1]*nm_vecy;
        if outward_check > 0: 
            vec_dvd[0] = outward_check;
            vec_dvd[1] = dvd_vec[0]*nm_vecy - dvd_vec[1]*nm_vecx;
        else:
            vec_dvd[0] = -outward_check;
            vec_dvd[1] = -(dvd_vec[0]*nm_vecy - dvd_vec[1]*nm_vecx);
        dvd_vec_all[j,:] = vec_dvd;
    return dvd_vec_all           


'''
Numerically calculate edge length based on coords
'''
def edge_length(xcoord, ycoord):
    nlen = xcoord.size
    xcm = np.concatenate([xcoord[1:nlen], xcoord[0:1]])
    ycm = np.concatenate([ycoord[1:nlen], ycoord[0:1]])
    edge_length = np.sqrt((xcm-xcoord)**2.0 + (ycm-ycoord)**2.0)
    edge_len_mod = np.concatenate([edge_length[nlen-1:nlen], edge_length[0:nlen-1]])
    half_edge_len = edge_length/2 + edge_len_mod/2
    return (edge_length, half_edge_len)

'''
Numerically calculate edge length and vectors based on coords
           2     [l_12, v12_x, v12_y, l_14, v14_x, v14_y]
         /   \   [l_23, v23_x, v23_y, l_21, v21_x, v21_y]
        3     1  [l_34, v34_x, v34_y, l_32, v32_x, v32_y]
         \   /   [l_41, v41_x, v41_y, l_43, v43_x, v43_y]
           4
'''    
def edge_vectors(xcoord, ycoord):
    nlen = xcoord.size
    vector_edge=np.zeros([nlen,6]);  vec_clockws=np.zeros([nlen,2])
    xcm = np.concatenate([xcoord[1:nlen], xcoord[0:1]])
    ycm = np.concatenate([ycoord[1:nlen], ycoord[0:1]])
    vec_counter= (xcm-xcoord, ycm-ycoord)  #1->2->3->4->1
    xci = np.concatenate([xcoord[nlen-1:nlen], xcoord[0:nlen-1]])
    yci = np.concatenate([ycoord[nlen-1:nlen], ycoord[0:nlen-1]])  
    vec_clockws = (xci-xcoord, yci-ycoord)  #1->4  2->1  3->2  4->3   
    edge_len1 = np.sqrt((vec_counter[0])**2.0 + (vec_counter[1])**2.0)
    edge_len2 = np.sqrt((vec_clockws[0])**2.0 + (vec_clockws[1])**2.0)
    vector_edge[:,0]=edge_len1
    vector_edge[:,1]=vec_counter[0]/edge_len1
    vector_edge[:,2]=vec_counter[1]/edge_len1
    vector_edge[:,3]=edge_len2
    vector_edge[:,4]=vec_clockws[0]/edge_len2
    vector_edge[:,5]=vec_clockws[1]/edge_len2
    return (vector_edge)


'''
Calculate angles between connected cell edges
'''
def edge_angles(vector_edge, nnum):
    edge_angles = np.zeros(nnum);
    for ii in range(nnum):
        angle1 = m.atan2(vector_edge[ii,5], vector_edge[ii,4]);
        angle2 = m.atan2(vector_edge[ii,2], vector_edge[ii,1]);
        angle = angle2 - angle1;
        if angle>float(0): ang = 2*np.pi - angle;
        else: ang = -angle
        edge_angles[ii] = ang;
    return edge_angles


'''
Function of assembling global derivative matrix and residues 
'''
def global_matrix_residue(Rho_a, Rac_a, Rho_i, Rac_i, Rho_an, Rac_an, Rho_in, \
                 Rac_in, K_plus, K_minus, kappa_p, kappa_m, M_plus, M_minus, \
                 mu_plus, mu_minus, D, edge_len, Rc, Rhoc, Rcn, Rhocn, Npts, dt ):
    K_matrix_total = np.zeros( [4*Npts + 2, 4*Npts + 2] );
    K_matrix_total[4*Npts, 4*Npts] = 1.0;
    K_matrix_total[4*Npts+1, 4*Npts+1] = 1.0;
    Residue = np.zeros([4*Npts + 2] )
    Rc_eqsecterm = 0.0 
    Rhoc_eqsecterm = 0.0
    #print('Rac_a', Rac_a)
    #print('Rac_i', Rac_i)
    Jflux_Raca = diffusive_flux_J(edge_len, Rac_a, D, dt )
    Raci_Jflux = diffusive_flux_J(edge_len, Rac_i, D, dt )
    Jflux_Rhoa = diffusive_flux_J(edge_len, Rho_a, D, dt )
    Rhoi_Jflux = diffusive_flux_J(edge_len, Rho_i, D, dt )
    #print('Jflux_Raca',Jflux_Raca)
    Jflux_term = Assemble_RacRho(Jflux_Raca, Raci_Jflux, Jflux_Rhoa, Rhoi_Jflux, 0.0, 0.0, Npts)
    for i in range(Npts):
      #for j in range(Npts):   #initial value from last step not specified
       rac_ai = Rac_a[i];     rac_ii = Rac_i[i];
       rho_ai = Rho_a[i];     rho_ii = Rho_i[i];
       rac_ain = Rac_an[i];   rac_iin = Rac_in[i];
       rho_ain = Rho_an[i];   rho_iin = Rho_in[i];      
       Ki_plus = K_plus[i,:];   Ki_minus = K_minus[i];
       kappa_ip = kappa_p[i,:]; kappa_im = kappa_m[i];
       Rc_eqsecterm += M_plus*Rc/Npts - M_minus*rac_ii   # - Ki_minus * (rac_ai - rac_ii)
       Rhoc_eqsecterm += mu_plus*Rhoc/Npts - mu_minus*rho_ii   # - kappa_im * (rho_ai - rho_ii)
       # Residue from Rho-Rac antagolistic effect and membrane-cytosol interactions
       Res_sub = sub_residue(rho_ai, rac_ai, rho_ii, rac_ii, rho_ain, rac_ain, rho_iin, \
                             rac_iin, Ki_plus, Ki_minus, kappa_ip, kappa_im, M_plus, M_minus, \
                             mu_plus, mu_minus, Rc, Rhoc, Npts, dt )     #Rc, Rhoc unknown
       Ksub = sub_derivative_matrix(rho_ai, rac_ai, rho_ii, rac_ii, Ki_plus, Ki_minus,  \
                             kappa_ip, kappa_im, M_plus, M_minus, mu_plus, mu_minus, dt )
       Ksub_Jii = derivative_J(edge_len, D, i, i, Npts,dt)
       K_matrix_total[4*i:4*i+4, 4*i:4*i+4] = Ksub + Ksub_Jii
       K_matrix_total[4*Npts, 4*i] = 0  #-dt*Ki_minus  #k_RcRi 
       K_matrix_total[4*Npts, 4*i+1] = -dt*M_minus # + dt*Ki_minus #k_RcRi
       K_matrix_total[4*Npts+1, 4*i+2] = 0. #-dt*kappa_im  #k_rhoci 
       K_matrix_total[4*Npts+1, 4*i+3] = -dt*mu_minus #+ dt*kappa_im  #k_rhoci 
       K_matrix_total[4*i+1, 4*Npts] = -dt*M_plus/Npts #k_RiRc 
       K_matrix_total[4*i+3, 4*Npts+1] = -dt*mu_plus/Npts #k_rhoic
       # assemble derivative matrix term from diffusion flux
       jp = i + 1;  Ksub_Jip = derivative_J(edge_len, D, i, jp, Npts, dt);
       jm = i - 1;  Ksub_Jim = derivative_J(edge_len, D, i, jm, Npts, dt);
       if i == 0 and Npts>=3:
          K_matrix_total[4*i:4*i+4, 4*jp:4*jp+4] = Ksub_Jip
          K_matrix_total[4*i:4*i+4, 4*(Npts-1):4*Npts] = Ksub_Jim
       elif i == Npts - 1 and Npts>=3:
          K_matrix_total[4*i:4*i+4, 0:4] = Ksub_Jip
          K_matrix_total[4*i:4*i+4, 4*jm:4*jm+4] = Ksub_Jim
       elif Npts>=3: 
          K_matrix_total[4*i:4*i+4, 4*jp:4*jp+4] = Ksub_Jip 
          K_matrix_total[4*i:4*i+4, 4*jm:4*jm+4] = Ksub_Jim
       # assemble derivative matrix term of the cytosol
       K_matrix_total[4*Npts, 4*Npts] = K_matrix_total[4*Npts, 4*Npts] + dt * M_plus/Npts 
       K_matrix_total[4*Npts+1, 4*Npts+1] = K_matrix_total[4*Npts+1, 4*Npts+1] + dt * mu_plus/Npts 
       # residue from cytosol - membrane interaction
       Residue[4*i:4*i+4] = Res_sub 
    # cytosol residue
    Residue[4*Npts] = Rc - Rcn + dt * Rc_eqsecterm 
    Residue[4*Npts + 1] = Rhoc - Rhocn + dt * Rhoc_eqsecterm
    Residue += Jflux_term
    return (K_matrix_total, Residue)


'''
Initialize Rac1 and RhoA signaling
'''
def Initialize_RacRho(Ini_Raca, Ini_Raci,Ini_Rhoa, Ini_Rhoi, polar_num, nnum, uniform, cycid):
    if uniform == int(0):
        Rac_a = Ini_Raca/nnum*np.random.random([nnum])   #Ini_Raca*np.random.random([nnum])
        Rac_a = np.maximum(0.7*Ini_Raca/nnum, Rac_a);   
        #Rac_a[0:polar_num] = 2.0*Ini_Raca/nnum*np.ones([polar_num])   # 2*Rac_a[0:polar_num]
        Rho_a = Ini_Rhoa/nnum*np.ones([nnum])   #Ini_Rhoa*np.random.random([nnum])
        Rac_i = Ini_Raci/nnum*np.ones([nnum])   #Ini_Raci*np.random.random([nnum])
        Rho_i = Ini_Rhoi/nnum*np.ones([nnum])    #Ini_Rhoi*np.random.random([nnum])
    else:
        Rac_a = Ini_Raca/nnum*np.random.random([nnum])   
        randnum=np.random.randint(-2,2,2);
        if randnum[0]<0: randnum[0] = randnum[0] + nnum;
        if randnum[1]<0: randnum[1] = randnum[1] + nnum;
        Rac_a[randnum[0]] =  np.amax([2.*Ini_Raca/nnum,  3.*Ini_Raca/nnum*np.random.rand()])
        Rac_a[randnum[1]] =  np.amax([2*Ini_Raca/nnum,  3.*Ini_Raca/nnum*np.random.rand()])
        Rho_a = Ini_Rhoa/nnum*np.random.random([nnum])   
        Rac_i = Ini_Raci/nnum*np.random.random([nnum])    
        Rho_i = Ini_Rhoi/nnum*np.random.random([nnum])   
    return (Rac_a, Rac_i, Rho_a, Rho_i)


'''
Macaulay bracket definition
'''
def macualay(inpt):
    if inpt>=0:
        oupt = inpt;
    else:
        oupt = 0;
    return oupt


'''
Numerically compute diffusive flux based on four points coordinates
and middle two points RhoA or Rac1 concentrations
'''
def membraneflux(D, x0, x1, x2, x3, R1, R2):  # assume i = 1
    l0 = sqrt((x0[0]-x1[0])**2 + (x0[1]-x1[1])**2)
    l1 = sqrt((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2)
    l2 = sqrt((x2[0]-x3[0])**2 + (x2[1]-x3[1])**2)
    LL1 = (l0 + l1)/2
    LL2 = (l1 + l2)/2
    J1 = -D*(R2/LL2 - R1/LL1)/l1
    return J1

'''
Normalize vectors
'''
def Normalize_vector(vx, vy):
    nm_value = np.sqrt(vx**2 + vy**2);  #this is the length from nucleus to membrane
    nn = np.size(nm_value);  
    vx_nm = vx/nm_value;  vy_nm = vy/nm_value; 
    nm_vx = np.zeros(nn); nm_vy = np.zeros(nn); 
    nm_vx[:nn-1] = vx_nm[1:nn]; nm_vx[nn-1] = vx_nm[0]; 
    nm_vy[:nn-1] = vy_nm[1:nn]; nm_vy[nn-1] = vy_nm[0]; 
    cos_val = vx_nm*nm_vx  + vy_nm*nm_vy; 
    sin_val = np.sqrt(1 -  cos_val**2.); 
    lg_value = np.zeros(nn); 
    lg_value[:nn-1] = nm_value[1:nn]; 
    lg_value[nn-1] = nm_value[0]; 
    area_sub = nm_value*lg_value*sin_val/2. 
    Area = np.sum(area_sub) 
    return (vx_nm, vy_nm, nm_value, Area) 



'''
Function of membrane Rac1 activation rate coefficient
'''
def Rac1_activation(kp, gama_r, beta_r, rho, rac, rac_0, rho_0, cr, norder):
    K_plus = (cr*kp + gama_r*rho_0**norder/(rho_0**norder + rho**norder) + beta_r*rac**norder/(rac_0**norder + rac**norder) ) 
    K_plus_drv1 = norder*beta_r*rac_0**norder*rac**(norder-1)/(rac_0**norder + rac**norder)**2.0  #partial(K_plus)/partial(rac)
    K_plus_drv2 = (-norder*gama_r*rho_0**norder*rho**(norder-1))/(rho_0**norder + rho**norder)**2.0
    return (K_plus, K_plus_drv1, K_plus_drv2)

'''
Function of membrane Rac1 inactivation rate coefficient
'''
def Rac1_inactivation(kb_minus, kpolar_minus, angle, Angle0):
    ang_diff = Angle0 - angle
    coef_diff = 10**(macualay(ang_diff/Angle0)) - 1.
    kb = kb_minus + kpolar_minus*coef_diff
    return kb  

'''
Function of membrane RhoA activation rate coefficient
'''
def RhoA_activation(kappa_p, gama_rho, beta_rho, rac, rho, rac_0, rho_0, dr, norder):
    kappa_plus = (dr*kappa_p + gama_rho*rac_0**norder/(rac_0**norder + rac**norder) + beta_rho*rho**norder/(rho_0**norder + rho**norder))
    kappa_plus_drv1 = (-norder*gama_rho*rac**(norder-1)*rac_0**norder)/(rac_0**norder + rac**norder)**2.0
    kappa_plus_drv2 = (norder*beta_rho*rho_0**norder*rho**(norder-1))/(rho_0**norder + rho**norder)**2.0 
    return (kappa_plus, kappa_plus_drv1, kappa_plus_drv2)

'''
Function of membrane RhoA inactivation rate coefficient
kappa_plus and kappa_minus is a 2d vector including values and derivatives
'''
def RhoA_inactivation(kapbb_minus, kap_polar_minus, angle, Angle0):
    ang_diff = angle - Angle0
    coef_diff = 10**(macualay(ang_diff/Angle0)) - 1.
    kappa_b = kapbb_minus + kap_polar_minus*coef_diff
    return kappa_b

'''
Calculate substrate stiffness based on stiffness gradient --- Durotactic study with option duro = 1
'''
def sub_stiff_visco(xcj, ycj, Rcell, K_sub, Grad, gama_sub, Ksub_ub, Gsub_ub, Ksub_lb, Gsub_lb, duro):
    if duro == int(1):
        ksub = K_sub + (xcj)*Grad   
        gama = gama_sub 
    else:
        ksub = K_sub;   gama = gama_sub;
    if ksub>Ksub_ub: ksub=Ksub_ub
    if gama>Gsub_ub: gama=Gsub_ub
    if ksub<Ksub_lb: ksub=Ksub_lb
    if gama<Gsub_lb: gama=Gsub_lb
    return (ksub, gama)

'''
Function of calculating residue 
'''
def sub_residue(rho_a, rac_a, rho_i, rac_i, rho_an, rac_an, rho_in, rac_in, \
                K_plus,  K_minus, kappa_plus, kappa_minus, M_plus, M_minus, \
                mu_plus, mu_minus, Rc, Rhoc, Npts, dt ):
    Res_racrho = np.zeros([4])
    Res_racrho[0] = rac_a - rac_an - dt * (K_plus[0]*rac_i - K_minus*rac_a)
    Res_racrho[1] = rac_i - rac_in + dt * (K_plus[0]*rac_i - K_minus*rac_a \
                    - M_plus*Rc/Npts + M_minus*rac_i)
    Res_racrho[2] = rho_a - rho_an - dt * (kappa_plus[0]*rho_i - kappa_minus*rho_a)
    Res_racrho[3] = rho_i - rho_in + dt * (kappa_plus[0]*rho_i - kappa_minus*rho_a \
                    - mu_plus*Rhoc/Npts + mu_minus*rho_i)
    return Res_racrho

                    
'''
Function of calculating derivative matrix
'''
def sub_derivative_matrix(rho_a, rac_a, rho_i, rac_i, K_plus, K_minus, kappa_plus, \
                        kappa_minus, M_plus, M_minus, mu_plus, mu_minus, dt ):
    kmatrix_racrho = np.zeros([4,4])
    kmatrix_racrho[0,0] = 1 + dt * K_minus - dt * K_plus[1] * rac_i 
    kmatrix_racrho[0,1] = - dt * K_plus[0]  
    kmatrix_racrho[0,2] = - dt * K_plus[2] * rac_i  
    kmatrix_racrho[1,0] =  + dt * K_plus[1] * rac_i - dt * K_minus 
    kmatrix_racrho[1,1] = 1 + dt * K_plus[0] + M_minus * dt 
    kmatrix_racrho[1,2] = dt * K_plus[2] * rac_i
    kmatrix_racrho[2,0] = - dt * kappa_plus[1] * rho_i  
    kmatrix_racrho[2,2] = 1 + dt * kappa_minus - dt * kappa_plus[2] * rho_i  
    kmatrix_racrho[2,3] = - dt * kappa_plus[0]
    kmatrix_racrho[3,0] = dt * kappa_plus[1] * rho_i  
    kmatrix_racrho[3,2] =  + dt * kappa_plus[2] * rho_i - dt * kappa_minus
    kmatrix_racrho[3,3] = 1 + dt * kappa_plus[0] + dt * mu_minus 
    return kmatrix_racrho
        



