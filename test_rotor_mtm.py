import rotor_mtm as rmtm
import ross as rs
import numpy as np

Nel = 30
shaft = [
    rs.ShaftElement(
        n=i,
        L=0.05,
        odl=0.05,
        idl=0,
        odr=0.05,
        idr=0,
        material=rs.materials.steel,
        rotary_inertia=True,
        gyroscopic=True,
    )
    for i in range(Nel)
]

disks = [rs.DiskElement(Id=1.7809,Ip=0.32956,m=32.59,n=10),
         rs.DiskElement(Id=1.7809,Ip=0.32956,m=32.59,n=15),
         rs.DiskElement(Id=1.7809,Ip=0.32956,m=32.59,n=20)]

c = 5e2
kxy = 0*2e5

brgLNA = rs.BearingElement(n=0,
                        kxx=[900000.0], kxy=[kxy],
                        kyx=[-kxy], kyy=[900000.0],
                        cxx=[c], cxy=[0],
                        cyx=[0], cyy=[c],
                        frequency=None,)

brgLA = rs.BearingElement(n=Nel,
                        kxx=[900000.0], kxy=[kxy],
                        kyx=[-kxy], kyy=[900000.0],
                        cxx=[c], cxy=[0],
                        cyx=[0], cyy=[c],
                        frequency=None,)

rotor = rs.Rotor(shaft,disks,[brgLA,brgLNA])

n_res = 15
# m_ratio = 0.1
i_d = 0.06
o_d = 0.14
L = 0.04
n_center = 15
var = 0.6
f_0 = 3770 #10000 # 1800/60*2*np.pi # em rad/s
f_1 = 634.793 # 1800/60*2*np.pi # em rad/s
p_damp = 1e-4
ge = True

rho = 7800
mr = np.pi/4*(o_d**2-i_d**2)*L * rho
Ip = 1/8*mr*(o_d**2+i_d**2)
It = 1/12*mr*(3/4*(o_d**2+i_d**2)+L**2)
dk_r = rs.DiskElement(n=0,m=mr,Id=It,Ip=Ip)

k0 = mr * f_0**2
k1 = It * f_1**2

n_pos = np.arange(n_center-int(n_res/2),n_center+n_res-int(n_res/2),1)

sp_arr = np.linspace(1,800,200)

r_det_flextun = rmtm.RotorMTM(rotor,n_pos,dk_r,k0,k1,var=0,var_k=0,p_damp=1e-4,ge=True)
r_var_flextun = rmtm.RotorMTM(rotor,n_pos,dk_r,k0,k1,var=var,var_k=0,p_damp=1e-4,ge=True,exp_var=3)
r_var3_flextun = rmtm.RotorMTM(rotor,n_pos,dk_r,k0,k1,var=var,var_k=0,p_damp=1e-4,ge=True,exp_var=3)
N = r_var_flextun.N
diff_lim = -r_var_flextun.A(0)[N+n_pos[0]*4+2,n_pos[0]*4+2]/k1
out = r_var_flextun.run_analysis(sp_arr,diff_lim=diff_lim)
w_var = out['w']
w_var_res = out['w_res']
r_var_map = out['r_b_map']
rmtm.plot_camp_heatmap(r_var_map,w_var,sp_arr,w_res=w_var_res).write_image('Heatmap_var_fow_flextunTESTE.png',scale=8)

r_det_sbg = rmtm.RotorMTM(rotor,n_pos,dk_r,k0/100,k1,var=0,var_k=0,p_damp=1e-4,ge=True)
r_var3_sbg = rmtm.RotorMTM(rotor,n_pos,dk_r,k0/100,k1,var=var,var_k=0,p_damp=1e-4,ge=True,exp_var=3)

r_det_transtun = rmtm.RotorMTM(rotor,n_pos,dk_r,k0/100,100*k1,var=0,var_k=0,p_damp=1e-4,ge=True)
r_var1_transtun = rmtm.RotorMTM(rotor,n_pos,dk_r,k0/100,100*k1,var=var,var_k=0,p_damp=1e-4,ge=True,exp_var=1)
r_var3_transtun = rmtm.RotorMTM(rotor,n_pos,dk_r,k0/100,100*k1,var=var,var_k=0,p_damp=1e-4,ge=True,exp_var=3)

out_var3_sbg = r_var3_sbg.run_analysis(sp_arr,diff_lim=diff_lim)
out_var3_transtun = r_var3_transtun.run_analysis(sp_arr,diff_lim=diff_lim)
rmtm.plot_frf([out_var3_sbg['rsolo_b'],out_var3_sbg['r_b'],out_var3_transtun['r_b']],sp_arr)

out_diff = r_var3_flextun.run_analysis(sp_arr,diff_lim=diff_lim,diff_analysis=True,dof=2,heatmap=True)
rmtm.plot_diff_modal(out_diff['w'],out_diff['diff'],sp_arr,mode='abs')
