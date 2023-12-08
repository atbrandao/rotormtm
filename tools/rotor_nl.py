import harmbal as hb
import rotor_mtm as rmtm
import ross as rs
import numpy as np
import time
import pickle
import plotly.graph_objects as go
from tools.harmbal import poincare_section as pcs
from multiprocessing import Pool

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
f_0 = 377 #10000 # 1800/60*2*np.pi # em rad/s
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

# sp_arr = np.linspace(300,400,25)


diff_lim = 1e9

rotor_dict = dict(
# Rotores com ressonadores flexurais
r_det_flextun = rmtm.RotorMTM(rotor,n_pos,dk_r,k0*100,k1,var=0,var_k=0,p_damp=1e-4,ge=True),
r_var1_flextun = rmtm.RotorMTM(rotor,n_pos,dk_r,k0*100,k1,var=var,var_k=0,p_damp=1e-4,ge=True,exp_var=1),
r_var3_flextun = rmtm.RotorMTM(rotor,n_pos,dk_r,k0*100,k1,var=var,var_k=0,p_damp=1e-4,ge=True,exp_var=3),
# Rotores com ressonadores translacionais
r_det_transtun = rmtm.RotorMTM(rotor,n_pos,dk_r,k0,100*k1,var=0,var_k=0,p_damp=1e-4,ge=True),
r_var1_transtun = rmtm.RotorMTM(rotor,n_pos,dk_r,k0,100*k1,var=var,var_k=0,p_damp=1e-4,ge=True,exp_var=1),
r_var3_transtun = rmtm.RotorMTM(rotor,n_pos,dk_r,k0,100*k1,var=var,var_k=0,p_damp=1e-4,ge=True,exp_var=3),
# Rotores com ambos os GDL sintonizados (super bandgap)
r_det_sbg = rmtm.RotorMTM(rotor,n_pos,dk_r,k0,k1,var=0,var_k=0,p_damp=1e-4,ge=True),
r_var3_sbg = rmtm.RotorMTM(rotor,n_pos,dk_r,k0,k1,var=var,var_k=0,p_damp=1e-4,ge=True,exp_var=3)
)

r = rotor_dict['r_det_transtun']

probe_node = r.N2//4 - 1
probe_dof_x = probe_node * 4
probe_dof_y = probe_node * 4 + 1

sp_arr = np.linspace(1, 800, 200)
f = {0: 100,
     1: -100.j}

omg = 400

tf = 3

Sys0 = r.create_Sys_NL(x_eq0=1e-3, sp=sp_arr[0], cp=1e-4, n_harm=5)

dt_base = Sys0.dt_max()
n_periods = max([1, np.round(tf / (2 * np.pi / omg))])
tf = n_periods * 2 * np.pi / omg
dt = 2 * np.pi / omg / (np.round(2 * np.pi / omg / dt_base))
t_rk = np.arange(0, tf + dt / 2, dt)

# x_rk, x0 = Sys0.solve_transient(f, t_rk, omg, np.zeros((Sys0.ndof*2, 1)), last_x=True, dt=dt)
#
# fig = go.Figure(data=[go.Scatter(x=t_rk, y=x_rk[probe_dof_x, :])])
# fig.write_html(f'Rotor_NL/{tf} s.html')

for f0 in [0.1, 1, 2, 3, 4]:
    f = {0: f0 * 100,
         1: -f0 * 100.j}

    fig = Sys0.plot_smart_frf(sp_arr, f,
                           tf=1,
                           stability_analysis=False,
                           probe_dof=[probe_dof_x, probe_dof_y],
                           downsampling=100,
                           save_rms=f'Rotor_NL/rotor_nl_frf_f-{f[0]}.dat')

    fig.write_html(f'Rotor_NL/rotor_nl_frf_f-{f[0]}.html')