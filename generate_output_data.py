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
sp_arr = np.linspace(1,800,200)

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

rotor_dict['r_var1_transtun'].plot_rotor().write_image('rotor.svg')

r = rotor_dict['r_var1_transtun']
Sys0 = r.create_Sys_NL(x_eq0=1e-3, sp=sp_arr[0], cp=1e-4, n_harm=10)
# print(Sys.A.shape)
# print(np.vstack([np.zeros((r.rotor_solo.ndof,1)),
#                Sys.x_eq*np.ones((r.n_res*4,1)),
#                np.zeros((r._rotor.ndof,1))]).shape)

probe_node = r.N2//4 - 1

f = {0: 100,
     1: -100.j}
z0 = Sys0.z0(omg=sp_arr[0],f_omg={0:0})

tf = 10
dt = 0.45 * 2*np.pi/np.max(np.imag(np.linalg.eigvals(r.rotor_solo.A(0))))
t_rk = np.arange(0, tf, dt)

rms_hb = np.zeros((3, len(sp_arr)))
rms_rk = np.zeros((1, len(sp_arr)))
cost_hb = []
pc = []
n_points = 50

for i, sp in enumerate(sp_arr):
    # sp=100
    t0 = time.time()
    Sys = r.create_Sys_NL(x_eq0=1e-3, sp=sp, cp=1e-4, n_harm=10)
    # Sys = r.create_Sys_NL(x_eq1=1e-1, sp=sp, cp=1e-4, n_harm=10, nu=1, N=1)
    z0 = Sys.z0(omg=sp, f_omg={0: 0})
    try:
        x_hb, res = Sys.solve_hb(f, sp, z0=z0, full_output=True, method=None) #, state_space=True)  # 'ls')
    except:
        rms_hb[1, i] = 2
        print('Failed to converge.')

    print(f'Harmbal took {(time.time() - t0):.1f} seconds to run.')

    t1 = time.time()
    # x_rk = Sys.solve_transient(f=f, t=t_rk, omg=sp, x0=x_hb[:,0].reshape((2*r.N,1)),
    #                            probe_dof=[-4,-40,-probe_node*4])
    print(f'RK4 took {(time.time() - t1):.1f} seconds to run.')
    # go.Figure(data=[go.Scatter(x=t_rk, y=x_rk[0, :])]).write_html(f'WF_test_{sp}_-4.html')
    # go.Figure(data=[go.Scatter(x=t_rk, y=x_rk[1, :])]).write_html(f'WF_test_{sp}_-40.html')
    # go.Figure(data=[go.Scatter(x=t_rk, y=x_rk[2, :])]).write_html(f'WF_test_{sp}_-shaft.html')
    # with open(f'x_rk {sp}.pic'.replace(':', '_'), 'wb') as file:
    #     pickle.dump([x_rk,t_rk], file)
    t1 = time.time()

    rms_hb[0, i] = np.sqrt(np.sum((x_hb[probe_node*4, :] - np.mean(x_hb[probe_node*4, :])) ** 2) / (len(Sys.t(sp)) - 1))
    rms_hb[2, i] = np.sqrt(
        np.sum((x_hb[r.N2, :] - np.mean(x_hb[r.N2, :])) ** 2) / (len(Sys.t(sp)) - 1))
    # rms_rk[0, i] = np.sqrt(
    #     np.sum((x_rk[0, int((tf / 2) / dt):] - np.mean(x_rk[0, int((tf / 2) / dt):])) ** 2) / (int((tf / 2) / dt)))
    # pc.append(pcs(x_rk, t_rk, sp, n_points))

    try:
        cost_hb.append(res.cost)
        z0 = res.x
        if not res.success:
            print(res.message)
            rms_hb[1, i] = 1
    except:
        cost_hb.append(np.linalg.norm(res[1]['fvec']))
        z0 = res[0]
        if res[-2] != 1:
            print(res[-1])
            rms_hb[1, i] = 1

    # with open(f'data_rk f {f} flextun.pic'.replace(':', '_'), 'wb') as file:
    #     pickle.dump([rms_rk, pc], file)

    print(f'Frequency: {sp:.1f} rad/s -> completed on {time.ctime()}.')

    with open(f'data_hb f {f} transtun.pic'.replace(':', '_'), 'wb') as file:
        pickle.dump([rms_hb, cost_hb], file)








#
# def save_full_out(k):
#     print(f'Running analysis for rotor {k}:')
#     if 'flex' in k:
#         dof = 2 # Select the flexural DoF to calculate diff
#     else:
#         dof = 0 # Select the translation DoF to calculate diff
#     out = rotor_dict[k].run_analysis(sp_arr, diff_lim=diff_lim, diff_analysis=True,
#                                      heatmap=True, dof=dof, dof_show=dof_show)
#
#     with open(f'out_data_{k}_{dof_show}.pic', 'wb') as handle:
#         pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# dof_show = 0
# if __name__ == '__main__':
#     with Pool(7) as p:
#         p.map(save_full_out,list(rotor_dict.keys()))
#
# dof_show = 2
# if __name__ == '__main__':
#     with Pool(2) as p:
#         p.map(save_full_out,list(rotor_dict.keys())[:3:2])


