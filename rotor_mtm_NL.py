
import rotor_mtm as rmtm
import ross as rs
import numpy as np
import time
import pickle
import plotly.graph_objects as go
from scripts.harmbal import poincare_section as pcs
from multiprocessing import Pool
import os

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
# r_det_flextun = rmtm.RotorMTM(rotor,n_pos,dk_r,k0*100,k1,var=0,var_k=0,p_damp=1e-4,ge=True),
# r_var3_flextun = rmtm.RotorMTM(rotor,n_pos,dk_r,k0*100,k1,var=var,var_k=0,p_damp=1e-4,ge=True,exp_var=3),
# Rotores com ressonadores translacionais
r_det_transtun = rmtm.RotorMTM(rotor,n_pos,dk_r,k0,100*k1,var=0,var_k=0,p_damp=1e-4,ge=True),
r_var1_transtun = rmtm.RotorMTM(rotor,n_pos,dk_r,k0,100*k1,var=var,var_k=0,p_damp=1e-4,ge=True,exp_var=1),
)

f = {0: 100,
     1: -100.j}

def run_rotor_hb(input):

    rotor_dict = input[0]
    k = input[1]
    f0 = input[2]

    sp_arr = np.linspace(1, 800, 200)

    probe_dof = [rotor_dict[k].N2 - 4]

    f = {0: f0,
         1: -f0 * 1.j}

    fm_flag = np.zeros(sp_arr.size)
    rms_hb = np.zeros((rotor_dict[k].N + 1, len(sp_arr)))

    with open(f'results/out_data_{k}.pic', 'rb') as file:
        out = pickle.load(file)
        r = [out['r_map'][j, j] for j in range(len(sp_arr))]
        r_solo = [out['rsolo_map'][j, j] for j in range(len(sp_arr))]

    for i, sp in enumerate(sp_arr):

        sys = rotor_dict[k].create_Sys_NL(x_eq0=1e-3, x_eq1=None, sp=sp, n_harm=10, nu=1, N=1,
                                                cp=1e-4)

        try:
            x_hb, res = sys.solve_hb(f, omg=sp, full_output=True)
        except:
            x_hb, res = sys.solve_hb(f, omg=sp, full_output=True, method='ls')
        print(f'Calculation finished for rotor {k}, f0 = {f0} and omg = {sp}')
        try:
            z = res.x
        except:
            z = res[0]

        try:
            fm = sys.floquet_multipliers(sp, z, dt_refine=10)
            if np.max(np.abs(fm)) > 1:
                fm_flag[i] = 1
        except:
            print(f"ERRO! Floquet Multipliers were not calculated for sp = {sp}")
            fm_flag[i] = 0

        try:
            if res[-2] != 1:
                print(res[-1])
                rms_hb[-1, i] = 1
        except:
            if not res.success:
                print(res.message)
                rms_hb[-1, i] = 1

        rms_hb[:-1, i] = np.array(
            [np.sqrt(np.sum((x_hb[j, :] - np.mean(x_hb[j, :])) ** 2) / (len(sys.t(sp)) - 1)) for j in
             range(x_hb.shape[0])])

    with open(f'tools/Rotor_NL/out_hb_{k}_{f0}.pic', 'wb') as file:
        pickle.dump(dict(force=f,
                         sp_arr=sp_arr,
                         rms_hb=rms_hb,
                         fm_flag=fm_flag),
                    file)

    sl = [False] * (np.max(probe_dof) + 1)
    sl[probe_dof[0]] = True
    fig = go.Figure(data=[go.Scatter(x=sp_arr, y=rms_hb[i, :], name=f'DoF {i}- HB') for i in probe_dof] + \
                         [go.Scatter(x=sp_arr, y=r * f0, name=f'Linear Resonators'),
                          go.Scatter(x=sp_arr, y=r_solo * f0, name=f'Bare Rotor')] + \
                         [go.Scatter(x=[sp_arr[i] for i in range(len(sp_arr)) if rms_hb[-1, i] == 1],
                                     y=[rms_hb[j, i] for i in range(len(sp_arr)) if rms_hb[-1, i] == 1],
                                     name='Flagged', mode='markers', marker=dict(color='black'),
                                     showlegend=sl[j],
                                     legendgroup='flag') for j in probe_dof] + \
                         [go.Scatter(x=[sp_arr[i] for i in range(len(sp_arr)) if fm_flag[i] == 1],
                                     y=[rms_hb[j, i] for i in range(len(sp_arr)) if fm_flag[i] == 1],
                                     name='Unstable', mode='markers', marker=dict(color='red', symbol='x'),
                                     showlegend=sl[j],
                                     legendgroup='fm_flag') for j in probe_dof]
                    )
    fig.update_layout(title={'xanchor': 'center',
                             'x': 0.4,
                             'font': {'family': 'Arial, bold',
                                      'size': 15},
                             },
                      yaxis={"gridcolor": "rgb(159, 197, 232)",
                             "zerolinecolor": "rgb(74, 134, 232)",
                             },
                      xaxis={'range': [0, np.max(sp_arr)],
                             "gridcolor": "rgb(159, 197, 232)",
                             "zerolinecolor": "rgb(74, 134, 232)"},
                      xaxis_title='Frequency (rad/s)',
                      yaxis_title='Amplitude',
                      font=dict(family="Calibri, bold",
                                size=18))
    fig.update_yaxes(type="log")

    fig.write_image(f'tools/Rotor_NL/{k}_force_{f0}.png', scale=8)

input_list = []
for k in rotor_dict:
    for f0 in [1, 10, 100, 1000]:
        input_list.append((rotor_dict, k, f0))

if __name__ == '__main__':
    with Pool(7) as p:
        p.map(run_rotor_hb, input_list)