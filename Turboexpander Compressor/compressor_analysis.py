import ross as rs
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from xlwings import Book
import rotor_mtm.rotor_mtm as rmtm
from rossxl import MaxBrg
from copy import deepcopy
from ross import Q_
from pickle import dump, load

sp_arr = Q_(np.arange(1000, 40000, 500), "RPM")

rotor_original = rs.Rotor.load('compressor_rotor.toml')

with open('rotor_dict_v1.pic', 'rb') as file:
    rotor_dict = load(file)

r = rotor_dict['r_det_transtun']
r_var = rotor_dict['r_var1_transtun']

f_0 = Q_(1944, 'rad/s').to('rad/s').m

r_nl = r.create_Sys_NL(x_eq0=(20e-6, None),
                       sp=f_0,
                       cp=1e-4,
                       n_harm=5)

sp_arr = Q_(np.arange(1000, 40000, 500), "RPM")
sp_arr = Q_(np.arange(15000, 25000, 3000), "RPM")

base_dof_x = [4 * a for a in r_nl.rotor.n_pos]
base_dof_y = [4 * a + 1 for a in r_nl.rotor.n_pos]
res_dof_x = [a for a in range(r_nl.rotor.N2, r_nl.rotor.N, 4)]
res_dof_y = [a + 1 for a in range(r_nl.rotor.N2, r_nl.rotor.N, 4)]
brg_dof = [4 * 21, 4 * 21 + 1, 4 * 33, 4 * 33 + 1]

base_names_x = [f'base{n}_x' for n in range(len(r_nl.rotor.n_pos))]
base_names_y = [f'base{n}_y' for n in range(len(r_nl.rotor.n_pos))]
res_names_x = [f'res{n}_x' for n in range(len(r_nl.rotor.n_pos))]
res_names_y = [f'res{n}_y' for n in range(len(r_nl.rotor.n_pos))]
brg_names = ['1st_imp_x', '1st_imp_y', '2nd_imp_x', '2nd_imp_y']

probe_dof = base_dof_x + base_dof_y + res_dof_x + res_dof_y + brg_dof
probe_names = base_names_x + base_names_y + res_names_x + res_names_y + brg_names

f = {
        27: 4 * 1e-6 * 6350 * rotor_dict['r_rigid'].rotor_solo_disks.m / (f_0 * 60 / (2 * np.pi)),
    }

tf = 1
downsampling = 100
unbalance = True

res = r_nl.plot_smart_frf(
                    sp_arr.to('rad/s').m,
                    f,
                    tf=tf,
                    stability_analysis=False,
                    probe_dof=base_dof_x + base_dof_y + res_dof_x + res_dof_y + brg_dof,
                    downsampling=downsampling,
                    # save_rms=f'Rotor_NL/rotor_nl_frf_f-{f[0]}.dat',
                    run_hb=True, # False,
                    # save_raw_data='Rotor_NL/',
                    return_results=True,
                    probe_names=base_names_x + base_names_y + res_names_x + res_names_y + brg_names,
                    gyroscopic=True,
                    unbalance=unbalance
                )

res.linear_results = r_nl.rotor.calc_frf(
    sp_arr=sp_arr.to('rad/s').m,
    f=np.ones(len(sp_arr)),
    probe_dof=base_dof_x + base_dof_y + res_dof_x + res_dof_y + brg_dof,
    probe_names=base_names_x + base_names_y + res_names_x + res_names_y + brg_names,
    f_node=27,
    silent=False,
    rotor_solo=False
    )

res.rigid_results = r_nl.rotor.calc_frf(
    sp_arr=sp_arr.to('rad/s').m,
    f=np.ones(len(sp_arr)),
    probe_dof=base_dof_x + base_dof_y + brg_dof,
    probe_names=base_names_x + base_names_y + brg_names,
    f_node=27,
    silent=False,
    rotor_solo=True
    )

with open(f'teste_comp_nl_res.pic', 'wb') as file:
    dump(res, file)