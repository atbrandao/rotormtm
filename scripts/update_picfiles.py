from os import listdir, getcwd
from os.path import isfile, join
from pickle import load, dump
from rotor_mtm.results import IntegrationResults
import plotly.graph_objs as go
import numpy as np
import ross as rs

picfiles = ['scripts\\' + f for f in listdir(getcwd() + '\\scripts') if (isfile(join(getcwd() + '\\scripts\\', f)) and '.pic' in f)]

for f in picfiles:
    with open(f, 'rb') as file:
        res = load(file)
    # res = IntegrationResults.update_class_object(res)
    update = False
    
    if 'system' in res.__dict__:

        r = res.system.rotor

        for el in r._rotor.elements:
            if type(el) == rs.PointMass and 'size' not in el.__dict__:
                update = True
                el.size = 2

        last_dof = [r.rotor_solo_disks.nodes[-1] * 4,
                    r.rotor_solo_disks.nodes[-1] * 4 + 1]
        last_dof_name = ['last_x', 'last_y']

        base_dof = []
        res_dof = []
        base_dof_name = []
        res_dof_name = []
        N2 = r.N2

        for i, a in enumerate(r.n_pos):
            
            base_dof += [4 * a, 4 * a + 1]
            res_dof += [N2 + 4 * i, N2 +  4 * i + 1]
            base_dof_name += [f'base{i}_x', f'base{i}_y']
            res_dof_name += [f'res{i}_x', f'res{i}_y']

        if 'linear_results' not in res.__dict__ or res.linear_results is None:
            res.linear_results = r.calc_frf(sp_arr=res.fl,
                                f=np.ones(len(res.fl)),
                                probe_dof=last_dof + base_dof + res_dof,
                                probe_names=last_dof_name + base_dof_name + res_dof_name,
                                f_node=0,
                                silent=False,
                                rotor_solo=False)
            update = True

        if 'rigid_results' not in res.__dict__ or res.rigid_results is None:
            res.rigid_results = r.calc_frf(sp_arr=res.fl,
                                f=np.ones(len(res.fl)),
                                probe_dof=last_dof,
                                probe_names=last_dof_name,
                                f_node=0,
                                silent=False,
                                rotor_solo=True)
            update = True
            
    if update:    
        with open(f, 'wb') as file:
            dump(res, file)

    