from pickle import load, dump
from rotor_mtm.results import IntegrationResults
import plotly.graph_objs as go
import numpy as np
from os import listdir, getcwd
from os.path import isfile, join

def get_results(f,
                tun='transtun',
                whirl='backward',
                mod=''):
    
    try:

        with open(f'{mod}results_{tun}_{whirl}_f{f}.pic', 'rb') as file:
            res = load(file)
        res = IntegrationResults.update_class_object(res)
        
    except:
        f = float(f)
        with open(f'{mod}results_{tun}_{whirl}_f{f}.pic', 'rb') as file:
            res = load(file)
        res = IntegrationResults.update_class_object(res)
    
           
    return res

def plot_frf_sensitivity(force_type,
                         mod,
                         f0,
                         var=False,
                         tun='transtun',
                         amplitude_units='rms',
                         unbalance_speed=377,
                         dof=['last_x'],
                         param='w0',
                         arr=[300, 310, 320, 330],                         
                         plot_linear=True,
                         plot_rigid=True
                         ):
    
    if type(dof) != list:
        dof = [dof]
    
    if var:
        var_str = 'var'
    else:
        var_str = ''

    if param == 'w0':          
        filename_base = f'{mod}results_{tun}PARAM{var_str}_{force_type}_f{f0}.pic'
        
    picfiles = [f for f in listdir(getcwd()) if (isfile(join(getcwd(), f)) and any(f.startswith(filename_base.replace('PARAM', str(a))) for a in arr))]
    print(picfiles)
    
    fig = go.Figure()

    for i, f in enumerate([filename_base.replace('PARAM', str(a)) for a in arr]):

        if isfile(join(getcwd(), f)):

            print(f)

            with open(f, 'rb') as file:
                res = load(file)
            res = IntegrationResults.update_class_object(res)

            if 'unb' in force_type:
                if mod is None:
                    mod = '05x0_'

                unb_base = 6350 * res.system.rotor.rotor_solo_disks.m / (unbalance_speed * 60 / (2 * np.pi))
                force_units = 'g-mm'
                force_units_legend = 'U'
                amp_units = f'micron ({amplitude_units}) / ' + force_units
                corr1 = 1
                corr2 = f0 * unb_base / 1e6
                
            else:
                if mod is None:
                    mod = '2x0_'

                force_units = 'N'
                force_units_legend = force_units
                amp_units = f'm ({amplitude_units}) / ' + force_units
                corr1 = 1
                corr2 = f0
            
            fig_aux = res.plot_frf(dof=dof,
                                    amplitude_units=amplitude_units)
            name = f'{arr[i]} {param}'        
            fig_aux.data[0].name = name
            fig_aux.data[0].y = fig_aux.data[0].y / corr2

            if i == 0:
                if 'linear_results' in res.__dict__:
                    fig_lin = res.linear_results.plot_frf(dof=dof,
                                                            whirl=force_type,
                                                            amplitude_units=amplitude_units)
                    fig_lin.data[0].line.color = 'blue'
                    fig_lin.data[0].y = fig_lin.data[0].y / corr1
                    fig_lin.data[0].name = 'Linear'

                if 'rigid_results' in res.__dict__:
                    fig_rig = res.rigid_results.plot_frf(dof=dof,
                                                        whirl=force_type,
                                                        amplitude_units=amplitude_units)
                    fig_rig.data[0].line.color = 'black'
                    fig_rig.data[0].line.dash = 'dash'
                    fig_rig.data[0].y = fig_rig.data[0].y / corr1
                    fig_rig.data[0].name = 'Rigid Disks'
                
                if fig_lin and plot_linear:
                    plot_linear = False
                    fig = fig_lin

                if fig_rig and plot_rigid:
                    plot_rigid = False
                    if len(fig.data) == 0:
                        fig = fig_rig
                    else:
                        fig.add_trace(fig_rig.data[0])

                if len(fig.data) == 0:
                    fig = fig_aux
                else:
                    fig.add_trace(fig_aux.data[0])

            else:
                fig.add_trace(fig_aux.data[0])
        else:
            print(f'File {f} not found.')

    fig.update_layout(yaxis_title=amp_units,
                          width=900)
    if 'unb' in force_type:
        fig.update_layout(title_text='Unbalance Response')
    elif 'for' in force_type:
        fig.update_layout(title_text='Forward Excitation')
    else:
        fig.update_layout(title_text='Backward Excitation')

    return fig
    


def plot_frf_set(force_type,
                 tun,
                 mod=None,
                 amplitude_units='rms',
                 unbalance_speed=377,
                 dof=['last_x'],
                 remove_list=[],
                 plot_linear=True,
                 plot_rigid=True):
    
    if type(dof) != list:
        dof = [dof]
              
    filename_base = f'{mod}results_{tun}_{force_type}'
    
    picfiles = [f for f in listdir(getcwd()) if (isfile(join(getcwd(), f)) and '.pic' in f and f.startswith(filename_base) and all([str(s) not in f for s in remove_list]))]
    print(picfiles)
    
    f0 = []
    fig = go.Figure()    

    for i, f in enumerate(picfiles):

        print(f)

        f0.append(float(f[len(filename_base) + 2 : -4]))
        print(f0[-1])

        with open(f, 'rb') as file:
            res = load(file)
        res = IntegrationResults.update_class_object(res)

        if 'unb' in force_type:
            if mod is None:
                mod = '05x0_'

            unb_base = 6350 * res.system.rotor.rotor_solo_disks.m / (unbalance_speed * 60 / (2 * np.pi))
            force_units = 'g-mm'
            force_units_legend = 'U'
            amp_units = f'micron ({amplitude_units}) / ' + force_units
            corr1 = 1
            corr2 = f0[-1] * unb_base / 1e6
            
        else:
            if mod is None:
                mod = '2x0_'

            force_units = 'N'
            force_units_legend = force_units
            amp_units = f'm ({amplitude_units}) / ' + force_units
            corr1 = 1
            corr2 = f0[-1]
           
        fig_aux = res.plot_frf(dof=dof,
                                amplitude_units=amplitude_units)
        name = f'{f0[-1]} {force_units_legend}'        
        fig_aux.data[0].name = name
        fig_aux.data[0].y = fig_aux.data[0].y / corr2

        if i == 0:
            if 'linear_results' in res.__dict__:
                fig_lin = res.linear_results.plot_frf(dof=dof,
                                                        whirl=force_type,
                                                        amplitude_units=amplitude_units)
                fig_lin.data[0].line.color = 'blue'
                fig_lin.data[0].y = fig_lin.data[0].y / corr1
                fig_lin.data[0].name = 'Linear'

            if 'rigid_results' in res.__dict__:
                fig_rig = res.rigid_results.plot_frf(dof=dof,
                                                    whirl=force_type,
                                                    amplitude_units=amplitude_units)
                fig_rig.data[0].line.color = 'black'
                fig_rig.data[0].line.dash = 'dash'
                fig_rig.data[0].y = fig_rig.data[0].y / corr1
                fig_rig.data[0].name = 'Rigid Disks'
            
            if fig_lin and plot_linear:
                plot_linear = False
                fig = fig_lin

            if fig_rig and plot_rigid:
                plot_rigid = False
                if len(fig.data) == 0:
                    fig = fig_rig
                else:
                    fig.add_trace(fig_rig.data[0])

            if len(fig.data) == 0:
                fig = fig_aux
            else:
                fig.add_trace(fig_aux.data[0])

        else:
            fig.add_trace(fig_aux.data[0])

    fig.update_layout(yaxis_title=amp_units,
                          width=900)
    if 'unb' in force_type:
        fig.update_layout(title_text='Unbalance Response')
    elif 'for' in force_type:
        fig.update_layout(title_text='Forward Excitation')
    else:
        fig.update_layout(title_text='Backward Excitation')

    return fig

def plot_frfs(lista,
              whirl,
              tun,
              dof,
              mod_unb='05x0_',
             speed=377,
             amplitude_units='rms'):

    if whirl == 'unbalance':        
        mod = mod_unb
        lin_f = 0.1
    else:
        mod = '2x0_'
        lin_f = 10
    
    res = []
    for i in lista:
        if i == 'linear':
            try:
                res.append(get_results(f=lin_f,
                                      tun=tun,
                                      whirl=whirl,
                                      mod=mod)
                                      )
            except:
                res.append(get_results(f=lin_f,
                                      tun=tun,
                                      whirl=whirl,
                                      mod='05x0_')
                                      )
            
        elif i == 'bare' or i == 'rigid':
            if 'trans' in tun:
                tun_rigid = 'transtun'
            else:
                tun_rigid = 'flextun'
            res.append(get_results(f=lin_f,
                                  tun=tun_rigid,
                                  whirl=whirl,
                                   mod='rigid_'))
            

        else:
            res.append(get_results(f=i,
                                  tun=tun,
                                  whirl=whirl,
                                  mod=mod))

    if whirl == 'unbalance':
        unb_base = 6350 * res[0].system.rotor.rotor_solo_disks.m / (speed * 60 / (2 * np.pi))
        force_units = 'g-mm'
        force_units_legend = 'U'
        amp_units = f'micron ({amplitude_units}) / ' + force_units
        corr = unb_base / 1e6#(np.sqrt(2) * 2 * 1e6)
        
    else:
        force_units = 'N'
        force_units_legend = force_units
        amp_units = f'm ({amplitude_units}) / ' + force_units
        corr = 1 / 1#(np.sqrt(2) * 2)
    
    for j, i in enumerate(lista):
        if i == 'linear':
            fig_aux = res[j].plot_frf(dof=[dof],
                                     amplitude_units=amplitude_units)
            div = lin_f * corr
            name = 'Linear'             
            fig_aux.data[0].line.color = 'blue'
            
        elif i == 'bare' or i == 'rigid':           
            fig_aux = res[j].plot_frf(dof=[dof],
                                     amplitude_units=amplitude_units)
            div = lin_f * corr
            name = 'Bare rotor'
            fig_aux.data[0].line.color = 'black'
            fig_aux.data[0].line.dash = 'dash'

        else:            
            fig_aux = res[j].plot_frf(dof=[dof],
                                     amplitude_units=amplitude_units)
            div = i * corr
          
            name = f'{i} {force_units_legend}'
            
        if j == 0:
            fig = fig_aux
        else:
            fig.add_trace(fig_aux.data[0])
            
        fig.data[-1].y = fig_aux.data[0].y / div
        fig.data[-1].name = name
        fig.update_layout(yaxis_title=amp_units)
    
    return fig
