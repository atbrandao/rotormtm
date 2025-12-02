import rotor_mtm as rmtm
import glob, os
import pickle
import numpy as np
import plotly.graph_objects as go

n_res = 15
n_center = 15

n_pos = np.arange(n_center-int(n_res/2),n_center+n_res-int(n_res/2),1)

files = glob.glob('results/*.pic')
for i, f in enumerate(files):
    print(f'{i} - {f}')

i_ls = [0,1,3,4,5,6]
i_ls_frf = [0,3]

sp_arr = np.linspace(1,800,200)
N = len(sp_arr)

for i in i_ls_frf:
    with open(files[i],'rb') as f:
        data = pickle.load(f)
        name = os.path.split(files[i])[1][11:-4]

        r = [np.array([data['rsolo_map'][j,j] for j in range(N)]).reshape(N,1),
             np.array([data['r_map'][j,j] for j in range(N)]).reshape(N,1)]

        r_b = [np.array([data['rsolo_b_map'][j, j] for j in range(N)]).reshape(N,1),
             np.array([data['r_b_map'][j, j] for j in range(N)]).reshape(N,1)]

    if 'flex' in files[i]:
        str_sub = 'var1'
        fname = 'flex'
    else:
        str_sub = 'var1'
        fname = 'lin'
    with open(files[i].replace('det',str_sub),'rb') as f:
        data = pickle.load(f)
        name = os.path.split(files[i])[1][11:-4]

        r.append(np.array([data['r_map'][j,j] for j in range(N)]).reshape(N,1))
        r_b.append(np.array([data['r_b_map'][j, j] for j in range(N)]).reshape(N,1))

    fig = rmtm.plot_frf(r,sp_arr)
    fig.update_layout(yaxis=dict(range=[-8,-4.5]))
    fig.write_image(f'Forward_excitation_{fname}.pdf')
    fig.write_html(f'Forward_excitation_{fname}.html')
    fig = rmtm.plot_frf(r_b, sp_arr)
    fig.update_layout(yaxis=dict(range=[-8, -4.5]))
    fig.write_image(f'Backward_excitation_{fname}.pdf')
    fig.write_html(f'Backward_excitation_{fname}.html')

for i in i_ls:
    with open(files[i],'rb') as f:
        f_min = 40
        data = pickle.load(f)
        name = os.path.split(files[i])[1][11:-4]

        if 'trans' in name:
            dof = 0
        else:
            dof = 2

        n_modes = 50
        diff = []
        for j in range(len(sp_arr)):
            x_mode = np.array([data['u'][j][a][dof : 184 : 4] for a in range(n_modes)])
            y_mode = np.array([data['u'][j][a][dof + 1 : 184 : 4] for a in range(n_modes)])
            fow_mode = 1 / 2 * (x_mode + 1.j * y_mode)
            back_mode = 1 / 2 * (x_mode - 1.j * y_mode)

            # aux = np.array([rmtm.res_diff(fow_mode[k, :], n_pos) for k in range(n_modes)])
            # aux = np.array([rmtm.res_diff(back_mode[k, :], n_pos) for k in range(n_modes)])
            aux = np.array([rmtm.res_diff(x_mode[k, :], n_pos) for k in range(n_modes)])
            # aux = np.array([rmtm.res_diff(y_mode[k, :], n_pos) for k in range(n_modes)])
            aux = np.real(aux) - 1.j * np.imag(aux)

            diff.append(aux)
        
        # diff = np.array(diff)

        fig = rmtm.plot_diff_modal(data['w'], data['diff'], sp_arr, mode='abs', saturate=10)
        # fig = rmtm.plot_diff_modal(data['w'], diff, sp_arr, mode='abs', saturate=10)
        fig.add_trace(go.Scatter(x=sp_arr,y=sp_arr,mode='lines',
                                 line=dict(dash='dash',color='black'),
                                 showlegend=False))
        fig.update_layout(yaxis=dict(range=[f_min, sp_arr[-1]]))
        fig.write_image(f'results/diff_modal_{name}_abs.pdf')

        fig = rmtm.plot_diff_modal(data['w'], data['diff'], sp_arr, mode='phase')
        # fig = rmtm.plot_diff_modal(data['w'], diff, sp_arr, mode='phase')
        fig.add_trace(go.Scatter(x=sp_arr, y=sp_arr, mode='lines',
                                 line=dict(dash='dash', color='black'),
                                 showlegend=False))
        fig.update_layout(yaxis=dict(range=[f_min, sp_arr[-1]]))
        fig.write_image(f'results/diff_modal_{name}_phase.pdf')

        r = (data['r_b_map'])
        fig = rmtm.plot_camp_heatmap(r, None, sp_arr,f_min=f_min,colorbar_title='Response [m]')
        fig.update_layout(yaxis=dict(range=[f_min,sp_arr[-1]],title='Excitation Frequency (rad/s)'),title='Response heatmap')
        fig.write_image(f'results/camp_heatmap_{name}.pdf')
        r = np.log10(data['r_b_map'])
        fig = rmtm.plot_camp_heatmap(r, None, sp_arr, f_min=f_min, colorbar_title='Response (log [m])')
        fig.update_layout(yaxis=dict(range=[f_min, sp_arr[-1]],title='Excitation Frequency [rad/s]'),title='Response heatmap')
        fig.write_image(f'results/camp_heatmap_log_{name}.pdf')

        if 'trans' in name or ('flex' in name and '2' in name):
            for excitation in ['backward', 'forward']:
                key = 'diff_map'
                if excitation == 'backward':
                    key += '_b'
                    # data[key] = np.conj(np.array(data[key])) #Usar somente nos resultados de backup de jan-2025

                z = np.mean(np.angle(np.array(data[key])), axis=2) * 180 / np.pi
                z = z[:, :, 0]
                
                # z[z < 0] += 360
            
                fig = rmtm.plot_camp_heatmap(z, None, sp_arr, f_min=f_min, colorbar_title='Phase [deg]', phase_angle=True)
                fig.update_layout(yaxis=dict(range=[f_min, sp_arr[-1]],title='Excitation Frequency [rad/s]'),title='Average resonator phase angle heatmap')
                fig.write_image(f'results/camp_diff_heatmap_phase_{name}_{excitation}.pdf')

                r = np.log10(np.mean(np.abs(np.array(data[key])), axis=2))
                r = r[:, :, 0]
                # print(r.shape)
                fig = rmtm.plot_camp_heatmap(r, None, sp_arr, f_min=f_min, colorbar_title='Amplific. (log[m])')
                fig.update_layout(yaxis=dict(range=[f_min, sp_arr[-1]],title='Excitation Frequency [rad/s]'),title='Average resonator amplification heatmap')
                fig.write_image(f'results/camp_diff_heatmap_abs_{name}_{excitation}.pdf')

        # Plot results for solo rotor
        if i == 0:
            r = (data['rsolo_b_map'])
            # print(r.shape)
            fig = rmtm.plot_camp_heatmap(r, None, sp_arr, f_min=f_min, colorbar_title='Response [m]')
            fig.update_layout(yaxis=dict(range=[f_min, sp_arr[-1]],title='Excitation Frequency [rad/s]'),title='Response heatmap')
            fig.write_image(f'results/camp_heatmap_solo.pdf')

            r = np.log10(data['rsolo_b_map'])
            fig = rmtm.plot_camp_heatmap(r, None, sp_arr, f_min=f_min, colorbar_title='Response (log [m])')
            fig.update_layout(yaxis=dict(range=[f_min, sp_arr[-1]],title='Excitation Frequency [rad/s]'),title='Response heatmap')
            fig.write_image(f'results/camp_heatmap_log_solo.pdf')


