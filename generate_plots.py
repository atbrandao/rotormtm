import rotor_mtm as rmtm
import glob, os
import pickle
import numpy as np
import plotly.graph_objects as go

files = glob.glob('results/*.pic')
for i, f in enumerate(files):
    print(f'{i} - {f}')

i_ls = [0,1,3,4,5,6,7]
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

        fig = rmtm.plot_diff_modal(data['w'], data['diff'], sp_arr, mode='abs', saturate=10)
        fig.add_trace(go.Scatter(x=sp_arr,y=sp_arr,mode='lines',
                                 line=dict(dash='dash',color='black'),
                                 showlegend=False))
        fig.update_layout(yaxis=dict(range=[f_min, sp_arr[-1]]))
        fig.write_image(f'results/diff_modal_{name}_abs.pdf')
        fig = rmtm.plot_diff_modal(data['w'], data['diff'], sp_arr, mode='phase')
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
            z = np.mean(np.angle(np.array(data['diff_map_b'])) * 180 / (np.pi),
                        axis=2)
            z[z < 0] += 360
           
            fig = rmtm.plot_camp_heatmap(z, None, sp_arr, f_min=f_min, colorbar_title='Phase [deg]', phase_angle=True)
            fig.update_layout(yaxis=dict(range=[f_min, sp_arr[-1]],title='Excitation Frequency [rad/s]'),title='Average resonator phase angle heatmap')
            fig.write_image(f'results/camp_diff_heatmap_phase_{name}.pdf')

            r = np.log10(np.mean(np.abs(np.array(data['diff_map_b'])),axis=2))
            fig = rmtm.plot_camp_heatmap(r, None, sp_arr, f_min=f_min, colorbar_title='Amplific. (log[m])')
            fig.update_layout(yaxis=dict(range=[f_min, sp_arr[-1]],title='Excitation Frequency [rad/s]'),title='Average resonator amplification heatmap')
            fig.write_image(f'results/camp_diff_heatmap_abs_{name}.pdf')

        # Plot results for solo rotor
        if i == 0:
            r = (data['rsolo_b_map'])
            fig = rmtm.plot_camp_heatmap(r, None, sp_arr, f_min=f_min, colorbar_title='Response [m]')
            fig.update_layout(yaxis=dict(range=[f_min, sp_arr[-1]],title='Excitation Frequency [rad/s]'),title='Response heatmap')
            fig.write_image(f'results/camp_heatmap_solo.pdf')
            r = np.log10(data['rsolo_b_map'])
            fig = rmtm.plot_camp_heatmap(r, None, sp_arr, f_min=f_min, colorbar_title='Response (log [m])')
            fig.update_layout(yaxis=dict(range=[f_min, sp_arr[-1]],title='Excitation Frequency [rad/s]'),title='Response heatmap')
            fig.write_image(f'results/camp_heatmap_log_solo.pdf')


