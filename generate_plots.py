import rotor_mtm as rmtm
import glob, os
import pickle
import numpy as np
import plotly.graph_objects as go

files = glob.glob('results/*.pic')

i = 4

sp_arr = np.linspace(1,800,200)

with open(files[i],'rb') as f:
    data = pickle.load(f)
    name = os.path.split(files[i])[1][11:-4]
    fig = rmtm.plot_diff_modal(data['w'], data['diff'], sp_arr, mode='abs', saturate=10)
    fig.add_trace(go.Scatter(x=sp_arr,y=sp_arr,mode='lines',
                             line=dict(dash='dash',color='black'),
                             showlegend=False))
    fig.write_image(f'results/diff_modal_{name}.pdf')




#
# for k in rotor_dict.keys():
#     if 'flex' in k:
#         dof = 2 # Select the flexural DoF to calculate diff
#     else:
#         dof = 0 # Select the translation DoF to calculate diff
#     print(f'Running analysis for rotor {k}:')
#     out = rotor_dict[k].run_analysis(sp_arr,diff_lim=diff_lim,diff_analysis=True,heatmap=True,dof=dof)
#     with open(f'filename_{k}.pic', 'wb') as handle:
#         pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

#
# N = r_var_flextun.N
# diff_lim = -r_var_flextun.A(0)[N+n_pos[0]*4+2,n_pos[0]*4+2]/k1
# out = r_var_flextun.run_analysis(sp_arr,diff_lim=diff_lim)
# w_var = out['w']
# w_var_res = out['w_res']
# r_var_map = out['r_b_map']
# rmtm.plot_camp_heatmap(r_var_map,w_var,sp_arr,w_res=w_var_res).write_image('Heatmap_var_fow_flextunTESTE.png',scale=8)
#
# r_det_sbg = rmtm.RotorMTM(rotor,n_pos,dk_r,k0,k1,var=0,var_k=0,p_damp=1e-4,ge=True)
# r_var3_sbg = rmtm.RotorMTM(rotor,n_pos,dk_r,k0,k1,var=var,var_k=0,p_damp=1e-4,ge=True,exp_var=3)
#
# r_det_transtun = rmtm.RotorMTM(rotor,n_pos,dk_r,k0/100,100*k1,var=0,var_k=0,p_damp=1e-4,ge=True)
# r_var1_transtun = rmtm.RotorMTM(rotor,n_pos,dk_r,k0/100,100*k1,var=var,var_k=0,p_damp=1e-4,ge=True,exp_var=1)
# r_var3_transtun = rmtm.RotorMTM(rotor,n_pos,dk_r,k0/100,100*k1,var=var,var_k=0,p_damp=1e-4,ge=True,exp_var=3)
#
# out_var3_sbg = r_var3_sbg.run_analysis(sp_arr,diff_lim=diff_lim)
# out_var3_transtun = r_var3_transtun.run_analysis(sp_arr,diff_lim=diff_lim)
# rmtm.plot_frf([out_var3_sbg['rsolo_b'],out_var3_sbg['r_b'],out_var3_transtun['r_b']],sp_arr)
#
# out_diff = r_var3_flextun.run_analysis(sp_arr,diff_lim=diff_lim,diff_analysis=True,dof=2,heatmap=True)
# rmtm.plot_diff_modal(out_diff['w'],out_diff['diff'],sp_arr,mode='abs')