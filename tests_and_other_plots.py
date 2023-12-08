import ross as rs
from multiprocessing import Pool
import rotor_mtm as rmtm
import glob, os
import pickle
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot

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
#
# def save_full_out(k):
#     print(f'Running analysis for rotor {k}:')
#     if 'flex' in k:
#         dof = 2 # Select the flexural DoF to calculate diff
#     else:
#         dof = 0 # Select the translation DoF to calculate diff
#     out = rotor_dict[k].run_analysis(sp_arr, diff_lim=diff_lim, diff_analysis=True,
#                                      heatmap=False, dof=dof, dof_show=dof_show)
#
#     with open(f'out_data_{k}_{dof_show}_simple.pic', 'wb') as handle:
#         pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

# dof_show = 0
# save_full_out('r_var1_flextun')

# Nova linha

#
# files = ['out_data_r_var1_flextun_0_simple.pic']
# i = 0
# with open('out_data_r_var1_flextun_0_simple.pic','rb') as f:
#     f_min = 40
#     data = pickle.load(f)
#     fig = rmtm.plot_campbell(w=data['ws'], sp_arr=sp_arr)
#     fig.write_image('campbell_solo.svg')
#
#     sp = [(0,62,3e2),(1,62,3e2),(2,193,1e3),(3,207,1e3),(4,377,3e3),(5,433,3e3)]
#     for k in sp:
#         _,_,u_list,*_ = rotor_dict['r_det_flextun'].omg_list(k[1],rotor_solo=True)
#         u = u_list[k[0]] * k[2]
#         fig = rmtm.plot_deflected_shape(rotor_dict['r_det_flextun'].rotor_solo_disks,u,[],'trans',isometric=True)
#         fig.update_layout(xaxis=dict(showgrid=False,showline=False,showticklabels=False),
#                           # zaxis=dict(showgrid=False,showline=False,showticklabels=False),
#                           yaxis=dict(showgrid=False,showline=False,showticklabels=False))
#
#         fig.write_image(f'mode_shape_solo_{k[0]}.svg')
#
# Generate heatmap and deflected shapes for transtun
files = ['results/out_data_r_var1_transtun.pic']
i = 0
with open(files[i], 'rb') as f:
    f_min = 40
    data = pickle.load(f)
    r = np.log10(data['r_b_map'])
    fig1 = rmtm.plot_camp_heatmap(r, None, sp_arr, f_min=f_min, colorbar_title='Response (log [m])')
    fig1.update_layout(yaxis=dict(range=[f_min, sp_arr[-1]],title='Excitation Frequency (rad/s)'), title='Response heatmap')


    sp = [(62, 62, 1e4), (190, 190, 5e4), (325, 325, 8e4), (377, 377, 1e5), (500, 500, 3e5), (590, 590, 2e5*np.exp(1.j*np.pi/2))]
    for k in sp:
        fig1.add_trace(go.Scatter(x=[k[0]],y=[k[1]],mode='markers',marker=dict(color='blue'),showlegend=False))
        u_f, u_b = rotor_dict['r_var1_transtun'].x_out(k[0],k[1])
        u = u_b * k[2]
        fig = rmtm.plot_deflected_shape(rotor_dict['r_var1_transtun'].rotor_solo_disks, u, n_pos, 'trans',
                                        isometric=True)
        fig.update_layout(xaxis=dict(range=(-0.5, 1.5), showgrid=False, showline=False, showticklabels=False),
                          # zaxis=dict(showgrid=False,showline=False,showticklabels=False),
                          yaxis=dict(range=(-0.5, 1.5), showgrid=False, showline=False, showticklabels=False))

        # fig.write_image(f'mode_shape_var1_transtun_{k[0]}.svg', width=1200, height=1100)
    fig1.write_image(f'camp_heatmap_log_var1_transtun.svg')


# Generate heatmap and deflected shapes for flextun
files = ['results/out_data_r_var3_flextun.pic']
i = 0
with open(files[i], 'rb') as f:
    f_min = 40
    data = pickle.load(f)
    r = np.log10(data['r_b_map'])
    fig1 = rmtm.plot_camp_heatmap(r, None, sp_arr, f_min=f_min, colorbar_title='Response (log [m])')
    fig1.update_layout(yaxis=dict(range=[f_min, sp_arr[-1]],title='Excitation Frequency (rad/s)'), title='Response heatmap')


    sp = [(62, 62, 1e4), (190, 190, 1e4), (325, 325, 8e3), (377, 377, 1e4), (415, 415, 1e4), (590, 590, 5e4*np.exp(1.j*np.pi/2))]
    for k in sp:
        fig1.add_trace(go.Scatter(x=[k[0]],y=[k[1]],mode='markers',marker=dict(color='blue'),showlegend=False))
        u_f, u_b = rotor_dict['r_var3_flextun'].x_out(k[0],k[1])
        u = u_b * k[2]
        fig = rmtm.plot_deflected_shape(rotor_dict['r_var3_flextun'].rotor_solo_disks, u, n_pos, 'flex',
                                        isometric=True)
        fig.update_layout(xaxis=dict(range=(-0.5, 1.5), showgrid=False, showline=False, showticklabels=False),
                          # zaxis=dict(showgrid=False,showline=False,showticklabels=False),
                          yaxis=dict(range=(-0.5, 1.5), showgrid=False, showline=False, showticklabels=False))

        # fig.write_image(f'mode_shape_var3_flextun_{k[0]}.svg', width=1200, height=1100)
    fig1.write_image(f'camp_heatmap_log_var3_flextun.svg')

# Generate heatmap and deflected shapes for solo and flex or trans dofs
files = ['results/out_data_r_var3_flextun.pic']
i = 0
with open(files[i], 'rb') as f:
    f_min = 40
    data = pickle.load(f)
    r = np.log10(data['rsolo_b_map'])
    fig1 = rmtm.plot_camp_heatmap(r, None, sp_arr, f_min=f_min, colorbar_title='Response (log [m])')
    fig1.update_layout(yaxis=dict(range=[f_min, sp_arr[-1]], title='Excitation Frequency (rad/s)'),
                       title='Response heatmap')
    # flex dof
    # sp = [(62, 62, 1e4), (190, 190, 3e4), (377, 377, 5e4),
    #       (580, 580, 5e4 * np.exp(1.j * np.pi / 2))]
    #trans dof
    sp = [(62, 62, 2e4), (190, 190, 5e4 * np.exp(1.j * np.pi/2)), (377, 377, 8e4 * np.exp(1.j * np.pi)),
          (580, 580, 1e5)]
    for k in sp:
        fig1.add_trace(go.Scatter(x=[k[0]], y=[k[1]], mode='markers', marker=dict(color='blue'), showlegend=False))
        u_f, u_b = rotor_dict['r_var3_flextun'].x_out(k[0], k[1], rotor_solo=True)
        u = u_b * k[2]
        fig = rmtm.plot_deflected_shape(rotor_dict['r_var3_flextun'].rotor_solo_disks, u, [], 'trans',
                                        isometric=True)
        fig.update_layout(xaxis=dict(range=(-0.5, 1.5), showgrid=False, showline=False, showticklabels=False),
                          # zaxis=dict(showgrid=False,showline=False,showticklabels=False),
                          yaxis=dict(range=(-0.5, 1.5), showgrid=False, showline=False, showticklabels=False))

        # fig.write_image(f'mode_shape_solo_trans_{k[0]}.svg', width=1200, height=1100)
    fig1.write_image(f'camp_heatmap_log_solo_trans.svg')

files = ['results/out_data_r_var3_flextun.pic']
i = 0
with open(files[i], 'rb') as f:
    f_min = 40
    data = pickle.load(f)
    r = np.log10(data['rsolo_b_map'])
    fig1 = rmtm.plot_camp_heatmap(r, None, sp_arr, f_min=f_min, colorbar_title='Response (log [m])')
    fig1.update_layout(yaxis=dict(range=[f_min, sp_arr[-1]], title='Excitation Frequency (rad/s)'),
                       title='Response heatmap')
    # flex dof
    sp = [(62, 62, 1e4), (190, 190, 3e4), (377, 377, 5e4),
          (580, 580, 5e4 * np.exp(1.j * np.pi / 2))]
    # trans dof
    # sp = [(62, 62, 2e4), (190, 190, 5e4 * np.exp(1.j * np.pi / 2)), (377, 377, 8e4 * np.exp(1.j * np.pi)),
    #       (580, 580, 1e5)]
    for k in sp:
        fig1.add_trace(go.Scatter(x=[k[0]], y=[k[1]], mode='markers', marker=dict(color='blue'), showlegend=False))
        u_f, u_b = rotor_dict['r_var3_flextun'].x_out(k[0], k[1], rotor_solo=True)
        u = u_b * k[2]
        fig = rmtm.plot_deflected_shape(rotor_dict['r_var3_flextun'].rotor_solo_disks, u, [], 'flex',
                                        isometric=True)
        fig.update_layout(xaxis=dict(range=(-0.5, 1.5), showgrid=False, showline=False, showticklabels=False),
                          # zaxis=dict(showgrid=False,showline=False,showticklabels=False),
                          yaxis=dict(range=(-0.5, 1.5), showgrid=False, showline=False, showticklabels=False))

        # fig.write_image(f'mode_shape_solo_flex_{k[0]}.svg', width=1200, height=1100)
    fig1.write_image(f'camp_heatmap_log_solo_flex.svg')

    # name = os.path.split(files[i])[1][11:-9]
    # fig = rmtm.plot_diff_modal(data['w'], data['diff'], sp_arr, mode='abs', saturate=10)
    # fig.add_trace(go.Scatter(x=sp_arr,y=sp_arr,mode='lines',
    #                          line=dict(dash='dash',color='black'),
    #                          showlegend=False))
    # fig.update_layout(yaxis=dict(range=[f_min, sp_arr[-1]]))
    # fig.write_image(f'results/diff_modal_{name}_abs.pdf')
    # fig = rmtm.plot_diff_modal(data['w'], data['diff'], sp_arr, mode='phase')
    # fig.add_trace(go.Scatter(x=sp_arr, y=sp_arr, mode='lines',
    #                          line=dict(dash='dash', color='black'),
    #                          showlegend=False))
    # fig.update_layout(yaxis=dict(range=[f_min, sp_arr[-1]]))
    # fig.write_image(f'results/diff_modal_{name}_phase.pdf')