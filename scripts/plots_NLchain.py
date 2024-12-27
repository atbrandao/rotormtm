# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 09:40:14 2022

@author: HR7O
"""


import harmbal as hb
import numpy as np
from pickle import load, dump
import plotly.graph_objects as go
from plotly_resampler import FigureResampler, FigureWidgetResampler
from plotly.subplots import make_subplots



def plots(f2, mass_slope):
    
    N = 20
    mass = 100
    L = 1
    
    d = L/N
    k = 8e4/d
    cp = 0.0002
    m = mass/N
    
    m_res = m/2
    
    x0 = d/2
    w_res = 250
    N_res = 10 # number of resonators
    n_start = 5
    
    slope = np.linspace(0.7,1.3,N_res)
    
    beta = w_res**2*m_res/2
    alpha = beta/x0**2
    delta = 2 * cp * 2 * beta
    
    if mass_slope:
        M_res = list(slope * (N_res*[m_res]))
        M = np.array(N*[m] + M_res) * np.eye(N+N_res)
    else:
        M = np.array(N * [m] + N_res * [m_res]) * np.eye(N + N_res)
    K = np.array(N*[2*k]+N_res*[0]) * np.eye(N+N_res)
    Snl = 0 * K
    
    for i in range(N-1):
        K[i, i + 1] = -k
        K[i + 1, i] = -k
    
    for i in range(N_res):
        dof = [n_start + i, N + i]
        k_el = np.array([[1, -1], [-1, 1]])
        # K[np.ix_(dof, dof)] += k_el * 2 * beta
        Snl[np.ix_(dof, dof)] += k_el
    
    C = cp * K + delta * Snl
    
    sys = hb.Sys_NL(M=M, K=K, Snl=Snl, beta=-beta, alpha=alpha, n_harm=10, nu=1, C=C)
    sys_lin = sys.eq_linear_system()
    sys_base = hb.Sys_NL(M=M, K=K, Snl=Snl, beta=-10*beta, alpha=alpha, n_harm=10, nu=1, C=C).eq_linear_system()
    
    f = 1000.0
    omg = 70
    
    omg_range = np.arange(1, 500, 1)
    base_x_last = []
    base_x_base = []
    base_x_res = []
    base_x_diff = []
    base_rms_ke_nl = []
    base_rms_ke_chain = []
    base_rms_pe_nl = []
    base_rms_pe_chain = []
    base_rms_p_in = []
    base_rms_x_last = []
    base_rms_x = []
    
    excit_w = []
    resp_w = []
    
    ####
    # Getting data from bare chain
    
    nu = 8
    w_max = 2000
    for omg in omg_range:
    
        with open(f'D:/Dados Doutorado/NL_Chain/raw data bare chain/f_{f}_omg_{omg}_hb.pic', 'rb') as file:
    #     with open(f'D:/Dados Doutorado/NL_Chain/raw data/f_{f}_omg_{omg}_baseear.pic', 'rb') as file:
            ls = load(file)
            t_hb = sys_base.t(omg)[:, 0] # ls[0]
            x_hb = ls[1]
            dt = t_hb[1]
            N = len(t_hb)
            
        T = 2 * np.pi / omg
        N_per = np.round((dt * N/nu) / T)
        N_int = int(np.round(N_per * T / dt))
            
        F = np.zeros((sys_base.ndof, len(t_hb)))
        F[0, :] = np.real(f) * np.cos(omg * t_hb) + np.imag(f) * np.sin(omg * t_hb)
    
        p_in = sys_base.power_in(x_hb, F)
        
    #     ke = sys.kinetic_energy(x_rk4, [a for a in range(20)])
    #     ke_chain = sys.kinetic_energy(x_rk4, [a for a in range(20)])
        ke_chain = sys_base.kinetic_energy(x_hb, [a for a in range(10, 20)])
        ke_nl = sys_base.kinetic_energy(x_hb, [a for a in range(20, sys.ndof)])
        
    #     pe_nl = sys_base.dof_nl_potential_energy(x_rk4)
    #     pe_nl -= np.mean(pe_nl)
    #     pe_chain = sys_base.base_potential_energy(x_rk4, [a for a in range(20)])
    #     pe_chain = pe_chain - np.mean(pe_chain)
        pe_nl = sys_base.base_potential_energy(x_hb, [a for a in range(20, sys_base.ndof)])
        pe_nl -= np.mean(pe_nl)
        pe_chain = sys_base.base_potential_energy(x_hb, [a for a in range(10, 20)])
        pe_chain = pe_chain - np.mean(pe_chain)
        
    #     pe = sys_base.base_potential_energy(x_rk4, [a for a in range(20, sys_base.ndof)])
    #     pe -= np.mean(pe)
        
    #     w_max = 1 / (2 * dt) * 2 * np.pi
        dw = 1 / (t_hb[-1]) * 2 * np.pi # Considerando parte da série temporal
        
        w = np.arange(dw, w_max, dw)
        resp_w += list(w)
        excit_w += [omg] * len(w)
            
        base_rms_ke_nl.append(np.sqrt(np.mean(ke_nl ** 2)) / len(sys_base.dof_nl))
        base_rms_ke_chain.append(np.sqrt(np.mean(ke_chain ** 2)) / 10)
        base_rms_pe_nl.append(np.sqrt(np.mean(pe_nl ** 2)) / len(sys_base.dof_nl))
        base_rms_pe_chain.append(np.sqrt(np.mean(pe_chain ** 2)) / 10)
        base_rms_p_in.append(np.sqrt(np.mean((p_in[0, :]/omg) ** 2)) / sys_base.ndof)
        base_rms_x_last.append(np.sqrt(np.mean(x_hb[19, :] ** 2)))
        base_rms_x.append(np.sqrt(np.mean(x_hb[:sys_base.ndof, :] ** 2, 1)))
        
        base_x_last += list(2 / (N) * np.abs(np.fft.fft(x_hb[19, :]))[1:len(w)+1])
        base_x_last += [0] * (len(resp_w) - len(base_x_last))
        base_x_base += list(2 / (N) * np.abs(np.fft.fft(x_hb[sys.base_dof[0], :]))[1:len(w)+1])
        base_x_base += [0] * (len(resp_w) - len(base_x_base))
        base_x_res += list(2 / (N) * np.abs(np.fft.fft(x_hb[sys.dof_nl[0], :]))[1:len(w)+1])
        base_x_res += [0] * (len(resp_w) - len(base_x_res))
        base_x_diff += list(np.mean((np.fft.fft(x_hb[sys.dof_nl, :])/np.fft.fft(x_hb[sys.base_dof, :])), 0)[1:len(w)+1])
        base_x_diff += [0] * (len(resp_w) - len(base_x_diff))
    #     print(omg)
    
    base_rms_ke_chain = np.array(base_rms_ke_chain)
    base_rms_ke_nl = np.array(base_rms_ke_nl)
    base_rms_pe_nl = np.array(base_rms_pe_nl)
    base_rms_pe_chain = np.array(base_rms_pe_chain)
    base_rms_p_in = np.array(base_rms_p_in)
    base_rms_x_last = np.array(base_rms_x_last)
    base_rms_x = np.array(base_rms_x)
    base_x_last = np.array(base_x_last)
    base_x_base = np.array(base_x_base)
    base_x_res = np.array(base_x_res)
    base_x_diff = np.array(base_x_diff)
    base_x_angle = np.abs(np.angle(base_x_diff))
    
    base_x_angle = np.abs(np.angle(base_x_diff))
    base_x_angle[-1] = 0
    base_x_angle[base_x_angle<0] += 2 * np.pi
    
    #####
    ## Getting data from linear resonators    
    
    f = 1000.0
    omg = 70
    
    omg_range = np.arange(1, 500, 1)
    lin_x_last = []
    lin_x_base = []
    lin_x_res = []
    lin_x_diff = []
    lin_rms_ke_nl = []
    lin_rms_ke_chain = []
    lin_avg_ke_nl = []
    lin_avg_ke_chain = []
    lin_rms_pe_nl = []
    lin_rms_pe_chain = []
    lin_rms_p_in = []
    lin_rms_x_last = []
    lin_rms_x = []
    
    excit_w = []
    resp_w = []
    
    nu = 1
    w_max = 2000
    for omg in omg_range:
        
        if mass_slope:
            file_name = f'D:/Dados Doutorado/NL_Chain/raw data slope/f_{f}_omg_{omg}_linear_hb.pic'
        else:
            file_name = f'D:/Dados Doutorado/NL_Chain/raw data/f_{f}_omg_{omg}_linear_hb.pic'
        with open(file_name, 'rb') as file:
    #     with open(f'D:/Dados Doutorado/NL_Chain/raw data/f_{f}_omg_{omg}_linear.pic', 'rb') as file:
            ls = load(file)
            t_hb = sys_lin.t(omg)[:, 0] # ls[0]
            x_hb = ls[1]
            dt = t_hb[1]
            N = len(t_hb)
            
        T = 2 * np.pi / omg
        N_per = np.round((dt * N/nu) / T)
        N_int = int(np.round(N_per * T / dt))
            
        F = np.zeros((sys_lin.ndof, len(t_hb)))
        F[0, :] = np.real(f) * np.cos(omg * t_hb) + np.imag(f) * np.sin(omg * t_hb)
    
        p_in = sys_lin.power_in(x_hb, F)
        
    #     ke = sys.kinetic_energy(x_rk4, [a for a in range(20)])
    #     ke_chain = sys.kinetic_energy(x_rk4, [a for a in range(20)])
        ke_chain = sys_lin.kinetic_energy(x_hb, [a for a in range(10, 20)])
        ke_nl = sys_lin.kinetic_energy(x_hb, [a for a in range(20, sys.ndof)])

        ke_chain2 = sys.kinetic_energy(x_hb, [a for a in range(20)], separate_dof=True)
        ke_nl2 = sys.kinetic_energy(x_hb, [a for a in range(20, sys.ndof)], separate_dof=True)
        
    #     pe_nl = sys_lin.dof_nl_potential_energy(x_rk4)
    #     pe_nl -= np.mean(pe_nl)
    #     pe_chain = sys_lin.base_potential_energy(x_rk4, [a for a in range(20)])
    #     pe_chain = pe_chain - np.mean(pe_chain)
        pe_nl = sys_lin.base_potential_energy(x_hb, [a for a in range(20, sys_lin.ndof)])
        pe_nl -= np.mean(pe_nl)
        pe_chain = sys_lin.base_potential_energy(x_hb, [a for a in range(10, 20)])
        pe_chain = pe_chain - np.mean(pe_chain)
        
    #     pe = sys_lin.base_potential_energy(x_rk4, [a for a in range(20, sys_lin.ndof)])
    #     pe -= np.mean(pe)
        
    #     w_max = 1 / (2 * dt) * 2 * np.pi
        dw = 1 / (t_hb[-1]) * 2 * np.pi # Considerando parte da série temporal
        
        w = np.arange(dw, w_max, dw)
        resp_w += list(w)
        excit_w += [omg] * len(w)
            
        lin_rms_ke_nl.append(np.sqrt(np.mean(ke_nl ** 2)) / len(sys_lin.dof_nl))
        lin_rms_ke_chain.append(np.sqrt(np.mean(ke_chain ** 2)) / 10)
        lin_avg_ke_nl.append(np.mean(ke_nl2[:, -N_int:], 1))
        lin_avg_ke_chain.append(np.mean(ke_chain2[:, -N_int:], 1))

        lin_rms_pe_nl.append(np.sqrt(np.mean(pe_nl ** 2)) / len(sys_lin.dof_nl))
        lin_rms_pe_chain.append(np.sqrt(np.mean(pe_chain ** 2)) / 10)
        lin_rms_p_in.append(np.sqrt(np.mean((p_in[0, :]/omg) ** 2)) / sys_lin.ndof)
        lin_rms_x_last.append(np.sqrt(np.mean(x_hb[19, :] ** 2)))
        lin_rms_x.append(np.sqrt(np.mean(x_hb[:sys_lin.ndof, :] ** 2, 1)))
        
        lin_x_last.append(2 / (N) * np.abs(np.fft.fft(x_hb[19, :]))[sys_lin.nu])
        lin_x_base.append(2 / (N) * np.abs(np.fft.fft(x_hb[sys.base_dof[0], :]))[sys_lin.nu])
        lin_x_res.append(2 / (N) * np.abs(np.fft.fft(x_hb[sys.dof_nl[0], :]))[sys_lin.nu])
        lin_x_diff.append(np.mean((np.fft.fft(x_hb[sys.dof_nl, :])/np.fft.fft(x_hb[sys.base_dof, :])), 0)[sys_lin.nu])
    #     print(omg)
    
    lin_rms_ke_chain = np.array(lin_rms_ke_chain)
    lin_rms_ke_nl = np.array(lin_rms_ke_nl)
    lin_avg_ke_chain = np.array(lin_avg_ke_chain)
    lin_avg_ke_nl = np.array(lin_avg_ke_nl)
    lin_rms_pe_nl = np.array(lin_rms_pe_nl)
    lin_rms_pe_chain = np.array(lin_rms_pe_chain)
    lin_rms_p_in = np.array(lin_rms_p_in)
    lin_rms_x_last = np.array(lin_rms_x_last)
    lin_rms_x = np.array(lin_rms_x)
    lin_x_last = np.array(lin_x_last)
    lin_x_base = np.array(lin_x_base)
    lin_x_res = np.array(lin_x_res)
    lin_x_diff = np.array(lin_x_diff)
    lin_x_angle = np.abs(np.angle(lin_x_diff))
    
    lin_x_angle = np.abs(np.angle(lin_x_diff))
    lin_x_angle[-1] = 0
    lin_x_angle[lin_x_angle<0] += 2 * np.pi
    
    
    ####
    # Getting data from nonlinear resonators
    
    omg = 70
    
    omg_range = np.arange(1, 500, 1)
    spec_ke = []
    spec_pe = []
    x_last = []
    x_base = []
    x_res = []
    x_diff = []
    rms_ke_nl = []
    rms_ke_chain = []
    avg_ke_nl = []
    avg_ke_chain = []
    rms_pe_nl = []
    rms_pe_chain = []
    rms_p_in = []
    rms_lag_nl = []
    rms_lag_chain = []
    rms_x_last = []
    rms_x = []
    lag_chain = []
    lag_nl = []
    
    nu = 10
    w_max = 1500
    for omg in omg_range:
        
        
        if mass_slope:
            file_name = f'D:/Dados Doutorado/NL_Chain/raw data slope/f_{f2}_omg_{omg}.pic'
        else:
            file_name = f'D:/Dados Doutorado/NL_Chain/raw data/f_{f2}_omg_{omg}.pic'
        with open(file_name, 'rb') as file:
    #     with open(f'D:/Dados Doutorado/NL_Chain/raw data/f_{f}_omg_{omg}.pic', 'rb') as file:
            ls = load(file)
            t_rk = ls[0]
            x_rk4 = ls[1]
            dt = t_rk[1]
            N = len(t_rk)

        T = 2 * np.pi / omg
        N_per = np.ceil((dt * N / nu) / T)
        N_int = int(np.round(N_per * T / dt))
        
        F = np.zeros((sys.ndof, len(t_rk)))
        F[0, :] = np.real(f2) * np.cos(omg * t_rk) + np.imag(f2) * np.sin(omg * t_rk)
    
        p_in = sys.power_in(x_rk4, F)
        
    #     ke = sys.kinetic_energy(x_rk4, [a for a in range(20)])
    #     ke_chain = sys.kinetic_energy(x_rk4, [a for a in range(20)])
        ke_chain = sys.kinetic_energy(x_rk4, [a for a in range(10, 20)])
        ke_nl = sys.kinetic_energy(x_rk4, [a for a in range(20, sys.ndof)])

        ke_chain2 = sys.kinetic_energy(x_rk4, [a for a in range(20)], separate_dof=True)
        ke_nl2 = sys.kinetic_energy(x_rk4, [a for a in range(20, sys.ndof)], separate_dof=True)
        
        pe_nl = np.sum(sys.dof_nl_potential_energy(x_rk4), 0)
        pe_nl -= np.mean(pe_nl)
        
        pe_chain = sys.base_potential_energy(x_rk4, [a for a in range(10, 20)])
    #     pe_chain = pe_chain - np.mean(pe_chain)
        
    #     pe = sys.dof_nl_potential_energy(x_rk4)
    #     pe -= np.mean(pe)
        
    #     w_max = 1 / (2 * dt) * 2 * np.pi
        dw = 1 / (t_rk[-1]/nu) * 2 * np.pi # Considerando parte da série temporal
        
        w = np.arange(dw, w_max, dw)

        rms_ke_nl.append(np.sqrt(np.mean(ke_nl[-N_int:] ** 2)) / len(sys.dof_nl))
        rms_ke_chain.append(np.sqrt(np.mean(ke_chain[-N_int:] ** 2)) / 10)
        avg_ke_nl.append(np.mean(ke_nl2[:, -N_int:], 1))
        avg_ke_chain.append(np.mean(ke_chain2[:, -N_int:], 1))

        rms_pe_nl.append(np.sqrt(np.mean(pe_nl[-N_int:] ** 2)) / len(sys.dof_nl))
        rms_pe_chain.append(np.sqrt(np.mean(pe_chain[-N_int:] ** 2)) / 10)
        rms_p_in.append(np.sqrt(np.mean((p_in[0, -N_int:]/omg) ** 2)) / sys.ndof)
        rms_x_last.append(np.sqrt(np.mean(x_rk4[19, -N_int:] ** 2)))
        rms_x.append(np.sqrt(np.mean((x_rk4[:sys.ndof, -N_int:] - np.mean(x_rk4[:sys.ndof, -N_int:], 1).reshape((sys.ndof, 1))) ** 2, 1)))

        x_last.append(2 / (N_int) * np.abs(np.fft.fft(x_rk4[19, -N_int:]))[1:len(w)+1])
        x_base.append(2 / (N_int) * np.abs(np.fft.fft(x_rk4[sys.base_dof[0], -N_int:]))[1:len(w)+1])
        x_res.append(2 / (N_int) * np.abs(np.fft.fft(x_rk4[sys.dof_nl[0], -N_int:]))[1:len(w)+1])
        x_diff.append(np.mean((np.fft.fft(x_rk4[sys.dof_nl, -N_int:])/np.fft.fft(x_rk4[sys.base_dof, -N_int:])), 0)[1:len(w)+1])
        print(omg)
        
    spec_ke = np.array(spec_ke)
    spec_pe = np.array(spec_pe)
    rms_ke_chain = np.array(rms_ke_chain)
    rms_ke_nl = np.array(rms_ke_nl)
    avg_ke_chain = np.array(avg_ke_chain)
    avg_ke_nl = np.array(avg_ke_nl)
    rms_pe_chain = np.array(rms_pe_chain)
    rms_pe_nl = np.array(rms_pe_nl)
    rms_lag_nl = np.array(rms_lag_nl)
    rms_lag_chain = np.array(rms_lag_chain)
    rms_p_in = np.array(rms_p_in)
    rms_x_last = np.array(rms_x_last)
    rms_x = np.array(rms_x)
    x_last = np.array(x_last)
    x_base = np.array(x_base)
    x_res = np.array(x_res)
    x_diff = np.array(x_diff)
    lag_chain = np.array(lag_chain)
    lag_nl = np.array(lag_nl)
    
    x_angle = np.abs(np.angle(x_diff))
    x_angle[-1,-1] = 0
    x_angle[x_angle < 0] += 2 * np.pi
    
    ######
    # Plots
    
    fig_rms = go.Figure(data=[go.Scatter(x=omg_range, y=(rms_x_last), mode='lines',
                                         name='Nonlinear resonators'),
                              go.Scatter(x=omg_range, y=(f2/1000*lin_rms_x_last), mode='lines',
                                         name='Linear resonators'),
                              go.Scatter(x=omg_range, y=(f2/1000*base_rms_x_last), mode='lines',
                                         name='Bare chain'),
                             ])
    # fig_rms.update_layout(yaxis=dict(range=[-15,-4]))
    fig_rms.update_yaxes(type='log', title='Displacement [m rms]', range=[-7, -1.5])
    fig_rms.update_xaxes(title='Excitation Frequency [rad/s]')
    fig_rms.update_layout(width=700,height=500)
    fig_rms.write_image(f'graficos artigo/FRF {f2} - slope-{mass_slope}.pdf')
    fig_rms.write_image(f'graficos artigo/FRF {f2} - slope-{mass_slope}.png', scale=10)
    
    
    fig_rms = go.Figure(data=[go.Scatter(x=omg_range, y=(rms_ke_nl), mode='lines',
                                         name='RMS KE resonators'),
                              go.Scatter(x=omg_range, y=(rms_ke_chain), mode='lines',
                                         name='RMS KE chain'),
    #                          go.Scatter(x=omg_range, y=(rms_pe_nl), mode='lines',
    #                                      name='RMS PE resonators'),
    #                         go.Scatter(x=omg_range, y=(rms_pe_chain), mode='lines',
    #                                      name='RMS PE chain'),
                             go.Scatter(x=omg_range, y=(rms_p_in), mode='lines',
                                          name='RMS Energy In'),
                             go.Scatter(x=omg_range, y=(f2**2/1e6*base_rms_ke_nl), mode='lines',
                                         name='Base RMS KE resonators'),
                              go.Scatter(x=omg_range, y=(f2**2/1e6*base_rms_ke_chain), mode='lines',
                                         name='Base RMS KE chain'),
    #                         go.Scatter(x=omg_range, y=(f2**2/1e6*base_rms_pe_nl), mode='lines',
    #                                      name='Base RMS PE resonators'),
    #                          go.Scatter(x=omg_range, y=(f2**2/1e6*base_rms_pe_chain), mode='lines',
    #                                      name='Base RMS PE chain'),
                             go.Scatter(x=omg_range, y=(f2**2/1e6*base_rms_p_in), mode='lines',
                                          name='Base RMS Energy In'),
                             go.Scatter(x=omg_range, y=(f2**2/1e6*lin_rms_ke_nl), mode='lines',
                                         name='Lin RMS KE resonators'),
                              go.Scatter(x=omg_range, y=(f2**2/1e6*lin_rms_ke_chain), mode='lines',
                                         name='Lin RMS KE chain'),
    #                         go.Scatter(x=omg_range, y=(f2**2/1e6*lin_rms_pe_nl), mode='lines',
    #                                      name='Lin RMS PE resonators'),
    #                          go.Scatter(x=omg_range, y=(f2**2/1e6*lin_rms_pe_chain), mode='lines',
    #                                      name='Lin RMS PE chain'),
                             go.Scatter(x=omg_range, y=(f2**2/1e6*lin_rms_p_in), mode='lines',
                                          name='Lin RMS Energy In')])
    # fig_rms.update_layout(yaxis=dict(range=[-6,8]))
    fig_rms.update_yaxes(type='log', title='Energy (J rms)', range=[-6, 2])
    fig_rms.update_xaxes(title='Excitation Frequency [rad/s]')
    
    fig_rms.write_html(f'graficos artigo/Energias {f2} - slope-{mass_slope}.html')

    fig_rms = go.Figure(data=[go.Scatter(x=omg_range, y=(rms_p_in / rms_ke_nl), mode='lines',
                                         name='Nonlinear Resonators Impedance'),
                              go.Scatter(x=omg_range, y=(lin_rms_p_in / lin_rms_ke_nl), mode='lines',
                                         name='Linear Resonators Impedance'),
                              ])

    fig_rms.update_yaxes(type='log', title='Nondimensional Impedance', range=[-2, 4])
    fig_rms.update_xaxes(title='Excitation Frequency [rad/s]')
    fig_rms.write_html(f'graficos artigo/Resonator Impedance  {f2} - slope-{mass_slope}.html')
    fig_rms.write_image(f'graficos artigo/Resonator Impedance  {f2} - slope-{mass_slope}.pdf')
    fig_rms.write_image(f'graficos artigo/Resonator Impedance  {f2} - slope-{mass_slope}.png', scale=10)
    
    
    fig_rms = go.Figure(data=[go.Scatter(x=omg_range, y=(rms_p_in / rms_ke_chain), mode='lines',
                                         name='Nonlinear Chain Impedance'),
                              go.Scatter(x=omg_range, y=(lin_rms_p_in / lin_rms_ke_chain), mode='lines',
                                         name='Linear Chain Impedance'),
                              ])
    # fig_rms.update_layout(yaxis=dict(range=[-6,8]))
    fig_rms.update_yaxes(type='log', title='Nondimensional Impedance', range=[-2, 4])
    fig_rms.update_xaxes(title='Excitation Frequency [rad/s]')
    fig_rms.write_html(f'graficos artigo/Chain Impedance  {f2} - slope-{mass_slope}.html')
    fig_rms.write_image(f'graficos artigo/Chain Impedance  {f2} - slope-{mass_slope}.pdf')
    fig_rms.write_image(f'graficos artigo/Chain Impedance  {f2} - slope-{mass_slope}.png', scale=10)
    
    
    
    fig_x_last = go.Figure(data=[go.Heatmap(y=w, x=omg_range, z=np.log10(x_last.transpose()),
                                            colorbar=dict(title='Resp. amplitude [m]',
                                                          tickvals=np.arange(-20, 0),
                                                          ticktext=[f'1e{a}' for a in np.arange(-20, 0)]),
                                            zmin=-8,
                                            zmax=np.max(np.log10(x_last))),
                                 go.Scatter(x=[0,500], y=[0,500], mode='lines',name='Synchronous line',
                                            line=dict(dash='dot',color='black',
                                                      width=0.5),
                                           showlegend=True)
                          ])
    fig_x_last.update_layout(xaxis=dict(title='Excitation Frequency (rad/s)',
                                        range=[0,500]),
                             yaxis=dict(title='Response frequency (rad/s)',
                                       range=[0,1500]),
                            legend=dict(orientation='h',
                                        xanchor='center',
                                        x=0.5,
                                       yanchor='bottom',
                                       y=1.05),
                            width=900,
                            height=500)
    
    fig_x_last.write_image(f'graficos artigo/heatmap last {f2} - slope-{mass_slope}.pdf')
    fig_x_last.write_html(f'graficos artigo/heatmap last {f2} - slope-{mass_slope}.html')
    fig_x_last.write_image(f'graficos artigo/heatmap last {f2} - slope-{mass_slope}.png', scale=10)

    omg_min = 40
    i_omg = int(omg_min / (omg_range[1] - omg_range[0]))
    ke_max = (np.log10(np.max(avg_ke_chain[i_omg:, :])))
    ke_min = np.ceil(np.log10(np.min(avg_ke_chain[i_omg:, :])))

    fig_x_last = make_subplots(rows=2, cols=1,
                               row_heights=[len(sys.dof_nl) / sys.ndof,
                                            1 - len(sys.dof_nl) / sys.ndof])


    fig_x_last.add_trace(go.Heatmap(y=1 + np.arange(len(sys.dof_nl)),
                                    x=omg_range[i_omg:],
                                    z=np.log10(avg_ke_nl.transpose()[:, i_omg:]),
                                    colorbar=dict(title='Avg. Kinetic Energy (J)',
                                                  tickvals=np.arange(np.ceil(ke_min), ke_max + 1),
                                                  ticktext=[f'1e{a:.0f}' for a in np.arange(ke_min, ke_max + 1)]
                                                  ),
                                    zmin=ke_min,
                                    zmax=ke_max
                                    ), row=1, col=1)
    fig_x_last.add_trace(go.Heatmap(y=1 + np.arange(sys.ndof - len(sys.dof_nl)),
                                    x=omg_range[i_omg:],
                                    z=np.log10(avg_ke_chain.transpose()[:, i_omg:]),
                                    colorbar=dict(title='Avg. Kinetic Energy (J)',
                                                  tickvals=np.arange(np.ceil(ke_min), ke_max + 1),
                                                  ticktext=[f'1e{a:.0f}' for a in np.arange(ke_min, ke_max + 1)]
                                                  ),
                                    zmin=ke_min,
                                    zmax=ke_max
                                    ), row=2, col=1)
    fig_x_last.update_xaxes(title_text='', tickvals=[], row=1, col=1)
    fig_x_last.update_xaxes(title_text='Excitation Frequency [rad/s]', row=2, col=1)
    fig_x_last.update_yaxes(title_text='Resonators', row=1, col=1)
    fig_x_last.update_yaxes(title_text='Chain', row=2, col=1)

    fig_x_last.update_layout(width=900,
                             height=500)
    
    fig_x_last.write_image(f'graficos artigo/heatmap position {f2} - slope-{mass_slope}.pdf')
    fig_x_last.write_image(f'graficos artigo/heatmap position {f2} - slope-{mass_slope}.png', scale=10)

    ke_max = (np.log10(np.max(lin_avg_ke_chain[i_omg:, :])))
    ke_min = np.ceil(np.log10(np.min(lin_avg_ke_chain[i_omg:, :])))

    fig_x_last = make_subplots(rows=2, cols=1,
                               row_heights=[len(sys.dof_nl) / sys.ndof,
                                            1 - len(sys.dof_nl) / sys.ndof])

    fig_x_last.add_trace(go.Heatmap(y=1 + np.arange(len(sys.dof_nl)),
                                    x=omg_range[i_omg:],
                                    z=np.log10(lin_avg_ke_nl.transpose()[:, i_omg:]),
                                    colorbar=dict(title='Avg. Kinetic Energy (J)',
                                                  tickvals=np.arange(np.ceil(ke_min), ke_max + 1),
                                                  ticktext=[f'1e{a:.0f}' for a in np.arange(ke_min, ke_max + 1)]
                                                  ),
                                    zmin=ke_min,
                                    zmax=ke_max
                                    ), row=1, col=1)
    fig_x_last.add_trace(go.Heatmap(y=1 + np.arange(sys.ndof - len(sys.dof_nl)),
                                    x=omg_range[i_omg:],
                                    z=np.log10(lin_avg_ke_chain.transpose()[:, i_omg:]),
                                    colorbar=dict(title='Avg. Kinetic Energy (J)',
                                                  tickvals=np.arange(np.ceil(ke_min), ke_max + 1),
                                                  ticktext=[f'1e{a:.0f}' for a in np.arange(ke_min, ke_max + 1)]
                                                  ),
                                    zmin=ke_min,
                                    zmax=ke_max
                                    ), row=2, col=1)
    fig_x_last.update_xaxes(title_text='', tickvals=[], row=1, col=1)
    fig_x_last.update_xaxes(title_text='Excitation Frequency [rad/s]', row=2, col=1)
    fig_x_last.update_yaxes(title_text='Resonators', row=1, col=1)
    fig_x_last.update_yaxes(title_text='Chain', row=2, col=1)

    fig_x_last.update_layout(width=900,
                             height=500)

    
    fig_x_last.write_image(f'graficos artigo/heatmap position linear - slope-{mass_slope}.pdf')
    fig_x_last.write_image(f'graficos artigo/heatmap position linear - slope-{mass_slope}.png', scale=10)
    
    
    
    fig_x_diff_amp = go.Figure(data=[go.Heatmap(y=w, x=omg_range, z=np.log10(np.abs(x_diff).transpose()),
                                                colorbar=dict(title='Amplification',
                                                              tickvals=np.arange(-2, 2),
                                                              ticktext=[f'1e{a}' for a in np.arange(-2, 2)]),
                                                zmin=-1,#np.min(np.log10(np.abs(x_diff))),
                                                zmax=1,),#np.max(np.log10(np.abs(x_diff)))),
                                     go.Scatter(x=[0,500], y=[0,500], mode='lines',name='Synchronous line',
                                            line=dict(dash='dot',color='black',
                                                      width=0.5),
                                           showlegend=True)
                          ])
    fig_x_diff_amp.update_layout(xaxis=dict(title='Excitation Frequency (rad/s)'),
                             yaxis=dict(title='Response frequency (rad/s)',
                                       range=[0,1500]),
                            legend=dict(orientation='h',
                                        xanchor='center',
                                        x=0.5,
                                       yanchor='bottom',
                                       y=1.05),
                            width=900,
                            height=500)
    
    
    fig_x_diff_amp.write_image(f'graficos artigo/heatmap amplification {f2} - slope-{mass_slope}.pdf')
    fig_x_diff_amp.write_image(f'graficos artigo/heatmap amplification {f2} - slope-{mass_slope}.png', scale=10)
    
    
    
    fig_x_diff_phase = go.Figure(data=[go.Heatmap(y=w, x=omg_range, z=x_angle.transpose(),
                                                  colorbar=dict(title='Phase [deg]',
                                                                tickvals=[0, np.pi / 2, np.pi],
                                                                ticktext=['0', '90', '180']),
                                                  zmin=0,  # np.min(np.log10(np.abs(x_diff))),
                                                  zmax=np.pi, 
                                                  colorscale='Phase'),
                                       go.Scatter(x=[0,500], y=[0,500], mode='lines',name='Synchronous line',
                                            line=dict(dash='dot',color='black',
                                                      width=0.5),
                                           showlegend=True)
                          ])
    fig_x_diff_phase.update_layout(xaxis=dict(title='Excitation Frequency (rad/s)'),
                             yaxis=dict(title='Response frequency (rad/s)',
                                       range=[0,1500]),
                            legend=dict(orientation='h',
                                        xanchor='center',
                                        x=0.5,
                                       yanchor='bottom',
                                       y=1.05),
                            width=900,
                            height=500)
                                 
    fig_x_diff_phase.write_image(f'graficos artigo/heatmap phase {f2} - slope-{mass_slope}.pdf')
    fig_x_diff_phase.write_image(f'graficos artigo/heatmap phase {f2} - slope-{mass_slope}.png', scale=10)
    
        
    
f_ = [1000., 3000., 6000., 9000.]
slope_ = [False, True]

for f2 in f_:
    for slope in slope_:
        plots(f2, slope)

