
import numpy as np
from harmbal import Sys_NL
from harmbal import poincare_section as pcs
import plotly.graph_objects as go
import time
from pickle import dump, load


m0 = 1
m1 = 0.5

w0 = 10

x_eq = 0.1
w1 = 10

k0 = w0**2 * m0
beta = -1/2 * w1**2 * m1
alpha = -beta / x_eq**2

M = np.array([[m0 , 0],
              [0 , m1]])

K_lin = np.array([[k0 , 0],
                  [0 , 0]])

Snl = np.array(np.array([[1 , -1],
                         [-1 , 1]]))

K = np.array([[k0 + beta , -beta],
              [-beta , beta]])

cp = 5e-3
C = cp * K

n_harm = 20
nu = 2
N = 2 # sinal no tempo terá comprimento 2*N*n_harm

S = Sys_NL(M=M,K=K_lin,Snl=Snl,beta=beta,alpha=alpha,
           n_harm=n_harm,nu=nu,N=N,cp=cp,C=K_lin*cp)
# S.base_dof = [0]
# S.dof_nl = [1]

omg_range = np.arange(1, 20, 0.05)

linear = False

# rms_rk = 1e-5 * np.ones((2*S.ndof, len(omg_range)))

f_range = np.arange(0.01, 1, 0.05)

for f0 in f_range:

    if linear:
        S = S.eq_linear_Stem()

    f = {0: f0}
    kin_en = np.zeros((2, len(omg_range)))
    en_flow = np.zeros((5, len(omg_range)))
    en_indiv = np.zeros((2, len(omg_range)))
    pow_in = np.zeros((1, len(omg_range)))

    for i, omg in enumerate(omg_range):
        x_hb, res = S.solve_hb(f=f, omg=omg, full_output=True, state_space=True)

        F = np.zeros((S.ndof, len(S.t(omg))))
        F[0, :] = np.real(f[0]) * np.cos(omg * S.t(omg).reshape(len(S.t(omg)))) + np.imag(f[0]) * np.sin(omg * S.t(omg).reshape(len(S.t(omg))))

        forces = S.dof_nl_forces(x_hb)
        v = x_hb[S.ndof:, :]
        en_flow_damp = (-forces[0] * v[S.base_dof, :])
        en_flow_el = (-forces[1] * v[S.base_dof, :])

        p_in = S.power_in(x_hb, F)
        pow_in[0, i] = np.mean(p_in[0, :])

        ke = S.kinetic_energy(x_hb)
        kin_en[0, i] = np.mean(ke[0, :])
        kin_en[1, i] = np.mean(ke[1:, :])

        en_flow_res = S.dof_nl_energy_flow(x_hb)
        en_flow[0, i] = np.mean(en_flow_res)
        en_flow_base = S.base_structure_energy_flow(x_hb)
        en_flow[1, i] = np.mean(en_flow_base)
        en_flow[2, i] = np.mean(en_flow_damp)
        en_flow[3, i] = np.mean(en_flow_el)
        en_flow[4, i] = np.mean(p_in[0, :])

    fig = go.Figure(data=[go.Scatter(x=omg_range, y=np.log10(kin_en[0, :]), name='Kin Energy RMS - Base'),
                          go.Scatter(x=omg_range, y=np.log10(kin_en[1, :]), name='Kin Energy RMS - Resonators'), ])
    # fig.update_layout(yaxis=dict(type='log'))
    fig.write_html(f'2GDL_duffin/sweep_kin_en_f{f[0]}_lin-{linear}.html')

    fig = go.Figure(data=[go.Scatter(x=omg_range, y=en_flow[0, :], name='En. flow to resonators'),
                          go.Scatter(x=omg_range, y=en_flow[1, :], name='En. flow to base'),
                          go.Scatter(x=omg_range, y=en_flow[2, :], name='En. flow base damping',
                                     line=(dict(dash='dash'))),
                          go.Scatter(x=omg_range, y=en_flow[3, :], name='En. flow base elastic',
                                     line=(dict(dash='dash'))),
                          go.Scatter(x=omg_range, y=en_flow[4, :], name='En. flow in'),
                          ])
    # fig.update_layout(yaxis=dict(type='log'))
    fig.write_html(
        f'2GDL_duffin/sweep_en_flow_f{f[0]}_lin-{linear}.html')
                                                                                  