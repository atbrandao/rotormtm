from rotor_mtm import harmbal as hb
import numpy as np
from pickle import load, dump
import plotly.graph_objects as go

N = 20
mass = 100
L = 1

d = L/N
k = 8e4/d
cp = 0.0002
m = mass/N

m_res = m / 2

x0 = d/2
w_res = 250
N_res = 10 # number of resonators
n_start = 5

m_slope = True
k_slope = False
slope = np.linspace(0.7,1.3,N_res)
#slope = np.linspace(1.3,0.7,N_res)

beta = w_res**2*m_res/2
alpha = beta/x0**2
delta = 2 * cp * 2 * beta

if m_slope:
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

f = {0: 1e3}
omg = 100

x_hb, res = sys.solve_hb(f=f, omg=omg, full_output=True, state_space=True)
z = res[0]
fm = sys.floquet_multipliers(omg, z, dt_refine=250)
print(np.max(np.abs(fm)) > 1)

dt = 0.5 * np.pi / np.max(np.imag(np.linalg.eig(sys.A_lin)[0]))
tf = 30
t_rk = np.arange(0, tf, dt)
x0 = x_hb[:, 0].reshape((sys.ndof*2, 1))
x_rk4 = sys.solve_transient(f=f, omg=omg, t=t_rk, x0=x0)#np.zeros((sys.ndof * 2, 1)))

forces=sys.dof_nl_forces(x_rk4)
kin_en=sys.kinetic_energy(x_rk4)
en_flow=sys.dof_nl_energy_flow(x_rk4)
#
# go.Figure(data=[go.Scatter(x=t_rk,y=en_flow[0,:])]).write_html(f'NL_Chain/teste_en_flow_f{f[0]}_omg{omg}.html')
# go.Figure(data=[go.Scatter(x=t_rk,y=forces[0][0,:], name='damping force'),
#                 go.Scatter(x=t_rk,y=forces[1][0,:], name='elastic force'),
#                 go.Scatter(x=t_rk,y=x_rk4[sys.ndof+20,:], name='velocidade'),]).write_html(f'NL_Chain/teste_forces_f{f[0]}_omg{omg}.html')
# go.Figure(data=[go.Scatter(x=t_rk,y=kin_en[19,:], name='main structure'),
#                 go.Scatter(x=t_rk,y=kin_en[20,:], name='first resonator')]).write_html(f'NL_Chain/teste_kin_en_f{f[0]}_omg{omg}.html')

omg_range = np.arange(1, 500, 1)

linear = True

# rms_rk = 1e-5 * np.ones((2*sys.ndof, len(omg_range)))

f_range = [1] # np.arange(9) + 1

for f0 in f_range:

    if linear:
        sys = sys.eq_linear_system()

    f = {0: f0 * 1e3}
    kin_en = np.zeros((2, len(omg_range)))
    en_flow = np.zeros((5, len(omg_range)))
    en_indiv = np.zeros((2, len(omg_range)))
    pow_in = np.zeros((1, len(omg_range)))

    for i, omg in enumerate(omg_range):
        x_hb, res = sys.solve_hb(f=f, omg=omg, full_output=True, state_space=True)
        x0 = x_hb[:, 0].reshape((sys.ndof * 2, 1))
        print(f)
        print(omg)


        # x_rk4 = sys.solve_transient(f=f, omg=omg, t=t_rk, x0=x0)  # np.zeros((sys.ndof * 2, 1)))
        # x_hb = sys.solve_hb(f=f, omg=omg)  # np.zeros((sys.ndof * 2, 1)))

        # with open(f'D:/Dados Doutorado/NL_Chain/raw data slope/f_{f[0]}_omg_{omg}_linear.pic', 'wb') as file:
        #     dump([t_rk, x_rk4], file)

        with open(f'D:/Dados Doutorado/NL_Chain/raw data slope/f_{f[0]}_omg_{omg}_linear_hb.pic', 'wb') as file:
            dump([sys.t(omg), x_hb], file)

        F = np.zeros((sys.ndof, len(t_rk)))
        F[0, :] = np.real(f[0]) * np.cos(omg * t_rk) + np.imag(f[0]) * np.sin(omg * t_rk)

        forces = sys.dof_nl_forces(x_rk4)
        v = x_rk4[sys.ndof:, :]
        en_flow_damp = (-forces[0] * v[sys.base_dof, :])
        en_flow_el = (-forces[1] * v[sys.base_dof, :])


        p_in = sys.power_in(x_rk4, F)
        pow_in[0, i] = np.mean(p_in[0,:])

        # ke = sys.kinetic_energy(x_rk4)
        # kin_en[0, i] = np.mean(np.sum(ke[:20, len(t_rk)//2:], 0))
        # kin_en[1, i] = np.mean(np.sum(ke[20:, len(t_rk)//2:], 0))

    #     en_flow_res = sys.dof_nl_energy_flow(x_rk4)
    #     en_flow[0, i] = np.mean(np.sum(en_flow_res[:, len(t_rk)//2:], 0))
    #     en_flow_base = sys.base_structure_energy_flow(x_rk4)
    #     en_flow[1, i] = np.mean(np.sum(en_flow_base[:, len(t_rk)//2:], 0))
    #     en_flow[2, i] = np.mean(np.sum(en_flow_damp[:, len(t_rk) // 2:], 0))
    #     en_flow[3, i] = np.mean(np.sum(en_flow_el[:, len(t_rk) // 2:], 0))
    #     en_flow[4, i] = np.mean(p_in[0, len(t_rk) // 2:])
    #
    # fig = go.Figure(data=[go.Scatter(x=omg_range, y=kin_en[0,:], name='Kin Energy RMS - Base'),
    #                 go.Scatter(x=omg_range, y=kin_en[1, :], name='Kin Energy RMS - Resonators'), ])
    # # fig.update_layout(yaxis=dict(type='log'))
    # fig.write_html(f'NL_Chain/sweep_kin_en_f{f[0]}_lin-{linear}.html')
    #
    # fig = go.Figure(data=[go.Scatter(x=omg_range, y=en_flow[0, :], name='En. flow to resonators'),
    #                       go.Scatter(x=omg_range, y=en_flow[1, :], name='En. flow to base'),
    #                       go.Scatter(x=omg_range, y=en_flow[2, :], name='En. flow base damping',
    #                                    line=(dict(dash='dash'))),
    #                       go.Scatter(x=omg_range, y=en_flow[3, :], name='En. flow base elastic',
    #                                    line=(dict(dash='dash'))),
    #                       go.Scatter(x=omg_range, y=en_flow[4, :], name='En. flow in'),
    #                       ])
    # # fig.update_layout(yaxis=dict(type='log'))
    # fig.write_html(
    #     f'NL_Chain/sweep_en_flow_f{f[0]}_lin-{linear}.html')


print('Finish!')

if True:

    try:
        with open(f'NL_Chain/data/rms_rk_f-{f}_slope-{m_slope}.pic'.replace(':', '_'), 'rb') as file:
            rms_rk = load(file)
            save_rms = None
    except:
        rms_rk = None
        save_rms = f'NL_Chain/data/rms_rk_f-{f}_slope-{m_slope}_slope-{m_slope}.pic'.replace(':', '_')
    save_hb = f'NL_Chain/data/rms_hb_f-{f}_slope-{m_slope}.pic'.replace(':', '_')

    fig = sys.plot_frf(omg_range=omg_range, tf=tf, dt_base=dt, stability_analysis=True, dt_refine=None,
                       f=f, probe_dof=[N-1], continuation='hb', save_rms_rk=save_rms, save_rms_hb=save_hb,
                       rms_rk=rms_rk)
    fig[0].write_html(f'NL_Chain/frf_omg-{omg}_f-{f}_slope-{m_slope}.html'.replace(':', '_'))
    fig[1].write_html(f'NL_Chain/frf_cost_omg-{omg}_f-{f}_slope-{m_slope}.html'.replace(':', '_'))
    fig[0].write_image(f'NL_Chain/frf_omg-{omg}_f-{f}_slope-{m_slope}.pdf'.replace(':', '_'))
    fig[1].write_image(f'NL_Chain/frf_cost_omg-{omg}_f-{f}_slope-{m_slope}.pdf'.replace(':', '_'))
    fig[0].write_image(f'NL_Chain/frf_omg-{omg}_f-{f}_slope-{m_slope}.png'.replace(':', '_'), scale=8)
    fig[1].write_image(f'NL_Chain/frf_cost_omg-{omg}_f-{f}_slope-{m_slope}.png'.replace(':', '_'), scale=8)

