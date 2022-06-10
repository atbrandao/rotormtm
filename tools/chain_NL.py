import harmbal as hb
import numpy as np
from pickle import load

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

m_slope = False
k_slope = False
slope = np.linspace(0.7,1.3,N_res)

beta = w_res**2*m_res/2
alpha = beta/x0**2
delta = 2 * cp * 2 * beta

M = np.array(N*[m]+N_res*[m_res]) * np.eye(N+N_res)
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
# x_rk4 = sys.solve_transient(f=f, omg=omg, t=t_rk, x0=x0)#np.zeros((sys.ndof * 2, 1)))

try:
    with open(f'rms_rk_f-{f}.pic'.replace(':', '_'), 'rb') as file:
        rms_rk = load(file)
        save_rms = None
except:
    rms_rk = None
    save_rms = f'rms_rk_f-{f}.pic'.replace(':', '_')

omg_range = np.arange(1, 500, 1)

# rms_rk = 1e-5 * np.ones((2*sys.ndof, len(omg_range)))

fig = sys.plot_frf(omg_range=omg_range, tf=tf, dt_base=dt, stability_analysis=True, dt_refine=None,
                   f=f, probe_dof=[N-1], continuation='hb', save_rms_rk=save_rms, rms_rk=rms_rk)
fig[0].write_html(f'NL_Chain/frf_omg-{omg}_f-{f}.html'.replace(':', '_'))
fig[1].write_html(f'NL_Chain/frf_cost_omg-{omg}_f-{f}.html'.replace(':', '_'))
