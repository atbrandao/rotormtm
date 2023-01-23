import harmbal as hb
import numpy as np
from pickle import load

N = 50
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

#M = np.array([[1, 0], [0, 2]])
#K = np.array([[3, 0], [0, 0]])
#Snl = np.array([[1, -1], [-1, 1]])

#beta = 1**2*1/2
#alpha = beta/x0**2

sys = hb.Sys_NL(M=M, K=K, Snl=Snl, beta=-beta, alpha=alpha, n_harm=10, nu=1)


sys.export_sys_data(p1={0:9+1.j, 1:8+2.j}, p2=64123782192678.2222)

import datetime
f = {0: 1e3}
omg = 100
t0 = datetime.datetime.now()
x_hb, res = sys.solve_hb(f=f, omg=omg, full_output=True, state_space=True)
print(datetime.datetime.now() - t0)
z = res[0]
# fm = sys.floquet_multipliers(omg, z, dt_refine=250)
# print(np.max(np.abs(fm)) > 1)



t0 = datetime.datetime.now()

dt = 0.5 * np.pi / np.max(np.imag(np.linalg.eig(sys.A_lin)[0]))
tf = 30
t_rk = np.arange(0, tf, dt)
x0 = x_hb[:, 0].reshape((sys.ndof * 2, 1))
t0 = datetime.datetime.now()
x_rk4 = sys.solve_transient(f=f, omg=omg, t=t_rk, x0=x0, run_fortran=True, keep_data=True)
print(datetime.datetime.now() - t0)
t0 = datetime.datetime.now()
x_rk42 = sys.solve_transient(f=f, omg=omg, t=t_rk, x0=x0)
print(datetime.datetime.now() - t0)

print()