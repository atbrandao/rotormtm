from rotor_mtm import harmbal as hb
import numpy as np
from pickle import load, dump
import plotly.graph_objects as go
# from plotly_resampler import FigureResampler, FigureWidgetResampler
from plotly.subplots import make_subplots
from rotor_mtm.results import IntegrationResults

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

m_slope = True
k_slope = False
slope = np.linspace(0.7,1.3,N_res)

beta = w_res**2*m_res/2
alpha = beta/x0**2
delta = 2 * cp * 2 * beta

if m_slope:
    M_res = list(slope * (N_res * [m_res])) 
    
else:
    M_res = list(N_res * [m_res])

M = np.array(N*[m] + M_res) * np.eye(N + N_res)
    
M_base = M[:N, :N] @ np.eye(N)
M_base[n_start : n_start + N_res, 
        n_start : n_start + N_res] += np.array(M_res) * np.eye(N_res)

K = np.array(N*[2*k]+N_res*[0]) * np.eye(N+N_res)

Snl = 0 * K

for i in range(N - 1):
    K[i, i + 1] = -k
    K[i + 1, i] = -k

K_base = K[:N, :N] @ np.eye(N)

for i in range(N_res):
    dof = [n_start + i, N + i]
    k_el = np.array([[1, -1], [-1, 1]])
    # K[np.ix_(dof, dof)] += k_el * 2 * beta
    Snl[np.ix_(dof, dof)] += k_el

C = cp * K + delta * Snl
C_base = cp * K_base

sys = hb.Sys_NL(M=M, K=K, Snl=Snl, beta=-beta, alpha=alpha, n_harm=10, nu=1, C=C)
sys_lin = sys.eq_linear_system()
sys_base = hb.Sys_NL(M=M_base, K=K_base, Snl=np.zeros((N, N)), beta=0, alpha=0, n_harm=10, nu=1, C=C_base)




omg_range = np.arange(1, 500, 1)

slope = m_slope

sl_str = ''
if slope:    
    sl_str = ' slope'

file_base = 'C:/Users/HR7O/OneDrive - PETROBRAS/Documents/Doutorado/Python/Rotor/Rotor simples/others/Dados Backup/'

for f2 in [1000.,
           2000.,
           3000.,
           4000.,
           5000.,
           6000.,
           7000.,
           8000.,
           9000.]:
    

    data_dict_list = []
    for i, omg in enumerate(omg_range):

        with open(f'{file_base}raw data{sl_str}/f_{f2}_omg_{omg}.pic', 'rb') as file:
            ls = load(file)
            t_rk = ls[0]
            x_rk4 = ls[1]            
        
        data_dict_list.append(dict(time=t_rk))
        for p in range(len(x_rk4[:, 0])):
            data_dict_list[-1][p] = x_rk4[p, :]
        
    res = IntegrationResults(data_dict_list=data_dict_list,
                            frequency_list=omg_range,
                            system=sys)
    res.linear_system = sys_lin
    res.rigid_system = sys_base
    
    filename = f'{file_base}results_NL_Chain{sl_str}_f_{f2}.pic'
    print(filename)

    with open(filename, 'wb') as file:
        dump(res, file)
        print('Saved!')

    
