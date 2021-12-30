# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 10:44:02 2021

@author: HR7O
"""

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
nu = 1
N = 2 # sinal no tempo terá comprimento N*n_harm

S = Sys_NL(M=M,K=K_lin,Snl=Snl,beta=beta,alpha=alpha,n_harm=n_harm,nu=nu,N=N,cp=cp)

omg = 11

f = {0: 1}
try:
    with open(f'data_rk f {f}.pic'.replace(':','_'),'rb') as file:
        aux = load(file)
        rms_rk = aux[0]
        pc = aux[1]
    calc = False
except:
    calc = True
    
tf = 500
dt = 0.01
t_rk = np.arange(0,tf,dt)

omg_range = np.arange(0.5,20,.05)
rms_hb = np.zeros((3,len(omg_range)))
if calc:
    rms_rk = np.zeros((2,len(omg_range)))
    pc = []
cost_hb = []
n_points = 50
z0 = None

for i, omg in enumerate(omg_range):
    t0 = time.time()
    if calc:        
        x_rk = S.solve_transient(f,t_rk,omg,np.array([[0],[x_eq],[0],[0]]))
        print(f'RK4 took {(time.time()-t0):.1f} seconds to run.')
    t1 = time.time()
    x_hb, res = S.solve_hb(f,omg,z0=z0,full_output=True,method=None)#'ls')
    
    try:
        cost_hb.append(res.cost)
        z0 = res.x
    except:
        cost_hb.append(np.linalg.norm(res[1]['fvec']))
        z0 = res[0]
    print(f'Harmbal took {(time.time()-t1):.1f} seconds to run: {(t1-t0)/(time.time()-t1):.1f} times faster.')
    
    if calc:
        rms_rk[0,i] = np.sqrt(np.sum((x_rk[0,int((tf/2)/dt):] - np.mean(x_rk[0,int((tf/2)/dt):])) ** 2) / (int((tf/2)/dt)))
        rms_rk[1,i] = np.sqrt(np.sum((x_rk[1,int((tf/2)/dt):] - np.mean(x_rk[1,int((tf/2)/dt):])) ** 2) / (int((tf/2)/dt)))
        pc.append(pcs(x_rk, t_rk, omg, n_points))
    
    rms_hb[0,i] = np.sqrt(np.sum((x_hb[0,:] - np.mean(x_hb[0,:])) ** 2) / (len(S.t(omg))-1))
    rms_hb[1,i] = np.sqrt(np.sum((x_hb[1,:] - np.mean(x_hb[1,:])) ** 2) / (len(S.t(omg))-1))
    try:
        if res[-2] != 1:
            print(res[-1])
            rms_hb[2,i] = 1
    except:
        if not res.success:
            print(res.message)
            rms_hb[2,i] = 1
    # print(res.message)
            
    print(f'Frequency: {omg:.1f} rad/s -> completed.')
    print('---------')
    
if calc:
    with open(f'data_rk f {f}.pic'.replace(':','_'),'wb') as file:
        dump([rms_rk,pc],file)

fig = go.Figure(data=[go.Scatter(x=omg_range,y=rms_hb[0,:],name='DoF 1 - HB'),
                      go.Scatter(x=omg_range,y=rms_hb[1,:],name='DoF 2 - HB'),
                      go.Scatter(x=[omg_range[i] for i in range(len(omg_range)) if rms_hb[2,i]==1],
                                 y=[rms_hb[0,i] for i in range(len(omg_range)) if rms_hb[2,i]==1],
                                 name='Flagged',mode='markers',marker=dict(color='black'),legendgroup='flag'),
                       go.Scatter(x=[omg_range[i] for i in range(len(omg_range)) if rms_hb[2,i]==1],
                                  y=[rms_hb[1,i] for i in range(len(omg_range)) if rms_hb[2,i]==1],
                                  legendgroup='flag',showlegend=False,mode='markers',marker=dict(color='black')),
                      go.Scatter(x=omg_range,y=rms_rk[0,:],name='DoF 1 - RK'),
                      go.Scatter(x=omg_range,y=rms_rk[1,:],name='DoF 2 - RK')])
fig.update_yaxes(type="log")
fig.write_html(f'FRF RK vs HB - f {f}.html'.replace(':','_'))

n_plot = 30
fig = go.Figure(data=[go.Scatter(x=np.repeat(omg_range,n_plot),
                                 y=np.array([a[0,-n_plot:] for a in pc]).flatten(),
                                 mode='markers',name='DoF 1 - HB'),])
fig.write_html(f'Bifurcation _ DoF 1 - f {f}.html'.replace(':','_'))

        
    
# t_hb = np.arange(0,tf,S.t(omg)[1])
# x_hb_ex = np.concatenate([x_hb]*(len(t_hb)//x_hb.shape[1]+1),axis=1)[:,:len(t_hb)]

# fig = go.Figure(data=[go.Scatter(x=t_hb,y=x_hb_ex[0,:],name='DoF 1 - HB'),
#                       go.Scatter(x=t_hb,y=x_hb_ex[1,:],name='DoF 2 - HB'),
#                       go.Scatter(x=t_rk,y=x_rk[0,:],name='DoF 1 - RK'),
#                       go.Scatter(x=t_rk,y=x_rk[1,:],name='DoF 2 - RK')])
# fig.write_html(f'RK vs HB _ omg  {omg} _ f {f}.html'.replace(':','_'))

