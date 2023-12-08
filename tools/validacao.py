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
nu = 2
N = 2 # sinal no tempo terá comprimento 2*N*n_harm

S = Sys_NL(M=M,K=K_lin+beta*Snl,Snl=Snl,beta=0,alpha=alpha,
           n_harm=n_harm,nu=nu,N=N,cp=cp,C=K_lin*cp)

S.dof_nl = [1]
S.x_eq = x_eq

omg = 11

f = {0: 0.9}
# try:
#     with open(f'data_rk f {f}.pic'.replace(':','_'),'rb') as file:
#         aux = load(file)
#         rms_rk = aux[0]
#         pc = aux[1]
#     calc = False
# except:
#     calc = True
    
calc = True

omg_range_up = np.arange(0.5,20,.05)
omg_range_down = np.arange(20,0.5,-.05)

for f0 in np.arange(0.01, 1, 0.05):
    
    f = {0: f0}
        
    # rms_hb = np.zeros((3,len(omg_range)))
    # rms_rk = np.zeros((2,len(omg_range)))
    # pc = []
    # cost_hb = []
    
    n_points = 50
    # z0 = S.z0(omg=omg_range[0],f_omg={0:0}) # None
    # x0 = np.array([[0],[x_eq],[0],[0]])
    
    string_f = f'{[(k,np.around(f[k],2)) for k in f]}'
    
    # fig1, fig2 = S.plot_frf(omg_range_up, f, dt_base=0.03, tf=180)
    # fig1.update_layout(title=f'Upsweep - f = {string_f}')
    # fig1.write_html(f'FRF/FRF Upsweep - f = {string_f}.html'.replace(':','_'))
    # # fig1.write_image(f'FRF/FRF Upsweep - f = {string_f}.pdf'.replace(':','_'))
    # fig1.write_image(f'FRF/FRF Upsweep - f = {string_f}.png'.replace(':','_'))
    
    # fig2.update_layout(title=f'Upsweep - f = {string_f}')
    # fig2.write_html(f'FRF/Cost Upsweep - f = {string_f}.html'.replace(':','_'))
    # # fig2.write_image(f'FRF/Cost Upsweep - f = {string_f}.pdf'.replace(':','_'))
    # fig2.write_image(f'FRF/Cost Upsweep - f = {string_f}.png'.replace(':','_'))
    
    # fig1, fig2 = S.plot_frf(omg_range_down, f, dt_base=0.03, tf=180)
    # fig1.update_layout(title=f'Downsweep - f = {string_f}')
    # fig1.write_html(f'FRF/FRF Downsweep - f = {string_f}.html'.replace(':','_'))
    # # fig1.write_image(f'FRF/FRF Downsweep - f = {string_f}.pdf'.replace(':','_'))
    # fig1.write_image(f'FRF/FRF Downsweep - f = {string_f}.png'.replace(':','_'))
    
    # fig2.update_layout(title=f'Downsweep - f = {string_f}')
    # fig2.write_html(f'FRF/Cost Downsweep - f = {string_f}.html'.replace(':','_'))
    # # fig2.write_image(f'FRF/Cost Downsweep - f = {string_f}.pdf'.replace(':','_'))
    # fig2.write_image(f'FRF/Cost Downsweep - f = {string_f}.png'.replace(':','_'))
    
    fig1, fig2 = S.plot_frf(omg_range_down, f, dt_base=0.03, tf=180, continuation='hb')
    fig1.update_layout(title=f'HB Est - f = {string_f}')
    fig1.write_html(f'FRF/FRF HB Est - f = {string_f}.html'.replace(':','_'))
    # fig1.write_image(f'FRF/FRF HB Est - f = {string_f}.pdf'.replace(':','_'))
    fig1.write_image(f'FRF/FRF HB Est - f = {string_f}.png'.replace(':','_'))
    
    fig2.update_layout(title=f'HB Est - f = {string_f}')
    fig2.write_html(f'FRF/Cost HB Est - f = {string_f}.html'.replace(':','_'))
    # fig2.write_image(f'FRF/Cost HB Est - f = {string_f}.pdf'.replace(':','_'))
    fig2.write_image(f'FRF/Cost HB Est - f = {string_f}.png'.replace(':','_'))
    
omg = 5

sp = 70

fig=go.Figure(data=[go.Scatter(x=q_p,y=h_p,name='Bomba 100% rotação'),
                    go.Scatter(x=q_p*sp/100,y=h_p*(sp/100)**2,name=f'Bomba {sp}% rotação'),
                    go.Scatter(x=q,y=p,mode='markers',name='Pontos de operação'),
                    ])
fig.update_layout(xaxis=dict(title='Vazão por bomba (m³/h)'),
                  yaxis=dict(title='Head (m)'),)

q_p = np.linspace(0,1.5*16500/2/24,100)
h_p = 3600 * 1.3 - 0.3*3600 * (q_p/(16500/2/24))**2

    
    
# SImulação individual


n_harm = 10
nu = 2
N = 50 # sinal no tempo terá comprimento 2*N*n_harm

S = Sys_NL(M=M,K=K_lin+beta*Snl,Snl=Snl,beta=0,alpha=alpha,
            n_harm=n_harm,nu=nu,N=N,cp=cp,C=K_lin*cp)

S.dof_nl = [1]
S.x_eq = x_eq

omg = 14.3
f = {0:0.21}

x0 = np.array([[0],[x_eq],[0],[0]])
tf = np.round(300/(2*np.pi/omg)) * 2*np.pi/omg
dt = 2*np.pi/omg / (np.round(2*np.pi/omg / 0.01))
t_rk = np.arange(0,tf + dt/2,dt)
z0 = S.z0(omg=omg,f_omg={0:0})

fig1, x0 = S.solve_hb(f,omg,z0=z0,plot_orbit=True,method=None,state_space=True)
fig2, x_rk = S.solve_transient(f,t_rk,omg,x0[:,0].reshape((4,1)),plot_orbit=True,dt=dt)

Ni = int(2*np.pi/omg / dt)
Np = int(len(t_rk) / Ni)
x2 = x_rk[:2,Np//2*Ni:Np//2*Ni+Ni*nu]
t2 = t_rk[Np//2*Ni:Np//2*Ni+Ni*nu] - t_rk[Np//2*Ni]

x = np.zeros((2*len(S.t(omg)),1))
x[:,0] = np.append(np.interp(S.t(omg)[:,0],t2,x2[0,:]),
                    np.interp(S.t(omg)[:,0],t2,x2[1,:]))
# x[1,:] = np.interp(S.t(omg)[:,0],x_rk[1,:],t_rk)
z = np.linalg.pinv(S.gamma(omg)) @ x

fig3, x0 = S.solve_hb(f,omg,z0=z,plot_orbit=True,method=None,state_space=True)

z2 = np.ones(z0.shape) *1e-5
z2[0:2] = z0[0:2]
fig4, x0 = S.solve_hb(f,omg,z0=z2,plot_orbit=True,method=None,state_space=True)
    
# fig2.add_trace(fig1.data[0])
# fig2.add_trace(fig1.data[1])
# fig2.add_trace(fig1.data[2])
# fig2.add_trace(fig1.data[3])    
fig2.add_trace(fig3.data[0])
fig2.add_trace(fig3.data[1])
fig2.add_trace(fig3.data[2])
fig2.add_trace(fig3.data[3]) 
fig2.show()
    
w=np.arange(len(t_rk[20000:]))*dt
N=len(t_rk[20000:])
w=np.arange(N//2)*(1/(dt*N))
spec=2/N*np.abs(np.fft.fft(x_rk[0,20000:]))
f_spec=go.Figure(data=go.Scatter(x=w,y=spec))
f_spec.update_layout(xaxis=dict(range=(0,8)))

f_spec.update_yaxes(type='log')
    
    
    
    
# def plot_frf(omg_range, )
    
#     for i, omg in enumerate(omg_range):
#         tf = np.round(300/(2*np.pi/omg)) * 2*np.pi/omg
#         dt = 2*np.pi/omg / (np.round(2*np.pi/omg / 0.01))
#         t_rk = np.arange(0,tf + dt/2,dt)
#         t0 = time.time()
#         if calc:        
#             x_rk, x0 = S.solve_transient(f,t_rk,omg,x0.reshape((4,1)),last_x=True,dt=dt)
#             print(f'RK4 took {(time.time()-t0):.1f} seconds to run.')
#         t1 = time.time()
#         z0 = S.z0(omg=omg_range[0],f_omg={0:0}) # None
#         x_hb, res = S.solve_hb(f,omg,z0=z0,full_output=True,method=None)#'ls')
        
#         try:
#             cost_hb.append(res.cost)
#             z0 = res.x
#         except:
#             cost_hb.append(np.linalg.norm(res[1]['fvec']))
#             z0 = res[0]
#         print(f'Harmbal took {(time.time()-t1):.1f} seconds to run: {(t1-t0)/(time.time()-t1):.1f} times faster.')
        
#         if calc:
#             rms_rk[0,i] = np.sqrt(np.sum((x_rk[0,int((tf/2)/dt):] - np.mean(x_rk[0,int((tf/2)/dt):])) ** 2) / (int((tf/2)/dt)))
#             rms_rk[1,i] = np.sqrt(np.sum((x_rk[1,int((tf/2)/dt):] - np.mean(x_rk[1,int((tf/2)/dt):])) ** 2) / (int((tf/2)/dt)))
#             pc.append(pcs(x_rk, t_rk, omg, n_points))
        
#         rms_hb[0,i] = np.sqrt(np.sum((x_hb[0,:] - np.mean(x_hb[0,:])) ** 2) / (len(S.t(omg))-1))
#         rms_hb[1,i] = np.sqrt(np.sum((x_hb[1,:] - np.mean(x_hb[1,:])) ** 2) / (len(S.t(omg))-1))
#         try:
#             if res[-2] != 1:
#                 print(res[-1])
#                 rms_hb[2,i] = 1
#         except:
#             if not res.success:
#                 print(res.message)
#                 rms_hb[2,i] = 1
#         # print(res.message)
                
#         print(f'Frequency: {omg:.1f} rad/s -> completed.')
#         print('---------')
        
#     if calc:
#         with open(f'data_rk f {f} _ up.pic'.replace(':','_'),'wb') as file:
#             dump([rms_rk,pc],file)
    
#     fig = go.Figure(data=[go.Scatter(x=omg_range,y=rms_hb[0,:],name='DoF 1 - HB'),
#                           go.Scatter(x=omg_range,y=rms_hb[1,:],name='DoF 2 - HB'),
#                           go.Scatter(x=[omg_range[i] for i in range(len(omg_range)) if rms_hb[2,i]==1],
#                                      y=[rms_hb[0,i] for i in range(len(omg_range)) if rms_hb[2,i]==1],
#                                      name='Flagged',mode='markers',marker=dict(color='black'),legendgroup='flag'),
#                            go.Scatter(x=[omg_range[i] for i in range(len(omg_range)) if rms_hb[2,i]==1],
#                                       y=[rms_hb[1,i] for i in range(len(omg_range)) if rms_hb[2,i]==1],
#                                       legendgroup='flag',showlegend=False,mode='markers',marker=dict(color='black')),
#                           go.Scatter(x=omg_range,y=rms_rk[0,:],name='DoF 1 - RK'),
#                           go.Scatter(x=omg_range,y=rms_rk[1,:],name='DoF 2 - RK')])
#     fig.update_yaxes(type="log")
#     # fig.write_html(f'FRF RK vs HB - f {f}.html'.replace(':','_'))
    
#     fig2 = go.Figure(data=[go.Scatter(x=omg_range,y=cost_hb,name='DoF 1 - HB'),
#                           go.Scatter(x=[omg_range[i] for i in range(len(omg_range)) if rms_hb[2,i]==1],
#                                      y=[cost_hb[i] for i in range(len(omg_range)) if rms_hb[2,i]==1],
#                                      name='Flagged',mode='markers',marker=dict(color='black'),legendgroup='flag'),
#                            ])
#     fig2.update_yaxes(type="log")

# # n_plot = 30
# # fig = go.Figure(data=[go.Scatter(x=np.repeat(omg_range,n_plot),
# #                                  y=np.array([a[0,-n_plot:] for a in pc]).flatten(),
# #                                  mode='markers',name='DoF 1 - HB'),])
# # fig.write_html(f'Bifurcation _ DoF 1 - f {f}.html'.replace(':','_'))




# # Plot rotor HB

# omg_range=np.linspace(1,800,200)
# omg_range2 = omg_range #np.linspace(300,400,25)
# N=len(omg_range)
# with open(r'results/out_data_r_det_flextun.pic','rb') as file:
#     data0=load(file)
# r_solo = np.array([data0['rsolo_b_map'][j,j] for j in range(N)])
# r_lin = np.array([data0['r_b_map'][j,j] for j in range(N)])


# with open('data_rk f {0_ 1, 1_ (-0-1j)} flextun.pic','rb') as file:
#     data=load(file)
# N=len(omg_range)
# cost_hb=data[1]
# rms_hb=data[0]

# fig = go.Figure(data=[go.Scatter(x=omg_range2,y=rms_hb[0,:],name='DoF 1 - HB'),
#                       go.Scatter(x=[omg_range[i] for i in range(len(omg_range2)) if rms_hb[1,i]==1],
#                                   y=[rms_hb[0,i] for i in range(len(omg_range2)) if rms_hb[1,i]==1],
#                                   name='Flagged',mode='markers',marker=dict(color='black'),legendgroup='flag'),
#                                   go.Scatter(x=[omg_range2[i] for i in range(len(omg_range2)) if rms_hb[1,i]==2],
#             y=[rms_hb[0,i] for i in range(len(omg_range2)) if rms_hb[1,i]==2],
#             name='Flagged',mode='markers',marker=dict(color='red'),legendgroup='flag'),
#                                   go.Scatter(x=omg_range,y=1*r_solo/np.sqrt(2),name='Bare Rotor'),
#                                   go.Scatter(x=omg_range,y=1*r_lin/np.sqrt(2),name='Linear Resonators')])
# fig.update_yaxes(type="log")


# # FFT

# with open('x_rk 330.pic','rb') as file:
#     data=load(file)
# N=len(data[1])
# t=data[1][2*N//3:]
# x=data[0][2,:][2*N//3:]

# dt=t[1]-t[0]
# tf=dt*len(t)
# dw=1/tf
# wf=1/(2*dt)
# w=np.arange(0,330*10/2/np.pi,dw)
# spec=np.abs(np.fft.fft(x)[:len(w)])
# fig=go.Figure(data=[go.Scatter(x=w*2*np.pi,y=spec)])

        
    
# t_hb = np.arange(0,tf,S.t(omg)[1])
# x_hb_ex = np.concatenate([x_hb]*(len(t_hb)//x_hb.shape[1]+1),axis=1)[:,:len(t_hb)]

# fig = go.Figure(data=[go.Scatter(x=t_hb,y=x_hb_ex[0,:],name='DoF 1 - HB'),
#                       go.Scatter(x=t_hb,y=x_hb_ex[1,:],name='DoF 2 - HB'),
#                       go.Scatter(x=t_rk,y=x_rk[0,:],name='DoF 1 - RK'),
#                       go.Scatter(x=t_rk,y=x_rk[1,:],name='DoF 2 - RK')])
# fig.write_html(f'RK vs HB _ omg  {omg} _ f {f}.html'.replace(':','_'))

