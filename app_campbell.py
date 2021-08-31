# -*- coding: utf-8 -*-

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import socket
import rotor_mtm as rmtm
import ross as rs

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Definindo rotores

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
f_0 = 3770 #10000 # 1800/60*2*np.pi # em rad/s
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

sp_arr = np.linspace(1,800,100)

r_det_flextun = rmtm.RotorMTM(rotor,n_pos,dk_r,k0,k1,var=0,var_k=0,p_damp=1e-4,ge=True)
r_var_flextun = rmtm.RotorMTM(rotor,n_pos,dk_r,k0,k1,var=var,var_k=0,p_damp=1e-4,ge=True,exp_var=3)
r_var3_flextun = rmtm.RotorMTM(rotor,n_pos,dk_r,k0,k1,var=var,var_k=0,p_damp=1e-4,ge=True,exp_var=3)

r_det_transtun = rmtm.RotorMTM(rotor,n_pos,dk_r,k0/100,100*k1,var=0,var_k=0,p_damp=1e-4,ge=True)
r_var_transtun = rmtm.RotorMTM(rotor,n_pos,dk_r,k0/100,100*k1,var=var,var_k=0,p_damp=1e-4,ge=True,exp_var=1)
r_var3_transtun = rmtm.RotorMTM(rotor,n_pos,dk_r,k0/100,100*k1,var=var,var_k=0,p_damp=1e-4,ge=True,exp_var=3)

rotor_dict = {'Rotor flextuned':r_det_flextun,
              'Rotor rainbow flextuned':r_var_flextun,
              'Rotor rainbow 3 flextuned':r_var3_flextun,
              'Rotor transtuned':r_det_transtun,
              'Rotor rainbow transtuned':r_var_transtun,
              'Rotor rainbow 3 transtuned':r_var3_transtun,}

# Definindo elementos do app

def generate_table(dataframe, max_rows=18):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

G = nx.Graph()



app.layout = html.Div([
    html.H1('MTM Rotor - Campbell and mode shape generator.'),
    html.Label('Choose your rotor model.'),
    html.Div([
        dcc.Dropdown(
            options=[{'label': k, 'value': k} for k in rotor_dict],
            value=None,
            multi=False,
            id='select-rotor'),
        ]),
    html.Div(children=[
                 html.Label('Choose the displayed coordinate.'),
                 dcc.RadioItems(id='select-coord',
                                options=[{'label':'X','value':0},
                                         {'label':'Y','value':1},
                                         {'label':'theta_X','value':2},
                                         {'label':'theta_Y','value':3}],
                                value=0,
                                className='two columns')
                 ]),
    dcc.Graph(
             id='fig-campbell',
    ),
    html.Label('Mode shape:'),
    dcc.Graph(
        id='fig-mode-shape'
    ),
    ])        




@app.callback(
    Output('fig-mode-shape', 'figure'),
    [Input('fig-campbell', 'clickData'),
     Input('select-rotor', 'value'),
     Input('select-coord', 'value'),])
def show_click_info(click_data,select_rotor,select_coord):
    if click_data == None or len(click_data.keys()) == 0:
        fig = go.Figure()
    else:
        ij = [int(a) for a in click_data['points'][0]['text'].split()]
        
        _,_,u_list,*_ = rotor_dict[select_rotor].omg_list(sp_arr[ij[0]],diff_lim=1e10)
        u = u_list[ij[1]]
        plot_orbits = [0]+list(n_pos)+[rotor_dict[select_rotor].rotor_solo_disks.nodes[-1]]
        fig1 = rmtm.plot_deflected_shape(rotor_dict[select_rotor].rotor_solo_disks,u,n_pos,'trans',plot_orbits)
        fig2 = rmtm.plot_deflected_shape(rotor_dict[select_rotor].rotor_solo_disks,u,n_pos,'flex',plot_orbits)
        fig = make_subplots(rows=1, cols=2, 
                            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])
        for i in range(len(fig1.data)):
            fig.add_trace(fig1.data[i],row=1,col=1)
            fig.add_trace(fig2.data[i],row=1,col=2)
        fig.layout.scene = fig1.layout.scene
        fig.layout.scene2 = fig2.layout.scene
        
    return fig

# @app.callback(
#     Output('select-doc', 'options'),
#     [Input('select-disc', 'value')])
# def drop_docs(select_disc):
#     if select_disc == None:
#         return []
#     elif len(select_disc) > 0:
#         df = pd.DataFrame(columns = df_int.columns)
#         for d in select_disc:
#             ind_select = [d == str(df_int.iloc[i,0]) or d == str(df_int.iloc[i,4]) for i in range(len(df_int))]
#             df = pd.concat([df,df_int.loc[ind_select]])
#         docs = set(list(df.iloc[:,1].unique())+list(df.iloc[:,5].unique()))
#         docs = [str(d) for d in docs]
#         docs.sort()
#         opt = [{'label': str(docs[i]), 'value': str(docs[i])} for i in range(len(docs))]
#         return opt
#     else:
#         return []
        

@app.callback(
    Output('fig-campbell', 'figure'),
    [Input('select-rotor', 'value'),
     Input('select-coord', 'value'),])
def update_campbell(select_rotor,select_coord):
    
    if select_rotor is None:
        fig3 = figure=make_subplots(rows=1, cols=2, )
    else:
        out = rotor_dict[select_rotor].run_analysis(sp_arr,diff_lim=1e10,diff_analysis=True,dof=select_coord)
        fig1 = rmtm.plot_diff_modal(out['w'],out['diff'],sp_arr,mode='abs',colorbar_left=True)
        fig2 = rmtm.plot_diff_modal(out['w'],out['diff'],sp_arr,mode='phase')
        # fig3 = [dcc.Graph(figure=fig1),
        #         dcc.Graph(figure=fig2)]
        
        fig3 = make_subplots(rows=1, cols=2)
        for i in range(len(fig1.data)):
            fig3.add_trace(fig1.data[i],row=1,col=1)
            fig3.add_trace(fig2.data[i],row=1,col=2)
        fig3.layout.xaxis.range = (sp_arr[0],sp_arr[-1])
        fig3.layout.xaxis2.range = (sp_arr[0],sp_arr[-1])
        fig3.layout.yaxis.range = (sp_arr[0],sp_arr[-1])
        fig3.layout.yaxis2.range = (sp_arr[0],sp_arr[-1])
    
    return fig3

ip_adress = socket.gethostbyname(socket.gethostname())

if __name__ == '__main__':
    app.run_server(debug=False, host=ip_adress)
