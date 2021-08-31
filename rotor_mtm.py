
import ross as rs
import numpy as np
from scipy import linalg as la
import sys
import plotly.graph_objects as go


class RotorMTM:

    def __init__(self,rotor,n_pos,dk_r,k0,k1,var=0,var_k=0,p_damp=1e-4,ge=True,exp_var=1):

        self.n_res = len(n_pos)
        self.var = var
        self.var_k = var_k
        self.p_damp = p_damp #1e-4
        self.ge = ge #True
        self.dk_r = dk_r #rs.DiskElement(n=0, m=mr, Id=It, Ip=Ip)
        self.k0 = k0 #mr * f_0 ** 2
        self.k1 = k1 #It * f_1 ** 2
        self.exp_var = exp_var

        self.n_pos = n_pos #np.arange(n_center - int(n_res / 2), n_center + n_res - int(n_res / 2), 1)

        if isinstance(dk_r,list):
            self.dk_r = dk_r
        else:
            self.dk_r = [rs.DiskElement(n=n_pos[i],m=dk_r.m,Id=dk_r.Id,Ip=dk_r.Ip,tag=f'{i}') for i in range(len(n_pos))]

        for i, d in enumerate(self.dk_r):
            self.dk_r[i].scale_factor = 0.4 * d.m/max([a.m for a in self.dk_r])
            self.dk_r[i].color = 'blue'

        PM = [rs.PointMass(rotor.bearing_elements[0].n, 0, tag=f'Res_{a}') for a in range(2 * self.n_res)]

        self.rotor_solo_disks = rs.Rotor(rotor.shaft_elements, rotor.disk_elements + self.dk_r, rotor.bearing_elements)
        self.rotor_solo = rs.Rotor(rotor.shaft_elements, rotor.disk_elements, rotor.bearing_elements)
        self._rotor = rs.Rotor(rotor.shaft_elements, rotor.disk_elements, rotor.bearing_elements, PM)

        self.m_ratio = np.sum([a.m  for a in self.dk_r]) / (self.rotor_solo.m)

        self.N = len(self._rotor.M())
        self.N2 = len(self.rotor_solo.M())

    def M(self):

        N = 4 * self.n_res

        M = self._rotor.M()

        M_add = np.zeros((N, N))
        for i in range(0, N, 4):
            f = 1 + (self.var / 2) * (2 * (i // 4) / (self.n_res - 1) - 1) ** self.exp_var
            M_add[i:i + 4, i:i + 4] = self.dk_r[i//4].M() * f

        dof = range(self.N2, N + self.N2)

        M[np.ix_(dof, dof)] = M_add

        return M

    def G(self):

        N = 4 * self.n_res

        G = self._rotor.G()

        G_add = np.zeros((N, N))
        for i in range(0, N, 4):
            f = 1 + (self.var / 2) * (2 * (i // 4) / (self.n_res - 1) - 1) ** self.exp_var
            G_add[i:i + 4, i:i + 4] = self.dk_r[i//4].G() * f

        dof = range(self.N2, N + self.N2)

        G[np.ix_(dof, dof)] = G_add

        return G

    def K(self,sp):#K, C, n_pos, k0, k1, var=0, p_damp=0):

        N = 4 * self.n_res

        K = self._rotor.K(sp)

        K_add = np.zeros((2 * N, 2 * N))
        C_add = np.zeros((2 * N, 2 * N))
        
        dof = []
        for i, n in enumerate(self.n_pos):
            dof += [a for a in range(4 * n, 4 * n + 4)]
            f = (1 - self.var_k / 2) + i / (self.n_res - 1) * self.var_k
            K_aux = np.array([[self.k0 * f, 0, 0, 0, -self.k0 * f, 0, 0, 0],
                              [0, self.k0 * f, 0, 0, 0, -self.k0 * f, 0, 0],
                              [0, 0, self.k1 * f, 0, 0, 0, -self.k1 * f, 0],
                              [0, 0, 0, self.k1 * f, 0, 0, 0, -self.k1 * f],
                              [-self.k0 * f, 0, 0, 0, self.k0 * f, 0, 0, 0],
                              [0, -self.k0 * f, 0, 0, 0, self.k0 * f, 0, 0],
                              [0, 0, -self.k1 * f, 0, 0, 0, self.k1 * f, 0],
                              [0, 0, 0, -self.k1 * f, 0, 0, 0, self.k1 * f], ])
            dof_aux = [4 * i + a for a in range(4)] + [N + 4 * i + a for a in range(4)]
            K_add[np.ix_(dof_aux, dof_aux)] = K_aux

        dof = dof + [a for a in range(self.N2, N + self.N2)]
        # print(dof)

        K[np.ix_(dof, dof)] += K_add

        return K

    def C(self, sp):  # K, C, n_pos, k0, k1, var=0, p_damp=0):

        N = 4 * self.n_res

        C = self._rotor.C(sp)

        C_add = np.zeros((2 * N, 2 * N))
        dof = []
        for i, n in enumerate(self.n_pos):
            dof += [a for a in range(4 * n, 4 * n + 4)]
            f = (1 - self.var_k / 2) + i / (self.n_res - 1) * self.var_k
            K_aux = np.array([[self.k0 * f, 0, 0, 0, -self.k0 * f, 0, 0, 0],
                              [0, self.k0 * f, 0, 0, 0, -self.k0 * f, 0, 0],
                              [0, 0, self.k1 * f, 0, 0, 0, -self.k1 * f, 0],
                              [0, 0, 0, self.k1 * f, 0, 0, 0, -self.k1 * f],
                              [-self.k0 * f, 0, 0, 0, self.k0 * f, 0, 0, 0],
                              [0, -self.k0 * f, 0, 0, 0, self.k0 * f, 0, 0],
                              [0, 0, -self.k1 * f, 0, 0, 0, self.k1 * f, 0],
                              [0, 0, 0, -self.k1 * f, 0, 0, 0, self.k1 * f], ])
            dof_aux = [4 * i + a for a in range(4)] + [N + 4 * i + a for a in range(4)]
            C_add[np.ix_(dof_aux, dof_aux)] = self.p_damp * K_aux

        dof = dof + [a for a in range(self.N2, N + self.N2)]
        # print(dof)

        C[np.ix_(dof, dof)] += C_add

        return C

    def A(self,sp):

        M = self.M()
        K = self.K(sp)
        C = self.C(sp)
        G = self.G()

        N = len(M)

        Z = np.zeros((N, N))
        I = np.eye(N)

        A = np.vstack(
            [np.hstack([Z, I]),
             np.hstack([la.solve(-M, K), la.solve(-M, (C + self.ge * sp * G))])])

        return A

    def plot_rotor(self):

        return self.rotor_solo_disks.plot_rotor()

    def calc_H(self,sp,f,rotor_solo=False):#K, C, G, M, sp):

        if rotor_solo:
            M = self.rotor_solo_disks.M()
            A = self.rotor_solo_disks.A(sp)
        else:
            M = self.M()
            A = self.A(sp)

        N = len(M)
        Z = np.zeros((N, N))
        M_inv = np.vstack([np.hstack([Z, Z]),
                           np.hstack([Z, la.inv(M)])])

        H = np.linalg.inv(1.j * f * np.eye(2 * N) - A)

        return H, M_inv

    def omg_list(self,sp,n_modes=50, dof=None, diff_lim=2, rotor_solo=False,
                 cross_prod=False, energy=False):

        if rotor_solo:
            A = self.rotor_solo_disks.A(sp)
        else:
            A = self.A(sp)

        if dof is None:
            step = 1
            dof = 0
        else:
            step = 4

        eig = la.eig(A)
        N1 = len(eig[0])
        w = eig[0][::2]
        u = eig[1][:int(N1 / 2), ::2]

        i_aux = np.argsort(np.abs(np.imag(w)))
        i_aux = np.array([a for a in i_aux if np.abs(np.real(w[a]) / np.imag(w[a])) < 1])
        csi = np.abs(np.real(w[i_aux]))
        w = np.abs(np.imag(w[i_aux]))
        u = u[:, i_aux]

        omg_list = []
        omg_list_res = []
        u_list = []
        u_list_res = []
        csi_list = []
        csi_list_res = []
        diff_list = []
        diff_list_res = []

        for i in range(n_modes):  # i < len(w) and w[i] <= w_max :
            if not rotor_solo:

                diff2 = res_diff(u[dof:N1 // 2:step, i], self.n_pos)
                if cross_prod:
                    diff = res_diff(u[dof:N1 // 2:step, i], self.n_pos, cross_prod=cross_prod)
                elif energy:
                    diff = res_diff(u[dof::step, i], self.n_pos, energy=energy)
                else:
                    diff = diff2

                if np.sum([a > diff_lim for a in
                           np.abs(diff2)]) >= 1:  # and np.sum([a>0.2 for a in np.angle(diff)]) > 3:#2*s_rotor > s_res_rel:
                    omg_list_res.append(w[i])
                    u_list_res.append(u[:, i])
                    csi_list_res.append(csi[i])
                    diff_list_res.append(diff)
                else:  # if s_res_abs > s_rotor:
                    omg_list.append(w[i])
                    u_list.append(u[:, i])
                    csi_list.append(csi[i])
                    diff_list.append(diff)

            else:
                omg_list.append(w[i])
                u_list.append(u[:, i])
                csi_list.append(csi[i])

            i += 1

        return omg_list, omg_list_res, u_list, u_list_res, csi_list, csi_list_res, diff_list, diff_list_res

    def run_analysis(self, sp_arr, n_modes=50, dof=0, dof_show=0, diff_lim=5,unb_node=0,probe_node=None,
                     heatmap=False, diff_analysis=False, cross_prod=False, energy=False):

        prog_bar_width = 40
        i_prog = 0
        sys.stdout.write("[%s]" % (" " * prog_bar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (prog_bar_width+1)) 

        if probe_node is None:
            probe_node = self.N2//4 - 1

        N = self.N
        N2 = self.N2

        # Variáveis de análise modal
        ws = []
        w = []
        w_res = []
        diff = []
        diff_res = []

        # Varíaveis para construção de heatmaps
        rsolo_map = []
        rsolo_b_map = []
        r_map = []
        r_b_map = []
        diff_map = []
        diff_map_b = []

        w_max = sp_arr[-1]
        if heatmap:
            sp_arr2 = sp_arr
        else:
            sp_arr2 = [0]

        for j, f in enumerate(sp_arr2):

            rsolo = []
            rsolo_b = []
            r = []
            r_b = []
            r_diff = []
            r_diff_b = []

            for i, sp in enumerate(sp_arr):

                if not heatmap:
                    f0 = sp  # 460
                else:
                    f0 = f
                
                Hs, Ms_inv = self.calc_H(sp,f0,rotor_solo=True)
                # As = self.rotor_solo_disks.A(sp)

                H, M_inv = self.calc_H(sp,f0)
                # A = self.A(sp)


                Fs = np.zeros((2 * N2, 1)).astype(complex)
                Fs[N2 + 4 * unb_node] = 1
                Fs[N2 + 4 * unb_node + 1] = -1.j
                Fs = Ms_inv @ Fs
                ys = Hs @ Fs
                fow = ys[dof_show + 4 * probe_node] / 2 + 1.j / 2 * ys[dof_show + 4 * probe_node + 1]
                back = np.conj(ys[dof_show + 4 * probe_node]) / 2 + 1.j / 2 * np.conj(
                    ys[dof_show + 4 * probe_node + 1])
                rsolo.append(np.abs(fow) + np.abs(back))

                Fs_b = np.zeros((2 * N2, 1)).astype(complex)
                Fs_b[N2 + 4 * unb_node] = 1
                Fs_b[N2 + 4 * unb_node + 1] = 1.j
                Fs_b = Ms_inv @ Fs_b
                ys_b = Hs @ Fs_b
                fow = ys_b[dof_show + 4 * probe_node] / 2 + 1.j / 2 * ys_b[dof_show + 4 * probe_node + 1]
                back = np.conj(ys_b[dof_show + 4 * probe_node]) / 2 + 1.j / 2 * np.conj(
                    ys_b[dof_show + 4 * probe_node + 1])
                rsolo_b.append(np.abs(fow) + np.abs(back))

                F = np.zeros((2 * N, 1)).astype(complex)
                F[N + 4 * unb_node] = 1
                F[N + 4 * unb_node + 1] = -1.j

                F_b = np.zeros((2 * N, 1)).astype(complex)
                F_b[N + 4 * unb_node] = 1
                F_b[N + 4 * unb_node + 1] = 1.j

                y = H @ (M_inv @ F)
                fow = y[dof_show::4] / 2 + 1.j / 2 * y[dof_show + 1::4]
                back = np.conj(y[dof_show::4]) / 2 + 1.j / 2 * np.conj(y[dof_show + 1::4])
                if energy:
                    r_diff.append(res_diff(y[dof_show::4], self.n_pos, cross_prod=cross_prod, energy=energy))
                else:
                    r_diff.append(res_diff(fow[:N // 4], self.n_pos, cross_prod=cross_prod, energy=energy))
                r.append(np.abs(fow[probe_node]) + np.abs(back[probe_node]))

                y_b = H @ (M_inv @ F_b)
                fow = y_b[dof_show::4] / 2 + 1.j / 2 * y_b[dof_show + 1::4]
                back = np.conj(y_b[dof_show::4]) / 2 + 1.j / 2 * np.conj(y_b[dof_show + 1::4])
                if energy:
                    r_diff_b.append(res_diff(y_b[dof_show::4], self.n_pos, cross_prod=cross_prod, energy=energy))
                else:
                    r_diff_b.append(res_diff(back[:N // 4], self.n_pos, cross_prod=cross_prod, energy=energy))
                r_b.append(np.abs(fow[probe_node]) + np.abs(back[probe_node]))

                if f == sp_arr2[0]:
                    aux = self.omg_list(sp,n_modes, dof=dof, diff_lim=diff_lim, rotor_solo=False,
                                        cross_prod=cross_prod, energy=cross_prod)
                    w.append(aux[0])
                    w_res.append(aux[1])
                    diff.append(np.array(aux[6]))
                    diff_res.append(np.array(aux[7]))

                    ws.append(self.omg_list(sp,n_modes,rotor_solo=True)[0])

                if not heatmap:
                    
                    if prog_bar_width * sp/max(sp_arr) > i_prog:
                        sys.stdout.write("-")
                        sys.stdout.flush()
                        i_prog += 1
                    
                   

            if heatmap:
                rsolo_map.append(rsolo)
                rsolo_b_map.append(rsolo_b)
                r_map.append(r)
                r_b_map.append(r_b)
                diff_map.append(r_diff)
                diff_map_b.append(r_diff_b)
                if prog_bar_width * f/max(sp_arr) > i_prog:
                    sys.stdout.write("-")
                    sys.stdout.flush()
                    i_prog += 1

        out = dict(ws=ws,
                   w=w,
                   w_res=w_res,
                   sp_arr=sp_arr)
        if heatmap:
            out.update(dict(rsolo_map=np.array(rsolo_map).reshape((len(sp_arr), len(sp_arr))),
                            rsolo_b_map=np.array(rsolo_b_map).reshape((len(sp_arr), len(sp_arr))),
                            r_map=np.array(r_map).reshape((len(sp_arr), len(sp_arr))),
                            r_b_map=np.array(r_b_map).reshape((len(sp_arr), len(sp_arr)))))
        else:
            
            out.update(dict(rsolo=np.array(rsolo),
                            rsolo_b=np.array(rsolo_b),
                            r=np.array(r),
                            r_b=np.array(r_b)))

        if diff_analysis:
            out.update(dict(diff=diff,
                            diff_res=diff_res))
            if heatmap:
                out.update(dict(diff_map=diff_map,
                                diff_map_b=diff_map_b,))
            else:
                out.update(dict(r_diff=r_diff,
                                r_diff_b=r_diff_b))

        return out


def plot_diff_modal(w, diff, sp_arr, mode='abs',n_plot=None,saturate=None, colorbar_left=False):



    if mode == 'abs':
        
        if n_plot is None:
            if colorbar_left:
                x0 = -0.2
            else:
                x0 = None
            data = [go.Scatter(x=[sp_arr[a]] * len(w),
                               y=w[a], mode='markers',
                               text=[f'{a} {i}' for i in range(len(w[a]))],
                               marker={'color': (np.mean(np.abs(diff[a]), 1)),
                                        'colorscale':["blue", "purple", "yellow"],
                                       'colorbar': dict(title=(['Amplification (log)'] + [None] * (len(sp_arr) - 1))[a],
                                                        x=([x0] + [None] * (len(sp_arr) - 1))[a],),
                                       'size': 3,
                                        'cmin': 0,
                                        'cmax': saturate
                                       },
                               showlegend=False) for a in range(len(sp_arr))]
        else:
            data = []
            for n in range(n_plot):
                if n == 0:
                    colorbar =(['Amplification (log)'] + [None] * (len(sp_arr) - 1))
                else:
                    colorbar = [None] * (len(sp_arr))
                data += [go.Scatter3d(y=[n] * len(w), x=[sp_arr[a]] * len(w),
                               z=w[a], mode='markers',
                               text=(np.abs(diff[a])[:,n]),
                               marker={'color': (np.abs(diff[a])[:,n]),
                                        'colorscale':["blue", "purple", "yellow"],
                                       'colorbar': dict(title=colorbar[a]),
                                       'size': 3,
                                        'cmin': 0, #np.log10(np.min(np.abs(diff))),
                                        'cmax': saturate #3.5#np.log10(np.max(np.abs(diff)))
                                       },
                               showlegend=False) for a in range(len(sp_arr))]
    if mode == 'phase':
        if n_plot is None:
            data = [go.Scatter(x=[sp_arr[a]] * len(w),
                               y=w[a], mode='markers',
                               text=[f'{a} {i}' for i in range(len(w[a]))],
                               marker={'color': (np.mean(np.angle(diff[a]), 1)),
                                       # 'colorscale':'spectral',
                                       'colorbar': dict(title=(['Phase (rad)'] + [None] * (len(sp_arr) - 1))[a],),
                                       'size': 3,
                                       'cmin': 0,
                                       'cmax': np.pi
                                       },
                               showlegend=False) for a in range(len(sp_arr))]
            
        else:
            data = []
            for n in range(n_plot):
                if n == 0:
                    colorbar =(['Phase (rad)'] + [None] * (len(sp_arr) - 1))
                else:
                    colorbar = [None] * (len(sp_arr))
                data += [go.Scatter3d(y=[n] * len(w), x=[sp_arr[a]] * len(w),
                               z=w[a], mode='markers',
                               text=np.angle(diff[a])[:,n],
                               marker={'color': np.angle(diff[a])[:,n],
                                        'colorscale':["blue", "purple", "yellow"],
                                       'colorbar': dict(title=colorbar[a]),
                                       'size': 3,
                                        'cmin': 0,
                                        'cmax': np.pi
                                       },
                               showlegend=False) for a in range(len(sp_arr))]
                
    if n_plot is None:
        
       
        fig = go.Figure(data=data)
        
            
        fig.update_layout(title={'xanchor': 'center',
                                  'x': 0.4,
                                  'font': {'family': 'Arial, bold',
                                          'size': 15},
                                  'text': f'Campbell Diagram'},
                          yaxis={'range': [0, sp_arr[-1]],
                            'dtick': 50,
                            "gridcolor": "rgb(159, 197, 232)",
                            "zerolinecolor": "rgb(74, 134, 232)"},
                        xaxis={'range': [0, sp_arr[-1]],
                                'dtick': 50,
                                "gridcolor": "rgb(159, 197, 232)",
                                "zerolinecolor": "rgb(74, 134, 232)"},
                          xaxis_title='Speed (rad/s)',
                          yaxis_title='Natural frequencies (rad/s)',
                          font=dict(family="Calibri, bold",
                                    size=15),
                          legend=dict(xanchor='center', x=0.5, yanchor='bottom',
                                      y=1, orientation='h'))
        if mode == 'abs' and colorbar_left:
            fig.update_layout(coloraxis_colorbar_x=-0.15)
    else:
        scene = dict(zaxis={'range': [0, sp_arr[-1]],
                           'dtick': 50,
                           "gridcolor": "rgb(159, 197, 232)",
                           "zerolinecolor": "rgb(74, 134, 232)"},
                    xaxis={'range': [0, sp_arr[-1]],
                           'dtick': 50,
                           "gridcolor": "rgb(159, 197, 232)",
                           "zerolinecolor": "rgb(74, 134, 232)"},)
        fig = go.Figure(data=data)
        fig.update_layout(title={'xanchor': 'center',
                                 'x': 0.4,
                                 'font': {'family': 'Arial, bold',
                                          'size': 15},
                                 'text': f'Campbell Diagram'},
                          scene=scene,
                          xaxis_title='Speed (rad/s)',
                          yaxis_title='Natural frequencies (rad/s)',
                          font=dict(family="Calibri, bold",
                                    size=15),
                          legend=dict(xanchor='center', x=0.5, yanchor='bottom',
                                      y=1, orientation='h'))
        
   

    return fig

def plot_camp_heatmap(r, w, sp_arr, w_res=None, colorbar_title='Response (log)'):

    w_max = sp_arr[-1]
    w_min = sp_arr[0]

    if w_res != None:
        data_res = [go.Scatter(x=[sp_arr[a] for a in range(len(sp_arr)) if len(w_res[a]) > b],
                               y=[w_res[a][b] for a in range(len(sp_arr)) if len(w_res[a]) > b], mode='markers',
                               marker={'color': 'grey', 'size': 2.5, 'line': dict(width=0.05)},
                               marker_symbol='x-thin', showlegend=False) for b in
                    range(max([len(c) for c in w_res]))]
    else:
        data_res = []

    fig = go.Figure(
        data=[go.Heatmap(x=sp_arr, y=sp_arr, z=np.log10(np.abs(r)), colorbar=dict(title=colorbar_title))] + \
             [go.Scatter(x=[sp_arr[a] for a in range(len(sp_arr)) if len(w[a]) > b],
                         y=[w[a][b] for a in range(len(sp_arr)) if len(w[a]) > b], mode='markers',
                         marker={'color': 'black', 'size': 3}, showlegend=False) for b in range(max([len(c) for c in w]))] + \
             data_res + \
             [go.Scatter(x=[0, w_max], y=[0, w_max], mode='lines', line={'dash': 'dash', 'color': 'blue'},
                         name='Synch. Frequency'),
              go.Scatter(x=[w_max, w_max + 1], y=[w_max, w_max + 1], mode='markers', marker_symbol='x-thin',
                         marker={'color': 'grey', 'size': 2.5, 'line': dict(width=0.5)},
                         name='Resonator frequencies'),
              go.Scatter(x=[w_max, w_max + 1], y=[w_max, w_max + 1], mode='markers', marker={'color': 'black'},
                         name='Rotor frequencies'),
              ], )
    fig.update_layout(title={'xanchor': 'center',
                             'x': 0.4,
                             'font': {'family': 'Arial, bold',
                                      'size': 15},
                             'text': f'Campbell and Response heatmap'},
                      yaxis={'range': [w_min, w_max],
                             'dtick': 50,
                             "gridcolor": "rgb(159, 197, 232)",
                             "zerolinecolor": "rgb(74, 134, 232)"},
                      xaxis={'range': [w_min, sp_arr[-1]],
                             'dtick': 50,
                             "gridcolor": "rgb(159, 197, 232)",
                             "zerolinecolor": "rgb(74, 134, 232)"},
                      xaxis_title='Speed (rad/s)',
                      yaxis_title='Natural frequencies (rad/s)',
                      font=dict(family="Calibri, bold",
                                size=15),
                      legend=dict(xanchor='center', x=0.5, yanchor='bottom',
                                  y=1, orientation='h'))
    return fig


def plot_frf(r, sp_arr):
    
    rsolo = r[0]
    r_det = r[1]
    r_var = r[2]

    max_y = np.log10(max([np.max(rsolo),np.max(r_det),np.max(r_var)]))
    max_y = max_y + 0.1*np.abs(max_y)
    min_y = np.log10(min([np.min(rsolo),np.min(r_det),np.min(r_var)]))
    min_y = min_y - 0.1*np.abs(min_y)
    fig = go.Figure(data=[go.Scatter(x=sp_arr,y=np.log10(rsolo[:,0]),name='Solo rotor'),
                          go.Scatter(x=sp_arr,y=np.log10(r_det[:,0]),name='Ressonators'),
                          go.Scatter(x=sp_arr,y=np.log10(r_var[:,0]),name='Ressonators + variability'),
                          go.Scatter(x=[375,375],y=[min_y,max_y],
                                    mode='lines',line={'color':'black','dash':'dash','width':1},name='Target frequency'),
                          # go.Scatter(x=[f_1,f_1],y=[min_y,max_y],
                          #            mode='lines',line={'color':'red','dash':'dash','width':1},name='f1'),
                          ])
    fig.update_layout(title={'xanchor': 'center',
                             'x':0.4,
                             'font':{'family':'Arial, bold',
                                     'size': 15},
                             # 'text':f'Forward excitation - m_ratio:{m_ratio:.3f} - var: {var}'
                             },
                      yaxis={'range':[min_y,max_y],
                             "gridcolor": "rgb(159, 197, 232)",
                             "zerolinecolor": "rgb(74, 134, 232)"},
                      xaxis={'range':[0,sp_arr[-1]],
                             'dtick':50,
                             "gridcolor": "rgb(159, 197, 232)",
                             "zerolinecolor": "rgb(74, 134, 232)"},
                      xaxis_title='Frequency (rad/s)',                  
                      yaxis_title='$log_{10}(M_{ax})$',
                      font=dict(family="Calibri, bold",
                            size=15))
    return fig

def res_diff(x, n_pos, cross_prod=False, energy=False):

    x1 = x[n_pos]

    if cross_prod:
        x2 = x[len(x) - len(n_pos):]
        dx = x2 - x1
        diff = np.real(x1) * np.imag(dx) - np.real(x2) * np.imag(dx)
    elif energy:
        x2 = x[len(x) // 2 - len(n_pos):len(x) // 2]
        dx = x2 - x1
        v1 = x[len(x) // 2 + n_pos]
        diff = np.real(dx) * np.real(v1) + np.imag(dx) * np.imag(v1)
    else:
        x2 = x[len(x) - len(n_pos):]
        diff = x2 / x1
        diff = (np.real(diff)) + 1.j * np.abs(np.imag(diff))
        diff = np.delete(diff, [7])

    return diff


def plot_deflected_shape(rotor,y,n_pos,dof,plot_orbits=None,ys=None):
    
    if dof == 'trans':
        dof = 0
        un = 'm'
    elif dof == 'flex':
        dof = 2
        un = 'rad'
    
    N2 = len(n_pos)
    N1 = rotor.ndof//4
    
    l = rotor.nodes_pos
    N = len(l)
    
    if plot_orbits == None:
        plot_orbits = [0,int(N1/2),N1-1]
        
    data_res = [go.Scatter3d(x=[rotor.nodes_pos[i] for i in n_pos],
                             z=np.real(y[dof+4*N1+1::4]).reshape((N2)),
                             y=np.real(y[dof+4*N1::4]).reshape((N2)),
                             mode='markers',marker={'size':2,'color':'red'},
                             legendgroup='res',name='Resonators')]
    
    for i in range(N2):
        data_res.append(go.Scatter3d(x=np.ones(2)*rotor.nodes_pos[n_pos[i]],
                                     z=np.linspace(np.real(y[dof+4*n_pos[i]+1]),np.real(y[dof+4*N1+4*i+1]),2).reshape((2)),
                                     y=np.linspace(np.real(y[dof+4*n_pos[i]]),np.real(y[dof+4*N1+4*i]),2).reshape((2)),
                                     mode='lines',line={'color':'red','width':2},
                                     legendgroup='res',showlegend=False))
    
    data_orbits = []
    sl1 = True
    sl2 = True
    t = np.arange(0,2*np.pi*0.95,np.pi/15)
    for i, p in enumerate(plot_orbits):
                    
        data_orbits.append(go.Scatter3d(x=[rotor.nodes_pos[p]]*len(t),
                                        z=np.abs(y[dof+4*p+1])*np.cos(t+np.angle(y[dof+4*p+1])),
                                        y=np.abs(y[dof+4*p])*np.cos(t+np.angle(y[dof+4*p])),
                                        mode = 'lines',
                                        showlegend=sl1,
                                        line={'color':'lightblue',
                                              'width':2,},
                                        name='Rotor Orbits',
                                        legendgroup='rot-orb'))
        sl1 = False
            
        if p in n_pos:
            p2 = N1 + np.argmin([np.abs(a-p) for a in n_pos])
            data_orbits.append(go.Scatter3d(x=[rotor.nodes_pos[n_pos[p2-N1]]]*len(t),
                                            z=np.abs(y[dof+4*p2+1])*np.cos(t+np.angle(y[dof+4*p2+1])),
                                            y=np.abs(y[dof+4*p2])*np.cos(t+np.angle(y[dof+4*p2])),
                                            mode = 'lines',
                                            showlegend=sl2,
                                            line={'color':'magenta',
                                                  'width':2,},
                                            name='Resonator Orbits',
                                            legendgroup='res-orb'))
            sl2 = False
    
    data_solo = []
    max_ys = 0
    if type(ys) != type(None):
        max_ys = max(np.max(np.abs(ys[dof:N1*4:4])),np.max(np.abs(ys[dof+1:N1*4:4])))
        data_solo.append(go.Scatter3d(x=l,z=np.real(ys[dof+1:4*N1:4]).reshape((N1)),
                                       y=np.real(ys[dof:4*N1:4]).reshape((N1)),
                                       mode='lines',line={'width':2,'color':'black'},name='Rotor solo'),)
    
    
    fig = go.Figure(data=[go.Scatter3d(x=l,z=np.real(y[dof+1:4*N1:4]).reshape((N1)),
                                       y=np.real(y[dof:4*N1:4]).reshape((N1)),mode='lines',line={'width':5},name='Deflected shape'),
                          go.Scatter3d(x=l,z=[0]*len(l),y=[0]*len(l),name='Neutral line',showlegend=False,
                                       mode='lines',line={'width':1,'color':'black','dash':'dash'}),                          
                          ]+data_orbits+data_res+data_solo)
    max_x = max((np.max(np.abs(y[dof::4])),np.max(np.abs(y[dof+1::4])),max_ys))
    
    
    fig.update_layout(scene=dict(yaxis={'range':[2*-max_x,2*max_x],
                                        'title':f'X [{un}]'},
                                 zaxis={'range':[2*-max_x,2*max_x],
                                        'title':f'Y [{un}]'},
                                 xaxis={'title':'Axial position [m]'},))
    
    return fig

