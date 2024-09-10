import plotly.graph_objs as go
import numpy as np
from numpy import linalg as la

class IntegrationResults():

    def __init__(self,
                 frequency_list,
                 data_dict_list,
                 system):

        self.frequency_list = frequency_list
        self.fl = self.frequency_list

        self.data_dict_list = data_dict_list
        self.ddl = self.data_dict_list

        if len(self.fl) != len(self.ddl):
            print('WARNING: length of frequency and data dict lists do not match.')

        self.system = system

    @classmethod
    def update_class_object(cls, obj):

        return cls(frequency_list=obj.fl,
                 data_dict_list=obj.ddl,
                 system=obj.system)

    @staticmethod
    def poincare_section(x,
                         t,
                         omg,
                         n_points=10,
                         cut=1):

        dt = t[1] - t[0]
        T = 2 * np.pi / omg
        N = int(np.ceil(T / dt))
        dt2 = T / N
        t0 = (t[-1] - t[0]) / cut
        t0 = np.round(t0 / T) * T
        t2 = np.arange(t0, t[-1], dt2)

        if len(x.shape) == 1:
            x2 = np.interp(t2, t, x)
            pc = x2[::N]
            pc = pc[-n_points:]
        else:
            x2 = np.zeros((x.shape[0], len(t2)))
            for i in range(len(x[:, 1])):
                x2[i, :] = np.interp(t2, t, x[i, :])
            pc = x2[:, ::N]
            pc = pc[:, -n_points:]

        return pc

    def _find_frequency_index(self, f):

        a = np.array(self.fl) - f
        i = np.argmin(abs(a))

        return i

    def _adjust_plot(self, fig):

        fig.update_layout(width=800,
                          height=700,
                          font=dict(family="Calibri, bold",
                                    size=18),
                          yaxis={"gridcolor": "rgb(159, 197, 232)",
                                "zerolinecolor": "rgb(74, 134, 232)",
                                },
                          xaxis={"gridcolor": "rgb(159, 197, 232)",
                                 "zerolinecolor": "rgb(74, 134, 232)"},
                          title={'xanchor': 'center',
                                 'x': 0.4,
                                 'font': {'family': 'Calibri, bold',
                                          'size': 25},
                                 },
                          )

        return fig

    def _adjust_plot3d(self, fig):

        fig.update_layout(width=1200,
                          height=1000,
                          font=dict(family="Calibri, bold",
                                    size=16),
                          yaxis={"gridcolor": "rgb(159, 197, 232)",
                                "zerolinecolor": "rgb(74, 134, 232)",
                                },
                          xaxis={"gridcolor": "rgb(159, 197, 232)",
                                 "zerolinecolor": "rgb(74, 134, 232)"},
                          title={'xanchor': 'center',
                                 'x': 0.4,
                                 'font': {'family': 'Calibri, bold',
                                          'size': 25},
                                 },
                          )

        return fig

    def _calc_amplitude(self,
                        x,
                        cut=2,
                        amplitude_units='rms'):

        if amplitude_units == 'rms':
            n_cut = int(len(x) / cut)
            x_cut = x[-n_cut:]

            amp = np.sqrt(
               np.sum((x_cut - np.mean(x_cut)) ** 2) / n_cut
            )

        elif amplitude_units == 'max_displacement':

            n_cut = int(len(x[0]) / cut)
            d_cut = np.sqrt(
                (
                        x[0][-n_cut:] - np.mean(x[0][-n_cut:])
                ) ** 2 + (
                        x[1][-n_cut:] - np.mean(x[1][-n_cut:])
                ) ** 2
            )

            amp = np.max(d_cut)

        elif amplitude_units == 'pk':

            n_cut = int(len(x) / cut)
            x_cut = x[-n_cut:]

            amp = np.max(np.abs(x_cut - np.mean(x_cut)))

        elif amplitude_units == 'pk-pk':

            n_cut = int(len(x) / cut)
            x_cut = x[-n_cut:]

            amp = np.max(x_cut) - np.min(x_cut)


        return amp

    def _calc_full_spectrum(self,
                            t,
                            x,
                            y,
                            cut=2,
                            hanning=True,
                            synch_freq=None,
                            return_complex=False,
                            ):

        w, spec_x = self._calc_fourier(t=t,
                                       x=x,
                                       cut=cut,
                                       hanning=hanning,
                                       synch_freq=synch_freq,
                                       return_complex=True)
        w, spec_y = self._calc_fourier(t=t,
                                       x=y,
                                       cut=cut,
                                       hanning=hanning,
                                       synch_freq=synch_freq,
                                       return_complex=True)

        fow = 1 / 2 * (spec_x + 1.j * spec_y)
        back = 1 / 2 * (np.conj(spec_x) + 1.j * np.conj(spec_y))

        w = np.concatenate([- w[-1::-1], np.zeros(1), w])
        spectrum = np.concatenate([back[-1::-1], np.zeros(1), fow])

        if not return_complex:
            spectrum = np.abs(spectrum)

        return w, spectrum

    def _calc_fourier(self,
                      t,
                      x,
                      cut=2,
                      hanning=True,
                      synch_freq=None,
                      return_complex=False
                      ):

        n_cut = int(len(x) / cut)
        dt = t[1] - t[0]

        if synch_freq:
            T = 2 * np.pi / synch_freq
            n_per = np.round((dt * n_cut) / T)
            n_cut = int(np.round(n_per * T / dt))

        x_cut = x[-n_cut:]

        t_cut = t[:n_cut]

        dw = 1 / (t_cut[-1])
        w_max = 1 / (2 * dt)
        w = np.arange(dw, w_max, dw) * 2 * np.pi

        if hanning:
            win = np.hanning(n_cut)
        else:
            win = np.ones(n_cut)

        spectrum = 2 / (np.sum(win)) * np.fft.fft(win * x_cut)
        if not return_complex:
            spectrum = np.abs(spectrum)

        spectrum = spectrum[1:]
        spectrum = spectrum[:len(w)]

        return w, spectrum

    def plot_bifurcation_diagram(self,
                                 dof=None,
                                 cut=2,
                                 n_points=20):

        fig = go.Figure()
        if dof is None:
            dof = [j for j in self.ddl[0].keys() if j != 'time']

        omg_bif = []
        bif_data_dict = {}
        for k in dof:
            bif_data_dict[k] = []
        for i, f in enumerate(self.fl):

            d = self.ddl[i]

            for k in dof:
                data = self.poincare_section(x=d[k],
                                             t=d['time'],
                                             omg=f,
                                             n_points = n_points,
                                             cut=cut
                                             )

                bif_data_dict[k] += list(data)

            omg_bif += [f] * len(data)

        for k in dof:
            fig.add_trace(go.Scatter(x=omg_bif,
                                     y=bif_data_dict[k],
                                     mode='markers',
                                     name=k,
                                     )
                          )
        fig.update_yaxes(title='Displacement [m]')
        fig.update_xaxes(title='Excitation Frequency [rad/s]',
                         range=[np.min(self.fl), np.max(self.fl)])
        fig = self._adjust_plot(fig)

        return fig

    def plot_frf(self,
                 dof=None,
                 cut=2,
                 amplitude_units='rms'):

        fig = go.Figure()
        if dof is None:
            dof = [j for j in self.ddl[0].keys() if j != 'time']

        rms = np.zeros((len(dof), len(self.fl)))

        for i, f in enumerate(self.fl):

            d = self.ddl[i]

            if amplitude_units == 'max_displacement':
                d_aux = [(d[k[0]], d[k[1]]) for k in dof]
                rms[:, i] = np.array(
                    [self._calc_amplitude(k,
                                          cut=cut,
                                          amplitude_units=amplitude_units) for k in d_aux])

            else:
                rms[:, i] = np.array(
                    [self._calc_amplitude(d[k],
                                          cut=cut,
                                          amplitude_units=amplitude_units) for k in dof])

        for i, p in enumerate(dof):
            fig.add_trace(go.Scatter(x=self.fl, y=rms[i, :], name=f'DoF: {p}'))

        fig.update_layout(
                          xaxis={'range': [0, np.max(self.fl)],
                                 },
                          xaxis_title='Frequency [rad/s]',
                          yaxis_title='Amplitude [m RMS]',
                          )
        fig.update_yaxes(type="log")
        fig = self._adjust_plot(fig)

        return fig


    def plot_waveform(self,
                      frequency,
                      dof=None):

        fig = go.Figure()

        i = self._find_frequency_index(frequency)
        d = self.ddl[i]
        t = d['time']
        k = list(d.keys())

        if dof is None:
            dof = [j for j in k if j != 'time']

        for j in dof:
            fig.add_trace(go.Scatter(x=t,
                                     y=d[j],
                                     name=j))

        fig.update_layout(title=f'Frequency: {self.fl[i]:.2f}',
                          xaxis_title='Time [s]',
                          yaxis_title='Amplitude [m]',
                          )
        fig = self._adjust_plot(fig)

        return fig

    def plot_orbit(self,
                   frequency,
                   dof=None,
                   cut=1):

        fig = go.Figure()

        i = self._find_frequency_index(frequency)
        d = self.ddl[i]
        t = d['time']
        k = list(d.keys())

        if dof is None:
            dof = [(k[0], k[1])]

        n_cut = int(len(d[dof[0][0]]) / cut)
        t_cut = t[:n_cut]

        T = 2 * np.pi / self.fl[i]
        t_pc = np.arange(0, t_cut[-1], T)

        max_amp = 0
        for i in dof:
            fig.add_trace(go.Scatter(x=d[i[0]][-n_cut:],
                                     y=d[i[1]][-n_cut:],
                                     name=f'{i[0]} vs {i[1]}',
                                     mode='lines',
                                     legendgroup=f'{i[0]}')
                          )

            fig.add_trace(go.Scatter(x=np.interp(t_pc, t_cut, d[i[0]][-n_cut:]),
                                     y=np.interp(t_pc, t_cut, d[i[1]][-n_cut:]),
                                     mode='markers',
                                     marker=dict(color='red'),
                                     name='Poincaré Section',
                                     showlegend=False,
                                     legendgroup=f'{i[0]}'
                                     )
                          )

            if np.max(np.abs(d[i[0]])) > max_amp:
                max_amp = np.max(np.abs(d[i[0]]))
            if np.max(np.abs(d[i[1]])) > max_amp:
                max_amp = np.max(np.abs(d[i[1]]))

        fig.update_layout(title=f'Orbit Plot',
                          xaxis_range=[- 1.1 * max_amp, 1.1 * max_amp],
                          yaxis_range=[- 1.1 * max_amp, 1.1 * max_amp],
                          xaxis_title=f'X [m]',
                          yaxis_title=f'Y [m]',
                          )
        fig = self._adjust_plot(fig)

        return fig


    # def plot_state_space(self,
    #                      dof,
    #                      frequency):
    #

    def plot_spectrum(self,
                      dof,
                      frequency,
                      full_spectrum=False,
                      hanning=True,
                      cut=2,
                      max_frequency=None):

        fig = go.Figure()

        i = self._find_frequency_index(frequency)
        d = self.ddl[i]
        t = d['time']

        if full_spectrum and len(dof) != 2:
            print('WARNING: Inform X and Y DOFs for full spectrum plot.')

        if full_spectrum:
            w, sp = self._calc_full_spectrum(t=t,
                                             x=d[dof[0]],
                                             y=d[dof[1]],
                                             cut=cut,
                                             hanning=hanning,
                                             synch_freq=self.fl[i])
        else:
            w, sp = self._calc_fourier(t=t,
                                       x=d[dof],
                                       cut=cut,
                                       hanning=hanning,
                                       synch_freq=self.fl[i])

        if max_frequency is None:
            max_frequency = np.max(w)

        fig.add_trace(go.Scatter(x=w,
                                 y=sp,
                                 name=str(dof)))
        fig.add_trace(go.Scatter(x=[self.fl[i]] * 2,
                                 y=[0, 1.1 * np.max(sp)],
                                 mode='lines',
                                 line=dict(dash='dash',
                                           color='black',
                                           width=1),
                                 legendgroup='synch',
                                 name='Synch. Freq.'))
        if full_spectrum:
            fig.add_trace(go.Scatter(x=[- self.fl[i]] * 2,
                                     y=[0, 1.1 * np.max(sp)],
                                     mode='lines',
                                     line=dict(dash='dash',
                                               color='black',
                                               width=1),
                                     legendgroup='synch',
                                     showlegend=False,
                                     name='Synch. Freq.'))
        if full_spectrum:
            freq_range = [- max_frequency, max_frequency]
        else:
            freq_range = [0, max_frequency]

        fig.update_layout(title=f'Frequency: {self.fl[i]:.2f}',
                          xaxis_range=freq_range,
                          yaxis_range=[0, 1.1 * np.max(sp)],
                          xaxis_title='Frequency [rad/s]',
                          yaxis_title='Amplitude [m]',
                          )
        fig = self._adjust_plot(fig)

        return fig

    def plot_spectrum_heatmap(self,
                              dof,
                              max_freq=None,
                              max_amp=None,
                              full_spectrum=False,
                              hanning=True,
                              cut=2):

        fig = go.Figure()
        t = self.ddl[0]['time']

        amp = None

        for i, f in enumerate(self.fl):
            d = self.ddl[i]
            if full_spectrum:
                w, aux = self._calc_full_spectrum(t=t,
                                                  x=np.interp(t, d['time'], d[dof[0]]),
                                                  y=np.interp(t, d['time'], d[dof[1]]),
                                                  hanning=hanning,
                                                  cut=cut
                                                  )
            else:
                w, aux = self._calc_fourier(t=t,
                                            x=np.interp(t, d['time'], d[dof]),
                                            hanning=hanning,
                                            cut=cut
                                            )
            if amp is None:
                amp = np.zeros((len(self.fl), len(aux)))

            amp[i, :] = aux

        if max_freq is None:
            max_freq = np.max(w)

        fig.add_trace(go.Heatmap(y=self.fl,
                                 x=w,
                                 z=np.log10(amp),
                                 colorbar=dict(title='Resp. amp. [log]')
                                 )
                      )

        fig.add_trace(go.Scatter(x=[0, np.max(self.fl)],
                                 y=[0, np.max(self.fl)],
                                 mode='lines',
                                 name='Synchronous line',
                                 line=dict(dash='dash',
                                           color='black',
                                           width=1),
                                 legendgroup='synch',
                                 showlegend=True
                                 )
                      )
        if full_spectrum:
            fig.add_trace(go.Scatter(x=[0, - np.max(self.fl)],
                                     y=[0, np.max(self.fl)],
                                     mode='lines',
                                     name='Synchronous line',
                                     line=dict(dash='dash',
                                               color='black',
                                               width=1),
                                     legendgroup='synch',
                                     showlegend=False
                                     )
                          )

        fig = self._adjust_plot(fig)

        if full_spectrum:
            freq_range = [- max_freq, max_freq]
        else:
            freq_range = [0, max_freq]

        fig.update_layout(
            xaxis_range=freq_range,
            yaxis_range=[np.min(self.fl), np.max(self.fl)],
            xaxis_title='Response Frequency [rad/s]',
            yaxis_title='Rotating Speed [rad/s]',
            legend=dict(orientation='h',
                        xanchor='center',
                        x=0.5,
                        yanchor='bottom',
                        y=1.05),
            width=900,
            height=700
        )

        return fig

    def plot_cascade3d(self,
                       dof,
                       max_freq=None,
                       max_amp=None,
                       full_spectrum=False,
                       hanning=True,
                       log_amplitude=False,
                       cut=2):

        fig = go.Figure()
        t = self.ddl[0]['time']

        amp = None

        for i, f in enumerate(self.fl):
            d = self.ddl[i]
            if full_spectrum:
                w, aux = self._calc_full_spectrum(t=t,
                                                  x=np.interp(t, d['time'], d[dof[0]]),
                                                  y=np.interp(t, d['time'], d[dof[1]]),
                                                  hanning=hanning,
                                                  cut=cut
                                                  )
            else:
                w, aux = self._calc_fourier(t=t,
                                            x=np.interp(t, d['time'], d[dof]),
                                            hanning=hanning,
                                            cut=cut
                                            )

            if log_amplitude:
                aux = np.log10(aux)

            fig.add_trace(go.Scatter3d(x=w,
                                       y=[f] * len(w),
                                       z=aux,
                                       showlegend=False,
                                       mode='lines',
                                       line=dict(color='blue')))

        if max_freq is None:
            max_freq = np.max(w)

        fig = self._adjust_plot3d(fig)

        if full_spectrum:
            freq_range = [- max_freq, max_freq]
        else:
            freq_range = [0, max_freq]

        fig.update_layout(
            scene=dict(
                xaxis=dict(range=freq_range,
                           title='Response Frequency [rad/s]'),
                yaxis=dict(range=[np.min(self.fl), np.max(self.fl)],
                           title='Rotating Speed [rad/s]'),
                zaxis=dict(title='Amplitude [m]'),
            ),
        )

        return fig

    def _calc_diff(self,
                   t,
                   x,
                   y=None,
                   cut=2,
                   hanning=True,
                   synch_freq=None):

        if y is not None:
            w, spec_1 = self._calc_full_spectrum(t=t,
                                             x=x[0],
                                             y=y[0],
                                             cut=cut,
                                             hanning=hanning,
                                             synch_freq=synch_freq,
                                             return_complex=True)

            _, spec_2 = self._calc_full_spectrum(t=t,
                                                 x=x[1],
                                                 y=y[1],
                                                 cut=cut,
                                                 hanning=hanning,
                                                 synch_freq=synch_freq,
                                                 return_complex=True)
        else:
            w, spec_1 = self._calc_fourier(t=t,
                                        x=x[0],
                                        cut=cut,
                                        hanning=hanning,
                                        synch_freq=synch_freq,
                                        return_complex=True)

            _, spec_2 = self._calc_fourier(t=t,
                                        x=x[1],
                                        cut=cut,
                                        hanning=hanning,
                                        synch_freq=synch_freq,
                                        return_complex=True)

        diff = spec_1 / spec_2
        # diff = np.zeros(spec_1.shape).astype(complex)
        # diff[spec_2.nonzero()] = spec_1[spec_2.nonzero()] / spec_2[spec_2.nonzero()]

        return w, diff

    def plot_diff_map(self,
                      dof,
                      dof_y=None,
                      mode='amp',
                      max_freq=None,
                      max_amp=None,
                      full_spectrum=False,
                      hanning=False,
                      cut=2,
                      log_mode=False):

        fig = go.Figure()
        t = self.ddl[0]['time']

        z = None

        for i, f in enumerate(self.fl):
            d = self.ddl[i]

            aux = 0
            for j, dof_i in enumerate(dof):
                if full_spectrum:
                    w, aux2 = self._calc_diff(t=t,
                                              x=(
                                                  np.interp(t, d['time'], d[dof_i[0]]),
                                                  np.interp(t, d['time'], d[dof_i[1]])
                                              ),
                                              y=(
                                                  np.interp(t, d['time'], d[dof_y[j][0]]),
                                                  np.interp(t, d['time'], d[dof_y[j][1]])
                                              ),
                                              hanning=hanning,
                                              cut=cut
                                              )
                else:
                    w, aux2 = self._calc_diff(t=t,
                                             x=(
                                                 np.interp(t, d['time'], d[dof_i[0]]),
                                                 np.interp(t, d['time'], d[dof_i[1]])
                                             ),
                                             hanning=hanning,
                                             cut=cut
                                             )

                aux += aux2

            aux = aux / len(dof)
                # if aux is None:
                #     aux = aux2
                # else:
                #     aux = (aux + aux2) / 2


            if z is None:
                z = np.zeros((len(self.fl), len(aux))).astype(complex)

            z[i, :] = aux

        if max_freq is None:
            max_freq = np.max(w)

        if full_spectrum:
            freq_range = [- max_freq, max_freq]
        else:
            freq_range = [0, max_freq]

        if mode == 'amp':
            if log_mode:
                z = np.log10(np.abs(z))
                zmin = np.min(z)
                zmax = 10
            else:
                z = np.abs(z)
                zmin = np.min(z)
                zmax = np.max(z)

            colorbar = dict(title='Amplification [log]')
            colorscale = 'Plasma'
            # for i, f in enumerate(self.fl):
            #     fig.add_trace(go.Scatter3d(x=w,
            #                                y=[f] * len(w),
            #                                z=z[i, :],
            #                                showlegend=False,
            #                                mode='lines',
            #                                line=dict(color='blue')))
            #
            # fig = self._adjust_plot3d(fig)
            #
            # fig.update_layout(
            #     scene=dict(
            #         xaxis=dict(range=freq_range,
            #                    title='Response Frequency [rad/s]'),
            #         yaxis=dict(range=[np.min(self.fl), np.max(self.fl)],
            #                    title='Rotating Speed [rad/s]'),
            #         zaxis=dict(title='Amplification [m]'),
            #     ),
            # )

        elif mode == 'angle':
            z = np.angle(z) * 180 / (np.pi)
            zmin = 0
            zmax = 360
            # z[-1, -1] = 0
            z[z < 0] += 360
            colorbar = dict(title='Phase angle [deg]',
                            tickvals=[0, 90, 180, 270, 360],
                            ticktext=["0", "90", "180", "270", "360"],)
            colorscale = 'Phase'
        # else:
        #     print('WARNING: mode must be either amp or angle. Plot will show amplification.')

        fig.add_trace(go.Heatmap(y=self.fl,
                                 x=w,
                                 z=z,
                                 colorbar=colorbar,
                                 colorscale=colorscale,
                                 zmin=zmin,
                                 zmax=zmax
                                 )
                      )

        fig.add_trace(go.Scatter(x=[0, np.max(self.fl)],
                                 y=[0, np.max(self.fl)],
                                 mode='lines',
                                 name='Synchronous line',
                                 legendgroup='synch',
                                 line=dict(dash='dot',
                                           color='black',
                                           width=1),
                                 showlegend=True
                                 )
                      )
        if full_spectrum:
            fig.add_trace(go.Scatter(x=[0, - np.max(self.fl)],
                                     y=[0, np.max(self.fl)],
                                     mode='lines',
                                     name='Synchronous line',
                                     legendgroup='synch',
                                     line=dict(dash='dot',
                                               color='black',
                                               width=1),
                                     showlegend=False
                                     )
                          )

        fig = self._adjust_plot(fig)

        fig.update_layout(
            xaxis_range=freq_range,
            yaxis_range=[np.min(self.fl), np.max(self.fl)],
            xaxis_title='Response Frequency [rad/s]',
            yaxis_title='Rotating Speed [rad/s]',
            legend=dict(orientation='h',
                        xanchor='center',
                        x=0.5,
                        yanchor='bottom',
                        y=1.05),
            width=900,
            height=700
        )

        return fig