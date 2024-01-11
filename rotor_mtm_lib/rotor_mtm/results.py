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

    def _calc_rms(self, x, cut=2):

        n_cut = int(len(x) / cut)
        x_cut = x[-n_cut:]

        rms = np.sqrt(
           np.sum((x_cut - np.mean(x_cut)) ** 2) / n_cut
        )

        return rms

    def _calc_fourier(self,
                      t,
                      x,
                      cut=2,
                      hanning=True,
                      synch_freq=None,
                      return_complex=False):

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

        return w, spectrum

    def plot_frf(self,
                 dof=None,
                 cut=2):

        fig = go.Figure()
        if dof is None:
            dof = [j for j in self.ddl[0].keys() if j != 'time']

        rms = np.zeros((len(dof), len(self.fl)))

        for i, f in enumerate(self.fl):

            d = self.ddl[i]

            rms[:, i] = np.array(
                [self._calc_rms(d[k],
                                cut=cut) for k in dof])

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

            if np.max(d[i[0]]) > max_amp:
                max_amp = np.max(d[i[0]])
            if np.max(d[i[1]]) > max_amp:
                max_amp = np.max(d[i[1]])

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

        w, sp = self._calc_fourier(t=t,
                                   x=d[dof],
                                   cut=cut,
                                   hanning=hanning,
                                   synch_freq=self.fl[i])

        if max_frequency is None:
            max_frequency = np.max(w)

        fig.add_trace(go.Scatter(x=w,
                                 y=sp,
                                 name=dof))
        fig.add_trace(go.Scatter(x=[self.fl[i]] * 2,
                                 y=[0, 1.1 * np.max(sp)],
                                 mode='lines',
                                 line=dict(dash='dash',
                                           color='black',
                                           width=1),
                                 name='Synch. Freq.'))

        fig.update_layout(title=f'Frequency: {self.fl[i]:.2f}',
                          xaxis_range=[0, max_frequency],
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

        fig.add_trace(go.Heatmap(y=w,
                                 x=self.fl,
                                 z=np.log10(amp.transpose()),
                                 colorbar=dict(title='Resp. amp. [log]')
                                 )
                      )

        fig.add_trace(go.Scatter(x=[0, np.max(self.fl)],
                                 y=[0, np.max(self.fl)],
                                 mode='lines',
                                 name='Synchronous line',
                                 line=dict(dash='dot',
                                           color='black',
                                           width=1),
                                 showlegend=True
                                 )
                      )

        fig = self._adjust_plot(fig)
        fig.update_layout(
            xaxis_range=[np.min(self.fl), np.max(self.fl)],
            yaxis_range=[0, max_freq],
            xaxis_title='Excitation frequency [rad/s]',
            yaxis_title='Response frequency [rad/s]',
            legend=dict(orientation='h',
                        xanchor='center',
                        x=0.5,
                        yanchor='bottom',
                        y=1.05),
            width=900,
            height=700
        )

        return fig

    def _calc_diff(self,
                   t,
                   x,
                   cut=2,
                   hanning=True,
                   synch_freq=None):

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

        return w, diff

    def plot_diff_heatmap(self,
                          dof,
                          mode='amp',
                          max_freq=None,
                          max_amp=None,
                          full_spectrum=False,
                          hanning=False,
                          cut=2
                          ):

        fig = go.Figure()
        t = self.ddl[0]['time']

        z = None

        for i, f in enumerate(self.fl):
            d = self.ddl[i]

            w, aux = self._calc_diff(t=t,
                                     x=(
                                         np.interp(t, d['time'], d[dof[0]]),
                                         np.interp(t, d['time'], d[dof[1]])
                                     ),
                                     hanning=hanning,
                                     cut=cut
                                     )
            if z is None:
                z = np.zeros((len(self.fl), len(aux))).astype(complex)

            z[i, :] = aux

        if mode == 'amp':
            z = np.log10(np.abs(z))
            colorbar = dict(title='Amplification [log]')
        elif mode == 'angle':
            z = (np.angle(z)) * 180 / (2 * np.pi)
            z[-1, -1] = 0
            z += 2 * np.pi
            colorbar = dict(title='Phase angle [deg]')
        else:
            print('WARNING: mode must be either amp or angle. Plot will show amplification.')

        if max_freq is None:
            max_freq = np.max(w)

        fig.add_trace(go.Heatmap(y=w,
                                 x=self.fl,
                                 z=z.transpose(),
                                 colorbar=colorbar,
                                 # zmin=-1,
                                 # zmax=1
                                 )
                      )

        fig.add_trace(go.Scatter(x=[0, np.max(self.fl)],
                                 y=[0, np.max(self.fl)],
                                 mode='lines',
                                 name='Synchronous line',
                                 line=dict(dash='dot',
                                           color='black',
                                           width=1),
                                 showlegend=True
                                 )
                      )

        fig = self._adjust_plot(fig)
        fig.update_layout(
            xaxis_range=[np.min(self.fl), np.max(self.fl)],
            yaxis_range=[0, max_freq],
            xaxis_title='Excitation frequency [rad/s]',
            yaxis_title='Response frequency [rad/s]',
            legend=dict(orientation='h',
                        xanchor='center',
                        x=0.5,
                        yanchor='bottom',
                        y=1.05),
            width=900,
            height=700
        )

        return fig