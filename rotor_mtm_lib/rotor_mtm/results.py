import plotly.graph_objs as go
import numpy as np
from numpy import linalg as la

class IntegrationResults():

    def __init__(self,
                 frequency_list,
                 data_dict_list,
                 system=None,
                 probe_dof=None,
                 linear_results=None,
                 rigid_results=None):
        """Class for storing the results of the integration process.
        Parameters
        ----------
        frequency_list : list
            A list of frequencies at which the results were computed.
        data_dict_list : list
            A list of dictionaries containing the results for each frequency.
        system : Sys_NL, optional
            The system object used for the integration.
        probe_dof : list, optional
            A list of degrees of freedom (DoF) that were probed during the integration.
        linear_results : dict, optional
            A dictionary containing the results of the linear system.
        rigid_results : dict, optional
            A dictionary containing the results of the rigid body system.
        Attributes
        ----------
        frequency_list : list
            A list of frequencies at which the results were computed.
        fl : list
            Alias for frequency_list.
        data_dict_list : list
            A list of dictionaries containing the results for each frequency.
        ddl : list
            Alias for data_dict_list.
        system : Sys_NL, optional
            The system object used for the integration.
        probe_dof : list, optional
            A list of degrees of freedom (DoF) that were probed during the integration.
        linear_results : dict, optional
            A dictionary containing the results of the linear system.
        rigid_results : dict, optional
            A dictionary containing the results of the rigid body system.
        """

        self.frequency_list = frequency_list
        self.fl = self.frequency_list

        self.data_dict_list = data_dict_list
        self.ddl = self.data_dict_list

        if len(self.fl) != len(self.ddl):
            print('WARNING: length of frequency and data dict lists do not match.')

        self.system = system
        self.probe_dof = probe_dof
        self.linear_results = linear_results
        self.rigid_results = rigid_results

    @classmethod
    def update_class_object(cls, 
                            obj):
        """Updates the class object with the attributes of another object.
        Parameters
        ----------
        obj : IntegrationResults
            An instance of IntegrationResults class to update the attributes from.
        Returns
        -------
        IntegrationResults
            A new instance of IntegrationResults class with updated attributes.
        """

        aux = cls(frequency_list=obj.fl,
                 data_dict_list=obj.ddl,
                 system=obj.system,
                 linear_results=obj.linear_results,
                 rigid_results=obj.rigid_results)
        
        for k in obj.__dict__:
            if k not in aux.__dict__:
                aux.__dict__[k] = obj.__dict__[k]
        
        return aux

    @staticmethod
    def poincare_section(x,
                         t,
                         omg,
                         n_points=10,
                         cut=1):
        """Computes the Poincaré section of a dynamical system.
        Parameters
        ----------
        x : array_like
            The state vector of the system.
        t : array_like
            The time vector corresponding to the state vector.
            This vector must have uniform time steps.
        omg : float
            The base frequency at which the Poincaré section is computed.
        n_points : int, optional
            The number of periods to include in the Poincaré section.
        cut : int, optional
            The factor by which to cut the time vector.
            The result will consider only the 1/cut last time steps.
            This is useful to remove residual transient responses from the results.
            Default is 1 (no cut).
        Returns
        -------
        array_like
            The Poincaré section of the system.
        Notes
        -----
        The Poincaré section is computed by interpolating the state vector at
        regular intervals determined by the frequency.
        The time vector is cut to reduce the number of points considered.
        The function returns the last n_points of the Poincaré section.
        """

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
        """Finds the index of the frequency in the frequency list.
        Parameters
        ----------
        f : float
            The frequency to find in the frequency list.
        Returns
        -------
        int
            The index of the frequency in the frequency list.
        """

        a = np.array(self.fl) - f
        i = np.argmin(abs(a))

        return i

    @staticmethod
    def _adjust_plot(fig,
                     font_size=30,
                     maintain_proportions=False,
                     colorbar_above=False):
        """
        Adjusts the layout of the Plotly figure to a standard layout.
        Parameters
        ----------
        fig : plotly.graph_objs.Figure
            The Plotly figure to adjust.
        font_size : int, optional
            The font size for the figure. Default is 30.
        maintain_proportions : bool, optional
            Whether to maintain the proportions of the figure. Default is False.
        colorbar_above : bool, optional
            Whether to place the colorbar above the heatmap. Default is False.
        Returns
        -------
        plotly.graph_objs.Figure
            The adjusted Plotly figure.
        """

        if not maintain_proportions:
            fig.update_layout(width=800,
                              height=700)


        fig.update_layout(font=dict(family="Calibri, bold",
                                    size=font_size),
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
        
        if fig.layout.yaxis.type == 'log':
            fig.layout.yaxis.tickvals = [10**i for i in range(-10, 4)]
        
        if len(fig.data) > 0 and isinstance(fig.data[0], go.Heatmap) and colorbar_above:
            fig.data[0].colorbar.orientation = 'h'
            fig.data[0].colorbar.yanchor = 'top'
            fig.data[0].colorbar.y = 1.25
            fig.layout.legend.yanchor = 'top'
            fig.layout.legend.y = 1.11
            fig.update_layout(width=800,
                              height=800)

        return fig

    def _adjust_plot3d(self, fig):
        """
        Adjusts the layout of the Plotly 3D figure to a standard layout.
        Parameters
        ----------
        fig : plotly.graph_objs.Figure
            The Plotly 3D figure to adjust.
        Returns
        -------
        plotly.graph_objs.Figure
            The adjusted Plotly 3D figure.
        """

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

    @staticmethod
    def _calc_amplitude(x,
                        cut=2,
                        amplitude_units='rms'):
        """Calculates the amplitude of a signal.
        Parameters
        ----------
        x : array_like
            The signal for which the amplitude is to be calculated.
        cut : int, optional
            The factor by which to cut the signal.
            The result will consider only the 1/cut last time steps.
            This should be used to remove residual transient responses from the results.
            Default is 2.
        amplitude_units : str, optional
            The units in which the amplitude is to be calculated.
            Can be one of {'rms', 'max_displacement', 'pk', 'pk-pk'}.
            Default is 'rms'.
        Returns
        -------
        float
            The calculated amplitude of the signal.
        Notes
        -----
        This function assumes that the input signal is one-dimensional.
        If the input signal is multi-dimensional, it will calculate the amplitude for each dimension separately.
        The amplitude is calculated based on the specified units:
        - 'rms': Root Mean Square of the last 1/cut part of the signal.
        - 'max_displacement': Maximum displacement of the last 1/cut part of the signal.
        - 'pk': Peak value of the last 1/cut part of the signal.
        - 'pk-pk': Peak-to-peak value of the last 1/cut part of the signal.
        """

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
                            backward_vector=False
                            ):
        """
        Calculates the full spectrum of a signal.
        Parameters
        ----------
        t : array_like
            The time vector corresponding to the signal.
        x : array_like
            The x-component of the signal.
        y : array_like
            The y-component of the signal.
        cut : int, optional
            The factor by which to cut the signal.
            The result will consider only the 1/cut last time steps.
            This should be used to remove residual transient responses from the results.
            Default is 2.
        hanning : bool, optional
            Whether to apply a Hanning window to the signal before calculating the spectrum.
            Default is True.
        synch_freq : float, optional
            The frequency at which the signal is synchronized.
            If provided, the spectrum will be calculated to minimize leakage at this
            frequency and its multiples.
            Default is None.
        return_complex : bool, optional
            Whether to return the complex spectrum.
            If True, the function will return the complex spectrum.
            If False, the function will return the absolute value of the spectrum.
            Default is False.
        backward_vector : bool, optional
            Whether to return the backward vector of the spectrum.
            If True, the function will return the backward vector.
            If False, the function will return the backward component.
            Default is False.
        Returns
        -------
        w : array_like
            The frequency vector corresponding to the spectrum.
        spectrum : array_like
            The calculated spectrum (complex or absolute value).
            The shape of the spectrum should matches the frequency vector w.
        
        """

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

        fow = (spec_x + 1.j * spec_y) / 2
        back = (spec_x - 1.j * spec_y) / 2
        if backward_vector:
            back = np.conj(back)

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
        """
        Calculates the Fourier transform of a signal.
        Parameters
        ----------
        t : array_like
            The time vector corresponding to the signal.
        x : array_like
            The signal to transform.
        cut : int, optional
            The factor by which to cut the signal.
            The result will consider only the 1/cut last time steps.
            This should be used to remove residual transient responses from the results.
            Default is 2.
        hanning : bool, optional
            Whether to apply a Hanning window to the signal before calculating the spectrum.
            Default is True.
        synch_freq : float, optional
            The frequency at which the signal is synchronized.
            If provided, the spectrum will be calculated to minimize leakage at this
            frequency and its multiples.
            Default is None.
        return_complex : bool, optional
            Whether to return the complex spectrum.
            If True, the function will return the complex spectrum.
            If False, the function will return the absolute value of the spectrum.
            Default is False.

        Returns
        -------
        w : array_like
            The frequency vector corresponding to the spectrum.
        spectrum : array_like
            The calculated spectrum (complex or absolute value).
            The shape of the spectrum should matches the frequency vector w.

        """

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
        """Plots the bifurcation diagram of the of the response.
        Parameters
        ----------
        dof : list, optional
            The degrees of freedom to include in the plot. DoFs names should be provided as strings.
            If None, all degrees of freedom will be included.
        cut : int, optional
            The factor by which to cut the signal.
            The result will consider only the 1/cut last time steps.
            This should be used to remove residual transient responses from the results.
            Default is 2.
        n_points : int, optional
            The number of points to use in the Poincaré section.
            Default is 20.

        Returns
        -------
        fig : plotly.graph_objs.Figure
            The resulting figure object.

        """

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
                                             n_points=n_points,
                                             cut=cut
                                             )

                bif_data_dict[k] += list(data)

            omg_bif += [f] * len(data)

        for k in dof:
            fig.add_trace(go.Scatter(x=omg_bif,
                                     y=bif_data_dict[k],
                                     mode='markers',
                                     marker=dict(size=4),
                                     name=k,
                                     )
                          )
        if len(fig.data) == 1:
            fig.data[0].marker.color = 'black'

        fig.update_yaxes(title='Displacement [m]')
        fig.update_xaxes(title='Excitation Frequency [rad/s]',
                         range=[np.min(self.fl), np.max(self.fl)])
        fig = self._adjust_plot(fig)

        return fig

    def plot_frf(self,
                 dof=None,
                 cut=2,
                 amplitude_units='rms',
                 frequency_units='rad/s'):
        """Plots the frequency response function (FRF) of the system.
        Parameters
        ----------
        dof : list, optional
            The degrees of freedom to include in the plot. DoFs names should be provided as strings.
            If None, all degrees of freedom will be included.
        cut : int, optional
            The factor by which to cut the signal.
            The result will consider only the 1/cut last time steps.
            This should be used to remove residual transient responses from the results.
            Default is 2.
        amplitude_units : str, optional
            The units to use for the amplitude. Can be 'rms' or 'max_displacement'.
            Default is 'rms'.
        frequency_units : str, optional
            The units to use for the frequency. Can be 'rad/s' or 'RPM'.
            Default is 'rad/s'.

        Returns
        -------
        fig : plotly.graph_objs.Figure
            The resulting figure object.

        """

        fig = go.Figure()
        if dof is None:
            dof = [j for j in self.ddl[0].keys() if (j != 'time' and j != 'solver')]

        rms = np.zeros((len(dof), len(self.fl)))

        freq_convert = 1
        if frequency_units.upper() == 'RPM':
            freq_convert = 60 / (2 * np.pi)

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
            fig.add_trace(go.Scatter(x=self.fl * freq_convert, y=rms[i, :], name=f'DoF: {p}'))

        fig.update_layout(
                          xaxis={'range': [0, np.max(self.fl) * freq_convert],
                                 },
                          xaxis_title=f'Frequency [{frequency_units}]',
                          yaxis_title=f'Amplitude [m {amplitude_units}]',
                          )
        fig.update_yaxes(type="log")
        fig = self._adjust_plot(fig)

        return fig


    def plot_waveform(self,
                      frequency,
                      dof=None):
        """Plots the waveform of the response at a given frequency.
        Parameters
        ----------
        frequency : float
            The frequency at which to plot the waveform.
        dof : list, optional
            The degrees of freedom to include in the plot. DoFs names should be provided as strings.
            If None, all degrees of freedom will be included.
        Returns
        -------
        fig : plotly.graph_objs.Figure
            The resulting figure object.    
        """

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
        """Plots the orbit of the response at a given frequency.
        Parameters
        ----------
        frequency : float
            The frequency at which to plot the orbit.
        dof : list, optional
            A list of tuples. Each tuple must contain the names of X and Y DoFs for the desired orbit.
            e.g. [("last_x", "last_y"), ("brg_x", "brg_y")]
            If None, the first two DoFs will be used as a X-Y pair.
        cut : int, optional
            The factor by which to cut the signal.
            The result will consider only the 1/cut last time steps.
            This should be used to remove residual transient responses from the results.
            Default is 1 (no cut).
        Returns
        -------
        fig : plotly.graph_objs.Figure
            The resulting figure object.
        """

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
    
    def _calc_velocity(self, x, t):
        """Calculates the velocity of a signal based on its time and displacement data.
        Parameters
        ----------
        x : numpy.ndarray
            The displacement data.
        t : numpy.ndarray
            The time data.
        Returns
        -------
        v : numpy.ndarray
            The velocity data.
        
        """

        v = 0 * x

        v[0] = (x[1] - x[0]) / (t[1] - t[0])
        v[-1] = (x[-1] - x[-2]) / (t[-1] - t[-2])

        for i in range(1, len(x) - 1):
            v_aux = (x[i + 1] - x[i]) / (t[i + 1] - t[i])
            v[i] = (v[i - 1] + v_aux) / 2

        return v


    def plot_poincare_section(self,
                            frequency,
                            dof=None,
                            cut=1):
        """Plots the Poincaré section of the response at a given frequency.
        Parameters
        ----------
        frequency : float
            The frequency at which to plot the Poincaré section.
        dof : list, optional
            A list of tuples. Each tuple must contain the names of X and Y DoFs for the desired Poincaré section.
            e.g. [("last_x", "last_y"), ("brg_x", "brg_y")]
            If None, the first two DoFs will be used as a X-Y pair.
        cut : int, optional
            The factor by which to downsample the data before plotting. Default is 1 (no cut).
        Returns
        -------
        fig : plotly.graph_objs.Figure
            The figure object containing the Poincaré section plot.
        """

        fig = go.Figure()

        i = self._find_frequency_index(frequency)
        d = self.ddl[i]
        t = d['time']
        k = list(d.keys())

        if dof is None:
            dof = [k[0]]

        for j, dof_i in enumerate(dof):
            if not isinstance(dof_i, tuple):
                v = self._calc_velocity(d[dof_i], t)
                dof[j] = (dof_i, str(dof_i) + '_vel')
                d[dof[j][1]] = v

        n_cut = int(len(d[dof[0][0]]) / cut)
        t_cut = t[:n_cut]

        T = 2 * np.pi / self.fl[i]
        t_pc = np.arange(0, t_cut[-1], T)

        max_amp_x = 0
        max_amp_y = 0
        for i in dof:           

            x = np.interp(t_pc, t_cut, d[i[0]][-n_cut:])
            y = np.interp(t_pc, t_cut, d[i[1]][-n_cut:])

            fig.add_trace(go.Scatter(x=x,
                                     y=y,
                                     mode='markers',
                                     marker=dict(color='black',
                                                 size=4),
                                     name=f'{i[0]} vs {i[1]}',
                                     showlegend=True,
                                     legendgroup=f'{i[0]}'
                                     )
                          )

            if np.max(x) > max_amp_x:
                max_amp_x = np.max(x)
            if np.max(y) > max_amp_y:
                max_amp_y = np.max(y)

        fig.update_layout(title=f'Poincaré Section',
                        #   xaxis_range=[- 1.1 * max_amp_x, 1.1 * max_amp_x],
                        #   yaxis_range=[- 1.1 * max_amp_y, 1.1 * max_amp_y],
                          xaxis_title=f'X [m]',
                          yaxis_title=f'Y [m]',
                          )
        fig = self._adjust_plot(fig)

        return fig

    def plot_spectrum(self,
                      dof,
                      frequency,
                      full_spectrum=False,
                      hanning=True,
                      cut=2,
                      max_frequency=None):
        """Plots the frequency spectrum of the response at a given frequency.
        Parameters
        ----------
        dof : list
            A list of degrees of freedom (DoF) to include in the plot.
            It must be a list of tuples containing X-Y pairs in case full_spectrum=True.
        frequency : float
            The frequency at which to plot the spectrum.
        full_spectrum : bool, optional
            Whether to plot the full spectrum (including negative frequencies).
            If True, the spectrum will include both backward and forward components.
            Default is False.
        hanning : bool, optional
            Whether to apply a Hanning window to the signal before calculating the spectrum.
            Default is True.
        cut : int, optional
            The factor by which to cut the signal.
            Default is 2.
        max_frequency : float, optional
            The maximum frequency to include in the plot.
            If None, the maximum frequency will be determined automatically.
            Default is None.
        Returns
        -------
        fig : plotly.graph_objs.Figure
            The figure object containing the frequency spectrum plot.
        """

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
        """Plots the frequency spectrum heatmap of the response.
        Parameters
        ----------
        dof : list
            The degrees of freedom to plot.
            It must be a list of tuples containing X-Y pairs in case full_spectrum=True.
        max_freq : float, optional
            The maximum frequency to include in the plot.
            If None, the maximum frequency will be determined automatically.
            Default is None.
        max_amp : float, optional
            The maximum amplitude to include in the plot.
            If None, the maximum amplitude will be determined automatically.
            Default is None.
        full_spectrum : bool, optional
            Whether to plot the full spectrum (including backward and forward components).
            Default is False.
        hanning : bool, optional
            Whether to apply a Hanning window to the signal before calculating the spectrum.
            Default is True.
        cut : int, optional
            The factor by which to cut the signal.
            The result will consider only the 1/cut last time steps.
            This should be used to remove residual transient responses from the results.
            Default is 2.
        Returns
        -------
        fig : plotly.graph_objs.Figure
            The figure object containing the heatmap and the synchronous line.
        """

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
            yaxis_title='Excitation Frequency [rad/s]',
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
        """Plots the 3D cascade plot of the response.
        Parameters
        ----------
        dof : list
            Degrees of freedom to plot.
            It must be a list of tuples containing X-Y pairs in case full_spectrum=True.
        max_freq : float, optional
            Maximum frequency to plot.
            Default is None, which means the maximum frequency will be determined automatically.
        max_amp : float, optional
            Maximum amplitude to plot.
            Default is None, which means the maximum amplitude will be determined automatically.
        full_spectrum : bool, optional
            Whether to plot the full spectrum (including backward and forward components).
            Default is False.
        hanning : bool, optional
            Whether to apply a Hanning window to the data.
            Default is True.
        log_amplitude : bool, optional
            Whether to plot the logarithm of the amplitude.
            Default is False.
        cut : float, optional
            The factor by which to cut the signal.
            The result will consider only the 1/cut last time steps.
            This should be used to remove residual transient responses from the results.
            Default is 2.

        Returns
        -------
        fig : plotly.graph_objs.Figure
            The resulting figure.
        """

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
                           title='Excitation Frequency [rad/s]'),
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
                   synch_freq=None,
                   backward_vector=False,
                   full_output=False):
        """Calculates the amplification vector between oscillators and base structure.

        Refer to Brandão et al. (2022) for further details.
        https://doi.org/10.1016/j.jsv.2022.116982
        Parameters
        ----------
        t : array_like
            Time vector.
        x : tuple of array_like
            Displacement time series of oscillator and base structure in X direction.
            The tuple should be arranged as (x_oscillator, x_base).
        y : tuple of array_like, optional
            Displacement time series of oscillator and base structure in Y direction.
            The tuple should be arranged as (y_oscillator, y_base).
            Only required if a full spectrum is to be computed.
            Default is None.
        cut : float, optional
            The factor by which to cut the signal.
            The result will consider only the 1/cut last time steps.
            Default is 2.
        hanning : bool, optional
            Whether to apply a Hanning window to the signal.
            Default is True.
        synch_freq : float, optional
            Frequency to which the signals will be synchronized.
            Default is None.
        backward_vector : bool, optional
            Whether to use the backward vector for the calculation.
            If False, uses backward component instead of backward vector.
            Default is False.
        full_output : bool, optional
            Whether to return the full output of the calculation.
            Full output includes the spectrum data for both oscillator and base DoFs.
            Default is False.
        Returns
        -------
        w : array_like
            Frequency vector corresponding to the spectrum.
        diff : array_like
            Difference between the two spectra.
            The result will consider only the 1/cut last frequency steps.
        spec_1 : array_like
            Spectrum of the first input (oscillator).
            Only if full_output is True.
        spec_2 : array_like
            Spectrum of the second input (base structure).
            Only if full_output is True.
        """


        if y is not None:
            w, spec_1 = self._calc_full_spectrum(t=t,
                                             x=x[0],
                                             y=y[0],
                                             cut=cut,
                                             hanning=hanning,
                                             synch_freq=synch_freq,
                                             return_complex=True,
                                             backward_vector=backward_vector)

            _, spec_2 = self._calc_full_spectrum(t=t,
                                                 x=x[1],
                                                 y=y[1],
                                                 cut=cut,
                                                 hanning=hanning,
                                                 synch_freq=synch_freq,
                                                 return_complex=True,
                                                 backward_vector=backward_vector)
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

        if full_output:
            return w, diff, spec_1, spec_2
        else:
            return w, diff

    def plot_diff_map(self,
                      dof,
                      dof_y=None,
                      mode='amp',
                      min_freq=None,
                      max_freq=None,
                      max_amp=None,
                      full_spectrum=False,
                      hanning=False,
                      cut=2,
                      log_mode=False,
                      backward_vector=False,
                      plot_3d=False
                      ):
        """Plots the differential amplification map of the response.

        Parameters
        ----------
        dof : list
            List of tuples, each representing one pair of DoFs to consider in the analysis.
            Each degree of freedom should be a tuple with the format, e.g. ('oscillator_x', 'base_x').
        dof_y : list, optional
            List of degrees of freedom to consider for the Y component.
            Must have a similar structure to `dof`.
            Needed only if full_spectrum=True.
            Default is None.
        mode : str, optional
            The mode of the plot. Can be 'amp', 'angle', or 'composition'.
            'amp' - Plot the amplitude of the differential response.
            'angle' - Plot the phase angle of the differential response.
            'composition' - Plot the composition of the differential response.
            Default is 'amp'.
        min_freq : float, optional
            Minimum frequency to consider. 
            If None, no minimum frequency will be applied.
            Default is None.
        max_freq : float, optional
            Maximum frequency to consider. 
            If None, no maximum frequency will be applied.
            Default is None.
        max_amp : float, optional
            Maximum amplitude to consider. 
            If None, no maximum amplitude will be applied.
            Default is None.
        full_spectrum : bool, optional
            Whether to plot the full spectrum (including backward and forward components).
            Default is False.
        hanning : bool, optional
            Whether to apply a Hanning window to the signals before computing the Fourier transform.
            Default is False.
        cut : int, optional
            The factor by which to cut the signal.
            The result will consider only the 1/cut last time steps.
            Default is 2.            
        log_mode : bool, optional
            Whether to use logarithmic scaling for the amplitude. 
            Default is False.
        backward_vector : bool, optional
            Whether to use the backward vector for the calculation.
            If False, uses backward component instead of backward vector.
            Default is False.
        plot_3d : bool, optional
            Whether to plot the data in 3D.
            If True, the data will be plotted in 3D.
            Default is False.
        Returns
        -------
        fig : plotly.graph_objs.Figure
            The figure object containing the plot.  
        """

        fig = go.Figure()
        t = self.ddl[0]['time']

        z = None

        for i, f in enumerate(self.fl):
            d = self.ddl[i]

            aux = 0
            for j, dof_i in enumerate(dof):
                if full_spectrum:
                    w, aux2, spec_1, spec_2 = self._calc_diff(
                        t=t,
                        x=(
                            np.interp(t, d['time'], d[dof_i[0]]),
                            np.interp(t, d['time'], d[dof_i[1]])
                        ),
                        y=(
                            np.interp(t, d['time'], d[dof_y[j][0]]),
                            np.interp(t, d['time'], d[dof_y[j][1]])
                        ),
                        hanning=hanning,
                        cut=cut,
                        backward_vector=backward_vector,
                        full_output=True
                    )
                    
                else:
                    w, aux2, spec_1, spec_2 = self._calc_diff(
                        t=t,
                        x=(
                            np.interp(t, d['time'], d[dof_i[0]]),
                            np.interp(t, d['time'], d[dof_i[1]])
                        ),
                        hanning=hanning,
                        cut=cut,
                        full_output=True
                    )

                if mode == 'amp':
                    aux += np.abs(aux2)

                elif mode == 'angle':
                    aux += np.angle(aux2) * 180 / (np.pi)

                elif mode == 'composition':
                    aux += np.abs(spec_1 - spec_2) * np.sin(np.angle(aux2)) * np.abs(w)
                    # aux2 = aux2 * np.abs(spec_2)
                    # aux3 = - 1.j * np.abs(w)
                    # aux += np.real(aux2) * np.real(aux3) + np.imag(aux2) * np.imag(aux3)
            
            aux = aux / len(dof)
            if min_freq is not None:
                for k, omg in enumerate(w):
                    if np.abs(omg) < min_freq:
                        aux[k] = np.nan

            if z is None:
                z = np.zeros((len(self.fl), len(aux)))#.astype(complex)

            z[i, :] = aux

        if max_freq is None:
            max_freq = np.max(w)

        if full_spectrum:
            freq_range = [- max_freq, max_freq]
        else:
            freq_range = [0, max_freq]

        if mode == 'amp' or mode == 'composition':
            if log_mode:
                z = np.log10(np.abs(z))
                zmin = np.min(z[np.isnan(z) == False])
                zmax = np.max(z[np.isnan(z) == False])
                legend_text = ' [log]'
            else:
                zmin = np.min(z[np.isnan(z) == False])
                zmax = np.max(z[np.isnan(z) == False])
                legend_text = ''

            if mode == 'composition':
                zmin = - np.max(np.abs(z[np.isnan(z) == False]))
                zmax = np.max(np.abs(z[np.isnan(z) == False]))

            colorbar = dict(title=f'Amplification{legend_text}')
            colorscale = 'Plasma'
            

        elif mode == 'angle':
            zmin = -180
            zmax = 180

            colorbar = dict(title='Phase angle [deg]',
                            tickvals=[-180, -90, 0, 90, 180],
                            ticktext=["-180", "-90", "0", "90", "180"],)
                            # tickvals=[0, 90, 180, 270, 360],
                            # ticktext=["0", "90", "180", "270", "360"],)
            colorscale = 'Phase'
        # else:
        #     print('WARNING: mode must be either amp or angle. Plot will show amplification.')

        if plot_3d:
            for i, f in enumerate(self.fl):
                d = self.ddl[i]
                

                fig.add_trace(go.Scatter3d(x=w,
                                        y=[f] * len(w),
                                        z=z[i, :],
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
                            title='Excitation Frequency [rad/s]'),
                    zaxis=dict(title='Amplitude [m]'),
                ),
            )

        else:
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
                yaxis_title='Excitation Frequency [rad/s]',
                legend=dict(orientation='h',
                            xanchor='center',
                            x=0.5,
                            yanchor='bottom',
                            y=1.05),
                width=900,
                height=700
            )

        return fig
    
class LinearResults():

    def __init__(self,
                 frequency_list,
                 res_forward,
                 res_backward,
                 system):
        """Initializes the LinearResults class with frequency list, forward and backward results, and system information.
        Parameters
        ----------
        frequency_list : list
            List of frequencies at which the results are computed.
        res_forward : dict
            Dictionary containing the forward excitation results.
        res_backward : dict
            Dictionary containing the backward excitation results.
        system : object
            System information object.

        Attributes
        ----------
        frequency_list : list
            List of frequencies at which the results are computed.
        res_forward : dict
            Dictionary containing the forward excitation results.
        res_backward : dict
            Dictionary containing the backward excitation results.
        system : object
            System information object.
        rf : dict
            Forward excitation results.
        rb : dict
            Backward excitation results.
        fl : list
            Frequency list.
        Notes
        -----
        This class is used to store the results of the linear analysis.

        """

        self.frequency_list = frequency_list
        self.res_forward = res_forward
        self.res_backward = res_backward
        self.system = system

        self.rf = self.res_forward
        self.rb = self.res_backward
        self.fl = self.frequency_list

    def _find_frequency_index(self, f):
        """
        Finds the index of the closest frequency in the frequency list.
        Parameters
        ----------
        f : float
            The frequency to find in the frequency list.
        Returns
        -------
        int
            The index of the closest frequency in the frequency list.
        """

        a = np.array(self.fl) - f
        i = np.argmin(abs(a))

        return i

    def _calc_amplitude(self,
                        x,
                        amplitude_units='rms'):
        """
        Calculates the amplitude of a signal.
        Parameters
        ----------
        x : array_like
            Input signal.
        amplitude_units : str, optional
            The units of the amplitude to be calculated.
            Can be one of the following: 'max_displacement', 'rms', 'pk', 'pk-pk'.
        Returns
        -------
        float
            The amplitude of the signal.
        

        """

        t = 5 * 2 * np.pi * np.linspace(0, 1, 50)
        if amplitude_units == 'max_displacement':
            aux = []
            for a in x:
                aux.append(np.abs(a) * np.sin(t + np.angle(a)))

            amp = IntegrationResults._calc_amplitude(x=aux,
                                                     cut=1,
                                                     amplitude_units=amplitude_units)
        elif amplitude_units == 'rms':
            amp = np.abs(x) / np.sqrt(2)

        elif amplitude_units == 'pk':
            amp = np.abs(x)

        elif amplitude_units == 'pk-pk':
            amp = np.abs(x) * 2

        return amp

    def plot_frf(self,
                 dof=None,
                 whirl='both',
                 amplitude_units='rms',
                 frequency_units='rad/s'):
        """Plots the frequency response function (FRF) for the given degrees of freedom (DoF).
        Parameters
        ----------
        dof : list, optional
            List of degrees of freedom (DoF) to plot. If None, all DoFs are plotted.
        whirl : str, optional
            The type of whirl to consider. Can be one of the following: 'both', 'forward', 'backward', 'unbalance'.
            Default is 'both'.
        amplitude_units : str, optional
            The units of the amplitude to be plotted. Can be one of the following: 'max_displacement', 'rms', 'pk', 'pk-pk'.
            Default is 'rms'.
        frequency_units : str, optional
            The units of the frequency to be plotted. Can be one of the following: 'rad/s', 'RPM'.
            Default is 'rad/s'.
        Returns
        -------
        fig_1 : plotly.graph_objs.Figure
            Figure object containing the forward excitation plot.
        fig_2 : plotly.graph_objs.Figure
            Figure object containing the backward excitation plot.
        """

        fig_1 = go.Figure()
        fig_2 = go.Figure()
        if dof is None:
            dof = [j for j in self.rf.keys()]
        if whirl == 'both':
            dl = [self.rf, self.rb]
        elif whirl == 'forward' or whirl == 'unbalance':
            dl = [self.rf]
        elif whirl == 'backward':
            dl = [self.rb]

        freq_convert = 1
        if frequency_units.upper() == 'RPM':
            freq_convert = 60 / (2 * np.pi)

        amp = np.zeros((len(dof), len(self.fl), len(dl)))

        for i, d in enumerate(dl):

            if amplitude_units == 'max_displacement' or amplitude_units == 'major_axis':
                d_aux = [(d[k[0]], d[k[1]]) for k in dof]
                amp[:, :, i] = np.array([
                    [self._calc_amplitude((k[0][j], k[1][j]),
                                          amplitude_units=amplitude_units) for j in range(len(self.fl))] for k in
                    d_aux])
            else:
                amp[:, :, i] = np.array([
                    [self._calc_amplitude(d[k][j],
                                          amplitude_units=amplitude_units) for j in range(len(self.fl))] for k in
                    dof])

        for i, p in enumerate(dof):
           
            fig_1.add_trace(go.Scatter(x=self.fl * freq_convert, y=amp[i, :, 0], name=f'DoF: {p}'))
            if 'unb' in whirl:
                fig_1.data[-1].y = fig_1.data[-1].y * self.fl ** 2

        if amp.shape[2] == 2:
            for i, p in enumerate(dof):
                fig_2.add_trace(go.Scatter(x=self.fl * freq_convert, y=amp[i, :, 1], name=f'DoF: {p}'))

        fig_1.update_layout(
            xaxis={'range': [0, np.max(self.fl) * freq_convert],
                   },
            xaxis_title=f'Frequency [{frequency_units}]',
            yaxis_title=f'Amplitude [m RMS]',
            title='Forward Excitation'
        )
        fig_1.update_yaxes(type="log")

        fig_2.update_layout(
            xaxis={'range': [0, np.max(self.fl)],
                   },
            xaxis_title=f'Frequency [{frequency_units}]',
            yaxis_title=f'Amplitude [m {amplitude_units}]',
            title='Backward Excitation'
        )
        fig_2.update_yaxes(type="log")

        fig_1 = IntegrationResults._adjust_plot(fig_1)
        fig_2 = IntegrationResults._adjust_plot(fig_2)

        if whirl == 'both':
            return fig_1, fig_2
        else:
            return fig_1
        
    def plot_orbit(self,
                   frequency,
                   dof,
                   whirl='forward',
                   f=1):
        """Plots the orbit plot for the given frequency and degrees of freedom (DoF).
        Parameters
        ----------
        frequency : float
            The frequency at which to plot the orbit.
        dof : list
            The degrees of freedom (DoF) to consider for the orbit plot.
            Each element should be a tuple (x, y) representing the DoF.
        whirl : str, optional
            The type of whirl to consider. Can be one of the following: 'forward', 'backward', 'unbalance'.
            Default is 'forward'.
        f : float, optional
            The scaling factor for the orbit plot.
            Default is 1.
        Returns
        -------
        fig : plotly.graph_objs.Figure
            The figure containing the orbit plot.
        """

        fig = go.Figure()
        
        if whirl == 'forward' or whirl == 'unbalance':
            dl = self.rf
        elif whirl == 'backward':
            dl = self.rb

        i = self._find_frequency_index(frequency)
        t = np.linspace(0, 1.9 * np.pi, 200)
        if whirl == 'unbalance':
            f = f * self.fl[i] ** 2

        max_amp = 0
        for j, d in enumerate(dof):
            x = f * np.abs(dl[d[0]][i]) * np.cos(t + np.angle(dl[d[0]][i]))
            y = f * np.abs(dl[d[1]][i]) * np.cos(t + np.angle(dl[d[1]][i]))
            fig.add_trace(go.Scatter(x=x,
                                     y=y,
                                     name=f'{d[0]} vs {d[1]}',
                                     mode='lines',
                                     legendgroup=f'{j}')
                          )

            fig.add_trace(go.Scatter(x=[x[0]],
                                     y=[y[0]],
                                     mode='markers',
                                     marker=dict(color='red'),
                                     name='Poincaré Section',
                                     showlegend=False,
                                     legendgroup=f'{j}'
                                     )
                          )
            
            if np.max(np.abs(x)) > max_amp:
                max_amp = np.max(np.abs(x))
            if np.max(np.abs(y)) > max_amp:
                max_amp = np.max(np.abs(y))

        fig.update_layout(title=f'Orbit Plot',
                          xaxis_range=[- 1.1 * max_amp, 1.1 * max_amp],
                          yaxis_range=[- 1.1 * max_amp, 1.1 * max_amp],
                          xaxis_title=f'X [m]',
                          yaxis_title=f'Y [m]',
                          )
        fig = IntegrationResults._adjust_plot(fig)

        return fig
