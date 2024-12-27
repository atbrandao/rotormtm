import numpy as np
import numpy.linalg as la
from scipy.optimize import least_squares, fsolve
import plotly.graph_objects as go
import time
from pickle import dump
import subprocess
import pandas as pd
import os
from .results import IntegrationResults, LinearResults
from copy import deepcopy

class Sys_NL:

    def __init__(self,
                 M,K,Snl,beta,alpha,
                 n_harm=10,nu=1,N=2,cp=1e-4,C=None,rotor=None):

        self.M = M
        self.cp = cp

        if C is None:
            self.C = K * cp
        else:
            self.C = C
        self.K = K
        self.Snl = Snl
        self.beta = beta
        self.alpha = alpha
        self.n_harm = n_harm
        self.nu = nu
        self.N = N
        if self.alpha != 0:
            self.x_eq = np.sqrt(-self.beta/self.alpha)
        else:
            self.x_eq = 0
        self.ndof = len(self.K)
        self.K_lin = self.K - self.Snl * 2 * self.beta

        self.dof_nl = []
        self.base_dof = []
        for i in range(len(self.Snl)):
            if self.Snl[i, i] != 0:
                if self.K[i, i] == 0:
                    self.dof_nl.append(i)
                else:
                    self.base_dof.append(i)

        Z = np.zeros((self.ndof, self.ndof))
        I = np.eye(self.ndof)

        self.Minv = np.vstack([np.hstack([I, Z]),
                           np.hstack([Z, la.inv(M)])])
        self.A_lin = np.vstack(
            [np.hstack([Z, I]),
             np.hstack([la.solve(-self.M, self.K_lin), la.solve(-self.M, self.C)])])
        self.A = np.vstack(
            [np.hstack([Z, I]),
             np.hstack([la.solve(-self.M, self.K), la.solve(-self.M, self.C)])])
        self.full_orbit = False

        self.rotor = rotor

    def H(self, omg):

        M = self.M
        A = self.A_lin

        N = len(M)

        H = np.linalg.inv(1.j * omg * np.eye(2 * N) - A)

        return H

    def A_hb(self,
             omg):

        Z = np.zeros(self.K.shape)
        Z2 = np.vstack([Z, Z])

        A_out = np.vstack([np.hstack([self.K]+[Z]*(2*self.n_harm))] + \
                          [np.hstack([Z2]*(1+2*i)+[np.vstack([np.hstack([self.K-((i+1)*(omg/self.nu))**2*self.M, -(i+1)*omg/self.nu*self.C]),
                                                               np.hstack([(i+1)*omg/self.nu*self.C, self.K-((i+1)*(omg/self.nu))**2*self.M])])]+ \
                                     [Z2]*(2*self.n_harm-2*(i+1))) for i in range(self.n_harm)])

        return A_out

    def dt(self, omg):

        w0 = omg / self.nu
        dt = 1 / (2 * w0 / (2 * np.pi) * self.n_harm * self.N)

        return dt

    def t(self, omg, t0=0):

        w0 = omg / self.nu
        dt = self.dt(omg)
        tf = 1 / (w0/(2*np.pi))
        t = np.arange(t0, tf - dt/10, dt) # 2 * np.pi / w0, 2 * self.N * self.n_harm)
        t = t.reshape((len(t), 1))

        return t

    def gamma(self, omg, t0=0, t=None):

        if t is None:
            t = self.t(omg, t0=t0)
        else:
            t = t.reshape((len(t), 1))

        id_n = np.eye(self.ndof)
        w0 = omg/self.nu
        wt = w0 * t # self.t(omg, t0=t0)
        # print(wt)
        gamma = np.hstack([np.kron(id_n,np.cos(0*wt))] + [np.hstack([np.kron(id_n,np.sin((i+1)*wt)),
                                                                     np.kron(id_n,np.cos((i+1)*wt))]) for i in range(self.n_harm)])

        return gamma

    def dgamma_dt(self, omg, t0=0, t=None):

        if t is None:
            t = self.t(omg, t0=t0)
        else:
            t = t.reshape((len(t), 1))

        id_n = np.eye(self.ndof)
        w0 = omg/self.nu
        wt = w0 * t
        # print(wt)
        dgamma_dt = np.hstack([np.kron(id_n,0*np.cos(0*wt))] + [np.hstack([np.kron(id_n,(i+1)*w0*np.cos((i+1)*wt)),
                                                                     np.kron(id_n,-(i+1)*w0*np.sin((i+1)*wt))]) for i in range(self.n_harm)])

        return dgamma_dt

    def B(self, f, omg, t):

        F_aux = np.zeros((2*self.ndof, 1))
        F = np.zeros((self.ndof, 1))
        for n in f:
            F[n, 0] = np.real(f[n]) * np.cos(omg * t) + np.imag(f[n]) * np.sin(omg * t)

        F_aux[len(F):, :] = F[:, :]
        B = self.Minv @ F_aux

        return B

    def dB_dt(self, f, omg, t):

        F_aux = np.zeros((2*self.ndof, 1))
        F = np.zeros((self.ndof, 1))
        for n in f:
            F[n, 0] = - np.real(f[n]) * np.sin(omg * t) + np.imag(f[n]) * np.cos(omg * t)

        F_aux[len(F):, :] = F[:, :]
        dB_dt = self.Minv @ F_aux

        return dB_dt

    def f_stsp(self, x, B):

        x = np.reshape(x, (len(x), 1))
        B = np.reshape(B, (len(B), 1))
        Z = np.zeros((self.ndof, self.ndof))

        A = self.A
        alpha = self.alpha
        beta = self.beta
        Minv = self.Minv
        Snl_stsp = np.vstack([np.hstack([Z, Z]),
                              np.hstack([-self.Snl, Z])])

        f_stsp = A @ x + B + Minv @ (beta * Snl_stsp @ x + alpha * (Snl_stsp @ x) ** 3)

        return f_stsp

    def df_stsp_dx(self, x):

        x = np.reshape(x, (len(x), 1))
        Z = np.zeros((self.ndof, self.ndof))

        A = self.A
        alpha = self.alpha
        beta = self.beta
        Minv = self.Minv
        Snl_stsp = np.vstack([np.hstack([Z, Z]),
                              np.hstack([-self.Snl, Z])])

        df_stsp_dx = A + Minv @ (beta * Snl_stsp + 3 * alpha * ((Snl_stsp @ x) ** 2 * np.eye(len(x))) @ Snl_stsp)

        return df_stsp_dx

    def f_nl(self,x):

        try:
            self.Snl2
        except:
            id_N = np.eye(2*self.N*self.n_harm)
            self.Snl2 = np.vstack(
                [np.hstack([id_N*self.Snl[i,j] for j in range(len(self.Snl))]) for i in range(len(self.Snl))])

        f_nl = self.beta * self.Snl2 @ x + self.alpha * (self.Snl2 @ x)**3

        return f_nl

    def df_dx(self,x):

        try:
            self.Snl2
        except:
            id_N = np.eye(2 * self.N * self.n_harm)
            self.Snl2 = np.vstack(
                [np.hstack([id_N * self.Snl[i, j] for j in range(len(self.Snl))]) for i in range(len(self.Snl))])

        # Pela definição: df_dx = - d(f_nl)_dx
        df_dx = - (self.beta * self.Snl2 + 3 * self.alpha * ((self.Snl2 @ x)**2 * np.eye(len(x))) @ self.Snl2)

        return df_dx

    def h(self,z,*args):

        z = z.reshape((len(z),1))

        omg = args[0]
        if len(args) == 1:
            f_omg = {}
        else:
            f_omg = args[1]

        if len(args) > 3:
            g = args[2]
            gi = args[3]
        else:
            g = self.gamma(omg)
            gi = la.pinv(g)

        b_f = np.zeros((self.ndof*(2*self.n_harm+1),1))
        for f in f_omg:
            b_f[self.ndof + 2 * (self.nu - 1) * self.ndof + f] = np.imag(f_omg[f])
            b_f[2 * self.ndof + 2 * (self.nu - 1) * self.ndof + f] = np.real(f_omg[f])
        # print(b_f)
        x = g @ z
        b = b_f - gi @ self.f_nl(x)
        h = self.A_hb(omg) @ z - b

        return h.reshape(len(h))

    def dh_dz(self,z,*args):

        omg = args[0]

        if len(args) > 3:
            g = args[2]
            gi = args[3]
        else:
            g = self.gamma(omg)
            gi = la.pinv(g)

        x = g @ z
        db_dz = gi @ self.df_dx(x) @ g
        dh_dz = self.A_hb(omg) - db_dz

        return dh_dz

    def z0(self,omg,f_omg):

        H = self.H(omg)

        F = np.zeros((2*self.ndof,1)).astype(complex)
        for f in f_omg:
            F[self.ndof+f] = f_omg[f]
        x = H @ self.Minv @ F

        z0 = np.zeros((self.ndof*(2*self.n_harm+1),1))
        z0[self.dof_nl] = self.x_eq
        for i, y in enumerate(x[:self.ndof]):
            z0[self.ndof + 2 * (self.nu - 1) * self.ndof + i] = np.imag(y)
            z0[2 * self.ndof + 2 * (self.nu - 1) * self.ndof + i] = np.real(y)

        return z0.reshape(len(z0))
    
    def calc_linear_frf(self,
                        sp_arr,
                        f=None,
                        probe_dof=None,
                        probe_names=None,
                        f_node=0,
                        rotor_solo=False,
                        silent=True
                        ):
        
        if self.rotor is not None:
            res = self.rotor.calc_frf(sp_arr,
                                        f,
                                        probe_dof=None,
                                        probe_names=None,
                                        f_node=0,
                                        rotor_solo=False,
                                        silent=True)
            return res
        
        else:

            if probe_dof is None:
                probe_dof = [a for a in range(self.ndof)]

            if probe_names is None:
                probe_names = [p for p in probe_dof]

            if f is None:
                f = np.ones(len(sp_arr))

            res_fow = {p: np.zeros(len(sp_arr)).astype(complex) for p in probe_names}
            res_back = {p: np.zeros(len(sp_arr)).astype(complex) for p in probe_names}
            for i, omg in enumerate(sp_arr):

                H = self.H(omg)

                F = np.zeros((2 * self.ndof, 1)).astype(complex)
                F[self.ndof + f_node] = 1
                r = f[i] * H @ self.Minv @ F

                for j, p in enumerate(probe_names):
                    res_fow[p][i] = r[probe_dof[j], 0]
                    res_back[p][i] = r[probe_dof[j], 0]

                if not silent:
                    print(f'Linear response calculated for frequency: {omg:.1f} rad/s')

            return LinearResults(sp_arr,
                                res_fow,
                                res_back,
                                self)

    def solve_hb(self, f, omg, z0=None, full_output=False, method=None, state_space=False, plot_orbit=False):

        g = self.gamma(omg)
        gi = la.pinv(self.gamma(omg))

        if z0 is None:
            z0 = self.z0(omg=omg, f_omg=f)

        if method is None:
            res = fsolve(func=self.h, x0=z0, fprime=self.dh_dz, args=(omg, f, g, gi), full_output=full_output)
        else:
            res = least_squares(fun=self.h, x0=z0, jac=self.dh_dz, args=(omg, f, g, gi))

        if full_output:
            if method is None:
                root = res[0]
            else:
                root = res.x
        else:
            if method is None:
                root = res
            else:
                root = res.x

        x = self.inv_fourier(root, omg, state_space)

        # x = self.gamma(omg) @ root.reshape((len(root), 1))
        # x = x.reshape(len(x))
        # x = x.reshape((self.ndof,len(x)//self.ndof))#[:,:-1]
        # if state_space:
        #     v = self.dgamma_dt(omg) @ root.reshape((len(root), 1))
        #     v = v.reshape(len(v))
        #     v = v.reshape((self.ndof, len(v) // self.ndof))#[:, :-1]
        #     x = np.vstack([x,
        #                    v])

        if plot_orbit:
            Ni = int(np.round(x.shape[1] / self.nu))
            fig = self.plot_orbit(x, Ni)
            return fig, x
        else:
            if full_output:
                return x, res
            else:
                return x

    def inv_fourier(self, z, omg, state_space=False):

        x = self.gamma(omg) @ z.reshape((len(z), 1))
        x = x.reshape(len(x))
        x = x.reshape((self.ndof, len(x) // self.ndof))
        if state_space:
            v = self.dgamma_dt(omg) @ z.reshape((len(z), 1))
            v = v.reshape(len(v))
            v = v.reshape((self.ndof, len(v) // self.ndof))  # [:, :-1]
            x = np.vstack([x,
                           v])
        return x

    def z_to_x(self, z, t, omg, state_space=False):

        aux = np.append(np.array([1]),
                        np.hstack([np.array([np.sin(omg * i / self.nu * t),
                                             np.cos(omg * i / self.nu * t)]) for i in range(1, self.n_harm + 1)]))
        x = np.zeros((self.ndof, 1))
        for i in range(self.ndof):
            x[i] = np.sum(aux * z[i::self.ndof])
        #
        # x = self.gamma(omg,
        #                t=np.array([t])) @ z
        x = x.reshape(len(x), 1)

        if state_space:
            aux_v = np.append(np.array([1]),
                              np.hstack([np.array([omg * i / self.nu * np.cos(omg * i / self.nu * t),
                                                   - omg * i / self.nu * np.sin(omg * i / self.nu * t)]) for i in
                                         range(1, self.n_harm + 1)
                                         ]
                                        )
                              )
            v = np.zeros((self.ndof, 1))
            for i in range(self.ndof):
                v[i] = np.sum(aux_v * z[i::self.ndof])
            # v = self.dgamma_dt(omg,
            #                    t=np.array([t])) @ z
            v = v.reshape(len(v), 1)
            x = np.vstack([x,
                           v])

        return x

    def floquet_multipliers(self, omg, z, dt_refine=None):

        M = np.eye(len(self.A))

        if dt_refine is None:
            dt_refine = np.ceil(self.dt(omg) / self.dt_max())

        t_base = self.t(omg)[:, 0]

        x = self.inv_fourier(z, omg, state_space=True)

        N0 = self.N
        self.N = self.N * dt_refine

        t = self.t(omg)
        dt = self.dt(omg)

        self.N = N0

        for i, t1 in enumerate(t):

            # x_1 = self.z_to_x(z, t1, omg, state_space=True)
            # x_2 = self.z_to_x(z, t1 + dt / 2, omg, state_space=True)
            # x_3 = self.z_to_x(z, t1 + dt, omg, state_space=True)

            x_1 = np.array([np.interp(t1, t_base, x[i, :]) for i in range(x.shape[0])]).reshape((x.shape[0], 1))
            x_2 = np.array([np.interp(t1 + dt / 2, t_base, x[i, :]) for i in range(x.shape[0])]).reshape((x.shape[0], 1))
            x_3 = np.array([np.interp(t1 + dt, t_base, x[i, :]) for i in range(x.shape[0])]).reshape((x.shape[0], 1))

            k1 = self.df_stsp_dx(x_1) @ M
            M1 = M + k1 * dt / 2
            k2 = self.df_stsp_dx(x_2) @ M1
            M2 = M + k2 * dt / 2
            k3 = self.df_stsp_dx(x_2) @ M2
            M3 = M + k3 * dt
            k4 = self.df_stsp_dx(x_3) @ M3

            M += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


        fm = la.eig(M)[0]

        return fm

    def dt_max(self):
        '''
        Maximum recommended value of time step for direct integration.
        '''

        dt_max = 0.5 * np.pi / np.max(np.imag(np.linalg.eig(self.A_lin)[0]))

        return dt_max

    def RK4_NL(self, B, x, dt):

        x = np.reshape(x, (len(x), 1))
        Z = np.zeros((self.ndof, self.ndof))

        A = self.A
        alpha = self.alpha
        beta = self.beta
        Minv = self.Minv
        Snl_stsp = np.vstack([np.hstack([Z, Z]),
                              np.hstack([-self.Snl, Z])])

        aux1 = np.zeros((len(B), 1)).astype(type(B[0, 0]))
        aux1[:, 0] = B[:, 0]
        aux2 = np.zeros((len(B), 1)).astype(type(B[0, 0]))
        aux2[:, 0] = B[:, 1]
        aux3 = np.zeros((len(B), 1)).astype(type(B[0, 0]))
        aux3[:, 0] = B[:, 2]

        v_aux = Snl_stsp @ x
        k1 = A @ x + aux1 + Minv @ (beta * v_aux + alpha * v_aux ** 3)
        x1 = x + k1 * dt / 2
        v_aux = Snl_stsp @ x1
        k2 = A @ x1 + aux2 + Minv @ (beta * v_aux + alpha * v_aux ** 3)
        x2 = x + k2 * dt / 2
        v_aux = Snl_stsp @ x2
        k3 = A @ x2 + aux2 + Minv @ (beta * v_aux + alpha * v_aux ** 3)
        x3 = x + k3 * dt
        v_aux = Snl_stsp @ x3
        k4 = A @ x3 + aux3 + Minv @ (beta * v_aux + alpha * v_aux ** 3)

        return np.reshape(x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), (len(x), 1))

    def export_sys_data(self, filename='data.dat', **kwargs):

        with open(filename, 'w') as f:
            f.write('### Custom parameters ###\n')
            f.write(f'{len(kwargs)}\t#Number of custom parameters\n')
            for k in kwargs:
                if isinstance(kwargs[k], dict):
                    f.write(f'{len(kwargs[k])}\t# Number of excitation points\n')
                    for i, k2 in enumerate(kwargs[k]):
                        f.write(f'{k2}\t')
                        f.write(f'{np.real(kwargs[k][k2]):.6f}\t')
                        f.write(f'{np.imag(kwargs[k][k2]):.6f}\t# Excitation #{i+1}\n')
                elif isinstance(kwargs[k], np.ndarray):
                    f.writelines([f'{kwargs[k].flatten()[i]:.6f}\t' for i in range(len(kwargs[k]))])
                    f.write('\n')
                elif isinstance(kwargs[k], list):
                    f.writelines([f'{kwargs[k][i]:.6f}\t' for i in range(len(kwargs[k]))])
                    f.write('\n')
                else:
                    f.write(f'{kwargs[k]:.6f}\t#{k}\n')
            f.write('### System parameters ###\n')
            f.write(f'{self.ndof}\t#ndof\n')
            f.write(f'{self.alpha:.6f}\t#alpha\n')
            f.write(f'{self.beta:.6f}\t#beta\n')
            f.writelines([f'{self.A.flatten()[i]:.6f}\t' for i in range((2 * self.ndof)**2)])
            f.write('\n')
            f.writelines([f'{self.Minv.flatten()[i]:.6f}\t' for i in range((2 * self.ndof) ** 2)])
            f.write('\n')
            f.writelines([f'{self.Snl.flatten()[i]:.6f}\t' for i in range(self.ndof ** 2)])

    def solve_transient(self, f, t, omg, x0, t_out=None, probe_dof=None, last_x=False,
                        plot_orbit=False, dt=None, orbit3d=False, run_fortran=False,
                        keep_data=False, ):

        if probe_dof is None:
            probe_dof = [i for i in range(len(x0))]

        if t_out is None:
            t_out = t

        x = np.zeros((2 * self.ndof, 2))
        x[:, 0] = x0[:, 0]
        F_aux = np.zeros((2 * self.ndof, 3))

        x_out = np.zeros((len(probe_dof), len(t_out)))
        x_out[:,0] = x0[probe_dof, 0]

        if run_fortran:

            self.export_sys_data(f=f,
                                 dt=t[1] - t[0],
                                 tf=t[-1],
                                 N=len(t),
                                 omg=omg,
                                 ndof=self.ndof,
                                 x0=x0,
                                 n_probe=len(probe_dof),
                                 probe_dof=probe_dof,
                                 )

            process = subprocess.Popen('rk4_fortran', shell=True, stdout=subprocess.PIPE)
            process.wait()
            if not keep_data:
                os.remove('data.dat')
            df = pd.read_csv('saida.txt', names=probe_dof, delim_whitespace=True)
            for i, p in enumerate(probe_dof):
                x_out[i, :] = df[p]

        else:
            j = 1
            for i in range(1, len(t)):
                if dt is None:
                    dt = t[i] - t[i-1]

                F = np.zeros((self.ndof, 3))
                for n in f:
                    F[n, 0] = np.real(f[n]) * np.cos(omg * t[i-1]) + np.imag(f[n]) * np.sin(omg * t[i-1])
                    F[n, 1] = np.real(f[n]) * np.cos(omg * t[i-1]+dt/2) + np.imag(f[n]) * np.sin(omg * t[i-1]+dt/2)
                    F[n, 2] = np.real(f[n]) * np.cos(omg * t[i-1]+dt) + np.imag(f[n]) * np.sin(omg * t[i-1]+dt)

                F_aux[len(F):, :] = F
                B_aux = self.Minv @ F_aux

                x[:,1] = self.RK4_NL(B_aux,x[:,0],dt)[:,0]

                if np.round(10*i/len(t)) > np.round(10*(i-1)/len(t)):
                    print(np.round(100*i/len(t)))
                if j < len(t_out) and t[i] == t_out[j]:
                    x_out[:, j] = x[probe_dof, 1]
                    j += 1
                elif j < len(t_out) and t[i] > t_out[j]:
                    x_out[:, j] = x[probe_dof, 0] + (x[probe_dof, 1] - x[probe_dof, 0]) * (t[i] - t_out[j]) / (t[i] - t[i-1])

                x[:,0] = x[:, 1]

        if plot_orbit:
            Ni = int(np.round(2*np.pi/omg / dt))
            if self.full_orbit:
                if orbit3d:
                    fig = self.plot_3d_orbit(x_out[:, :], Ni)
                else:
                    fig = self.plot_orbit(x_out[:,:], Ni)
            else:
                if orbit3d:
                    fig = self.plot_3d_orbit(x_out[:, x_out.shape[1] // 2:], Ni)
                else:
                    fig = self.plot_orbit(x_out[:,x_out.shape[1]//2:],Ni)
            return fig, x_out
        elif last_x:
            return x_out, x[:,1]
        else:
            return x_out

    def power_in(self, x, F):
        """
        This function computes the instantaneous power input from a excitation source F into the system
        with state x.

        Parameters
        -------
        x: array
            State space vector of the system's state at a particular point in time.

        F: array
            Vector containing the instantaneous force, in Newtons, applied to each of the system's DoFs.

        Returns
        -------
        array
            Array with the power input, in Watts, of each of the system's DoFs.

        """

        if len(x.shape) > 1:
            v = x[self.ndof:, :]
        else:
            v = x[self.ndof:]
        power_in = F * v

        return power_in

    def kinetic_energy(self, x, dof=None, separate_dof=False):
        """
        System's instantaneous kinetic energy given a state x.

        Parameters
        -------
        x: array
            State space vector of the system's state at a particular point in time.

        Returns
        -------
        array
            Array with the kinetic energy, in Joules, of each of the system's DoFs.

        """

        if dof is None:
            dof = [a for a in range(self.ndof)]
        else:
            dof = list(dof)

        if len(x.shape) > 1:
            v = x[self.ndof:, :]
            v = v[dof, :]
        else:
            v = x[self.ndof:]
            v = v[dof].reshape((len(dof), 1))

        if separate_dof:
            kinetic_energy = 1 / 2 * v * (self.M[np.ix_(dof, dof)] @ v)
        else:
            kinetic_energy = 1 / 2 * np.sum(v * (self.M[np.ix_(dof, dof)] @ v), 0)

        return kinetic_energy

    def base_potential_energy(self, x, dof=None):
        """
        Calculates the base structure instantaneous potential energy, which excludes
         the potential energy stored on the Duffin oscillators, given a state x.

        Parameters
        -------
        x: array
            State space vector of the system's state at a particular point in time.

        dof: list
            A list of the linear DoFs to be considered for the potential energy calculation.

        Returns
        -------
        array
            Array with the base structures total potential energy, in Joules, of each given x.
        """

        if dof is None:
            dof = [a for a in range(self.ndof)]
        else:
            dof = list(dof)

        if len(x.shape) > 1:
            x = x[dof, :]
        else:
            x = x[dof].reshape((len(dof), 1))

        potential_energy = 1 / 2 * np.sum(x * (self.K[np.ix_(dof,dof)] @ x), 0)

        # if len(x.shape) > 1:
        #     potential_energy = np.diag(potential_energy)

        return potential_energy

    def dof_nl_potential_energy(self, x):
        """
        Calculates the system's instantaneous potential energy on the nonlinear attachments given a state x.

        Parameters
        -------
        x: array
            State space vector of the system's state at a particular point in time.

        Returns
        -------
        array
            Array with the potential energy, in Joules, of each of the system's nonlinear DoFs.

        """

        if len(x.shape) > 1:
            x = x[:self.ndof, :]
        else:
            x = x[:self.ndof, :].reshape((self.ndof, 1))

        potential_energy = 1/2 * self.beta * (self.Snl[self.dof_nl, :] @ x) ** 2 \
                           + 1/4 * self.alpha * (self.Snl[self.dof_nl, :] @ x) ** 4

        return potential_energy

    def dof_nl_forces(self, x):
        """
        Calculates the forces to which the nonlinear DoFs are subjected to given a certain state x.

        Parameters
        -------
        x: array
            State space vector of the system's state at a particular point in time.

        Returns
        -------
        array
            Array with the damping forces, in Newtons, applied on each of the system's nonlinear DoFs.

        array
            Array with the elastic forces, in Newtons, applied on each of the system's nonlinear DoFs.

        """

        if len(x.shape) > 1:
            v = x[self.ndof:, :]
            x = x[:self.ndof, :]
        else:
            v = x[self.ndof:].reshape((self.ndof, 1))
            x = x[:self.ndof, :].reshape((self.ndof, 1))

        # x = x[:self.ndof]
        F_damping = - self.C[self.dof_nl, :] @ v
        F_elastic = - self.K[self.dof_nl, :] @ x \
                    - self.beta * self.Snl[self.dof_nl, :] @ x \
                    - self.alpha * (self.Snl[self.dof_nl, :] @ x) ** 3

        return F_damping, F_elastic

    def dof_nl_energy_flow(self, x):
        """
        Calculates the energy flow from main structure to nonlinear resonators at a given state x.

        Parameters
        -------
        x: array
            State space vector of the system's state at a particular point in time.

        Returns
        -------
        array
            Array with the energy flow, in Watts, going into each of the system's nonlinear DoFs.
        """

        if len(x.shape) > 1:
            v = x[self.ndof:, :][self.dof_nl, :]
        else:
            v = x[self.ndof:].reshape((self.ndof, 1))[self.dof_nl, :]

        Fd, Fe = self.dof_nl_forces(x)
        F = Fd + Fe
        energy_flow = F * v

        return energy_flow

    def base_structure_energy_flow(self, x):
        """
        Calculates the energy flow from nonlinear resonators to main structure at a given state x.

        Parameters
        -------
        x: array
            State space vector of the system's state at a particular point in time.

        Returns
        -------
        array
            Array with the energy flow, in Watts, going into each of the base structure's DoFs.
        """

        if len(x.shape) > 1:
            v = x[self.ndof:, :][self.base_dof, :]
        else:
            v = x[self.ndof:].reshape((self.ndof, 1))[self.base_dof, :]

        Fd, Fe = self.dof_nl_forces(x)
        F = Fd + Fe
        energy_flow = - F * v

        return energy_flow

    def plot_frf(self, omg_range, f, tf=300, dt_base=None, rms_rk=None,
                 continuation=True, method=None, probe_dof=None, dt_refine=None,
                 stability_analysis=True, save_rms_rk=None, save_rms_hb=None,
                 energy_analysis=False, run_hb=True):

        rms_hb = np.zeros((1+2*self.ndof, len(omg_range)))
        fm_flag = np.zeros(len(omg_range))
        pc = []
        cost_hb = []

        if dt_base is None:
            dt_base = self.dt_max()

        if probe_dof is None:
            probe_dof = [i for i in range(self.ndof)]

        if rms_rk is None:
            calc = True
            rms_rk = np.zeros((2 * self.ndof, len(omg_range)))
        else:
            calc = False

        if energy_analysis:
            calc = True

        n_points = 50
        z0 = self.z0(omg=omg_range[0], f_omg={0: 0})  # None
        x0 = np.vstack([z0[:self.ndof].reshape((self.ndof, 1)),
                        0 * z0[:self.ndof].reshape((self.ndof, 1))])
        thb = 0

        for i, omg in enumerate(omg_range):
            t0 = time.time()

            # Harmonic Balance
            z0 = self.z0(omg=omg_range[0], f_omg={0: 0})  # None
            if run_hb:
                try:
                    x_hb, res = self.solve_hb(f, omg, z0=z0, full_output=True, method=method, state_space=True)  # 'ls')
                except:
                    x_hb, res = self.solve_hb(f, omg, z0=z0, full_output=True, method='ls', state_space=True)  # 'ls')
                thb = time.time()
                try:
                    z = res.x
                except:
                    z = res[0]

                print(f'Harmonic Balance took {(thb - t0):.1f} seconds to run.')


                try:
                    cost_hb.append(res.cost)
                    z0 = res.x
                except:
                    cost_hb.append(np.linalg.norm(res[1]['fvec']))
                    z0 = res[0]

                rms_hb[:-1, i] = np.array(
                    [np.sqrt(np.sum((x_hb[i, :] - np.mean(x_hb[i, :])) ** 2) / (len(self.t(omg)) - 1)) for i in
                     range(x_hb.shape[0])])

                try:
                    if res[-2] != 1:
                        print(res[-1])
                        rms_hb[-1, i] = 1
                    else:
                        if stability_analysis:
                            fm = self.floquet_multipliers(omg, z, dt_refine=dt_refine)
                            tfm = time.time()
                            print(f'Floquet Multipliers calculation took {(tfm - thb):.1f} seconds to run.')
                            if np.max(np.abs(fm)) > 1:
                                fm_flag[i] = 1
                        else:
                            fm_flag[i] = 0
                except:
                    if not res.success:
                        print(res.message)
                        rms_hb[-1, i] = 1
                    else:
                        if stability_analysis:
                            fm = self.floquet_multipliers(omg, z, dt_refine=dt_refine)
                            tfm = time.time()
                            print(f'Floquet Multipliers calculation took {(tfm - thb):.1f} seconds to run.')
                            if np.max(np.abs(fm)) > 1:
                                fm_flag[i] = 1
                        else:
                            fm_flag[i] = 0

            # Runge-Kutta 4th order

            n_periods = max([1, np.round(tf / (2 * np.pi / omg))])
            tf = n_periods * 2 * np.pi / omg
            dt = 2 * np.pi / omg / (np.round(2 * np.pi / omg / dt_base))
            t_rk = np.arange(0, tf + dt / 2, dt)
            if isinstance(continuation,bool):
                if not continuation:
                    x0 = np.vstack([z0[:self.ndof].reshape((self.ndof,1)),
                                    0 * z0[:self.ndof].reshape((self.ndof,1))])
            else:
                x0 = x_hb[:, 0]
            t1 = time.time()
            if calc:
                x_rk, x0 = self.solve_transient(f, t_rk, omg, x0.reshape((self.ndof*2, 1)),
                                                last_x=True,
                                                dt=dt,
                                                probe_dof=probe_dof,)

            print(
                f'RK4 took {(time.time() - t1):.1f} seconds to run: {(time.time() - t1) / (thb - t0):.1f} times longer.')

            if calc:
                rms_rk[:, i] = np.array(
                    [np.sqrt(
                    np.sum((x_rk[i, int((tf / 2) / dt):] - np.mean(x_rk[i, int((tf / 2) / dt):])) ** 2) / (
                        int((tf / 2) / dt))) for i in range(x_rk.shape[0])])
                pc.append(IntegrationResults.poincare_section(x_rk, t_rk, omg, n_points))

            print(f'Frequency: {omg:.1f} rad/s -> completed.')
            print('---------')

        if save_rms_rk:
            with open(save_rms_rk, 'wb') as file:
                dump(rms_rk, file)

        if save_rms_hb:
            with open(save_rms_hb, 'wb') as file:
                dump({'rms_hb': rms_hb,
                      'fm_flag': fm_flag}, file)

        sl = [False] * (np.max(probe_dof) + 1)
        sl[probe_dof[0]] = True
        fig = go.Figure(data=[go.Scatter(x=omg_range, y=rms_hb[i, :], name=f'DoF {i}- HBM') for i in probe_dof] + \
                             [go.Scatter(x=omg_range, y=rms_rk[i, :], name=f'DoF {i} - RK4') for i in range(x_rk.shape[0])] + \
                             [go.Scatter(x=[omg_range[i] for i in range(len(omg_range)) if rms_hb[-1, i] == 1],
                                         y=[rms_hb[j, i] for i in range(len(omg_range)) if rms_hb[-1, i] == 1],
                                         name='Flagged', mode='markers', marker=dict(color='black'),
                                         showlegend=sl[j],
                                         legendgroup='flag') for j in probe_dof] + \
                             [go.Scatter(x=[omg_range[i] for i in range(len(omg_range)) if fm_flag[i] == 1],
                                         y=[rms_hb[j, i] for i in range(len(omg_range)) if fm_flag[i] == 1],
                                         name='Unstable', mode='markers', marker=dict(color='red',symbol='x'),
                                         showlegend=sl[j],
                                         legendgroup='fm_flag') for j in probe_dof]
                             )
        fig.update_layout(title={'xanchor': 'center',
                                 'x': 0.4,
                                 'font': {'family': 'Arial, bold',
                                          'size': 15},
                                 },
                          yaxis={"gridcolor": "rgb(159, 197, 232)",
                                 "zerolinecolor": "rgb(74, 134, 232)",
                                 },
                          xaxis={'range': [0, np.max(omg_range)],
                                 "gridcolor": "rgb(159, 197, 232)",
                                 "zerolinecolor": "rgb(74, 134, 232)"},
                          xaxis_title='Frequency (rad/s)',
                          yaxis_title='Amplitude',
                          font=dict(family="Calibri, bold",
                                    size=18))
        fig.update_yaxes(type="log")

        fig_cost = go.Figure(data=[go.Scatter(x=omg_range, y=cost_hb, name='Cost HB'),
                               go.Scatter(x=[omg_range[i] for i in range(len(omg_range)) if rms_hb[-1, i] == 1],
                                          y=[cost_hb[i] for i in range(len(omg_range)) if rms_hb[-1, i] == 1],
                                          name='Flagged', mode='markers', marker=dict(color='black'),
                                          legendgroup='flag'),
                               ])
        fig_cost.update_layout(title={'xanchor': 'center',
                                 'x': 0.4,
                                 'font': {'family': 'Arial, bold',
                                          'size': 15},
                                 },
                          yaxis={"gridcolor": "rgb(159, 197, 232)",
                                 "zerolinecolor": "rgb(74, 134, 232)",
                                 },
                          xaxis={'range': [0, np.max(omg_range)],
                                 "gridcolor": "rgb(159, 197, 232)",
                                 "zerolinecolor": "rgb(74, 134, 232)"},
                          xaxis_title='Frequency (rad/s)',
                          yaxis_title='Cost function norm',
                          font=dict(family="Calibri, bold",
                                    size=18))
        fig_cost.update_yaxes(type="log")

        return fig, fig_cost

    def update_speed(self, speed):

        aux = self.rotor.p_damp
        self.rotor.p_damp = self.cp
        self.C = self.rotor.C(speed) + self.rotor.G() * speed
        self.rotor.p_damp = aux

        Z = np.zeros((self.ndof, self.ndof))
        I = np.eye(self.ndof)

        self.A_lin = np.vstack(
            [np.hstack([Z, I]),
             np.hstack([la.solve(-self.M, self.K_lin), la.solve(-self.M, self.C)])])
        self.A = np.vstack(
            [np.hstack([Z, I]),
             np.hstack([la.solve(-self.M, self.K), la.solve(-self.M, self.C)])])

    def plot_smart_frf(self, omg_range, force, tf=300, dt_base=None,
                       continuation=True, method=None, probe_dof=None, dt_refine=None,
                       stability_analysis=True, save_rms=None, downsampling=1, run_hb=True,
                       save_raw_data=False, return_results=False, probe_names=None,
                       gyroscopic=True, unbalance=False):

        rms = np.zeros((len(probe_dof), len(omg_range)))
        pc = []
        cost_hb = []

        if dt_base is None:
            dt_base = self.dt_max()

        if probe_names is None:
            probe_names = [str(a) for a in probe_dof]

        if probe_dof is None:
            probe_dof = [i for i in range(self.ndof)]

        n_points = 50
        z0 = self.z0(omg=omg_range[0], f_omg={0: 0})  # None
        x0 = self.inv_fourier(z0,
                              omg_range[0],
                              state_space=True)[:, 0]

        # x0 = np.vstack([z0[:self.ndof].reshape((self.ndof, 1)),
        #                 0 * z0[:self.ndof].reshape((self.ndof, 1))])
        thb = 0

        data_dict_list = []
        self.update_speed(speed=0)
        if unbalance:
            f_kgm = {}
            f = {}
            for k in force.keys():
                f_kgm[k * 4] = force[k]
                f_kgm[k * 4 + 1] = force[k] * 1.j
        else:
            f = force

        for i, omg in enumerate(omg_range):

            if unbalance:
                for k in f_kgm.keys():
                    f[k] = f_kgm[k] * omg ** 2

            if self.rotor is not None and gyroscopic:
                self.update_speed(speed=omg)


            t0 = time.time()
            if run_hb:
                # Harmonic Balance
                z0 = self.z0(omg=omg_range[0], f_omg={0: 0})  # None

                try:
                    x_hb, res = self.solve_hb(f, omg, z0=z0, full_output=True, method=method, state_space=True)  # 'ls')
                except:
                    x_hb, res = self.solve_hb(f, omg, z0=z0, full_output=True, method='ls', state_space=True)  # 'ls')
                thb = time.time()
                try:
                    z = res.x
                except:
                    z = res[0]

                print(f'Harmonic Balance took {(thb - t0):.1f} seconds to run.')

                try:
                    cost_hb.append(res.cost)
                    z0 = res.x
                except:
                    cost_hb.append(np.linalg.norm(res[1]['fvec']))
                    z0 = res[0]

                rms[:, i] = np.array(
                    [np.sqrt(np.sum((x_hb[i, :] - np.mean(x_hb[i, :])) ** 2) / (len(self.t(omg)) - 1)) for i in
                     probe_dof])

                try:
                    if res[-2] != 1:
                        print(res[-1])
                        calc = True
                    else:
                        if stability_analysis:
                            fm = self.floquet_multipliers(omg, z, dt_refine=dt_refine)
                            tfm = time.time()
                            print(f'Floquet Multipliers calculation took {(tfm - thb):.1f} seconds to run.')
                            if np.max(np.abs(fm)) > 1:
                                calc = True
                            else:
                                calc = False
                        else:
                            calc = False


                except:
                    if not res.success:
                        print(res.message)
                        calc = True
                    else:
                        if stability_analysis:
                            fm = self.floquet_multipliers(omg, z, dt_refine=dt_refine)
                            tfm = time.time()
                            print(f'Floquet Multipliers calculation took {(tfm - thb):.1f} seconds to run.')
                            if np.max(np.abs(fm)) > 1:
                                calc = True
                            else:
                                calc = False
                        else:
                            calc = False
            else:
                calc = True

            # Runge-Kutta 4th order

            n_periods = max([1, np.round(tf / (2 * np.pi / omg))])
            tf2 = n_periods * 2 * np.pi / omg
            dt = 2 * np.pi / omg / (np.round(2 * np.pi / omg / dt_base))
            t_rk = np.arange(0, tf2 + dt / 2, dt)
            t_out = t_rk[::downsampling]
            if isinstance(continuation, bool):
                if not continuation:
                    x0 = self.inv_fourier(z0,
                                          omg_range[0],
                                          state_space=True)[:, 0]
                    # x0 = np.vstack([z0[:self.ndof].reshape((self.ndof,1)),
                    #                 0 * z0[:self.ndof].reshape((self.ndof, 1))])
            else:
                x0 = x_hb[:, 0]

            t1 = time.time()
            if calc:
                x_rk, x0 = self.solve_transient(f, t_rk, omg,
                                                t_out=t_out,
                                                x0=x0.reshape((self.ndof*2, 1)),
                                                last_x=True,
                                                dt=dt,
                                                probe_dof=probe_dof)
                if save_raw_data:
                    with open(f'{save_raw_data}data_rk4 f {f} _ omg-{omg}.pic'.replace(':', '_'), 'wb') as file:
                        dump([x_rk, t_out], file)
                if return_results:
                    data_dict_list.append(dict(time=t_out))
                    for j, p in enumerate(probe_names):
                        data_dict_list[-1][p] = x_rk[j, :]


                rms[:, i] = np.array(
                    [np.sqrt(
                        np.sum((x_rk[i, int(x_rk.shape[1] / 2):] - np.mean(x_rk[i, int(x_rk.shape[1] / 2):])) ** 2) / (
                            int(x_rk.shape[1] / 2))) for i in range(x_rk.shape[0])])

                print(
                    f'RK4 took {(time.time() - t1):.1f} seconds to run: {(time.time() - t1) / (thb - t0):.1f} times longer.')

            print(f'Frequency: {omg:.1f} rad/s -> completed.')
            print('---------')

        self.update_speed(0)

        if save_rms:
            with open(save_rms, 'wb') as file:
                dump(rms, file)


        fig = go.Figure(data=[go.Scatter(x=omg_range, y=rms[i, :], name=f'DoF {p}') for i, p in enumerate(probe_dof)]
                             )
        fig.update_layout(title={'xanchor': 'center',
                                 'x': 0.4,
                                 'font': {'family': 'Arial, bold',
                                          'size': 15},
                                 },
                          yaxis={"gridcolor": "rgb(159, 197, 232)",
                                 "zerolinecolor": "rgb(74, 134, 232)",
                                 },
                          xaxis={'range': [0, np.max(omg_range)],
                                 "gridcolor": "rgb(159, 197, 232)",
                                 "zerolinecolor": "rgb(74, 134, 232)"},
                          xaxis_title='Frequency (rad/s)',
                          yaxis_title='Amplitude',
                          font=dict(family="Calibri, bold",
                                    size=18))
        fig.update_yaxes(type="log")

        if return_results:
            return IntegrationResults(data_dict_list=data_dict_list,
                                      frequency_list=omg_range,
                                      system=self)
        else:
            return fig

    def eq_linear_system(self):

        eq_sys = Sys_NL(self.M, self.K_lin, self.Snl, beta=0, alpha=0, C=self.C)
        eq_sys.dof_nl = self.dof_nl
        eq_sys.base_dof = self.base_dof

        return eq_sys

    @classmethod
    def plot_orbit(cls, x, Ni, color='black'):

        n_probes = x.shape[0] // 2
        fig = go.Figure()
        for i in range(n_probes):
            fig.add_trace(go.Scatter(x=x[i,:],y=x[i+n_probes,:],mode='lines',
                                     line=dict(color=color, width=1),
                                     name=f'probe {i}',legendgroup=f'{i}'))
            fig.add_trace(go.Scatter(x=x[i,::Ni],y=x[i+n_probes,::Ni],
                                     name=f'probe {i}',legendgroup=f'{i}',
                                     mode='markers',showlegend=False,marker=dict(color='red',size=5)))

        avgx = np.mean(x[:n_probes,:])
        x_range = [avgx + (np.min(x[:n_probes,:]) - avgx) * 1.1,
                   avgx + (np.max(x[:n_probes,:]) - avgx) * 1.1]
        v_range = [np.min(x[n_probes:, :] * 1.1), np.max(x[n_probes:, :] * 1.1)]
        fig.update_layout(xaxis=dict(title='X',
                                     range=x_range),
                          yaxis=dict(title='dX/dt',
                                     range=v_range))

        return fig

    @classmethod
    def plot_3d_orbit(cls, x, Ni, color='black'):

        n_probes = x.shape[0] // 2
        fig = go.Figure()
        for i in range(n_probes):
            fig.add_trace(go.Scatter3d(x=x[i, :], y=x[i + n_probes, :], z=np.arange(len(x[i,:])),mode='lines',
                                     name=f'probe {i}', legendgroup=f'{i}',line=dict(color=color, width=2)))
            fig.add_trace(go.Scatter3d(x=x[i, ::Ni], y=x[i + n_probes, ::Ni], z=np.arange(0, len(x[i, :]), Ni),
                                     name=f'probe {i}', legendgroup=f'{i}',
                                     mode='markers', showlegend=False, marker=dict(color='red', size=2)))

        x_range = [np.min(x[:n_probes, :] * 1.1), np.max(x[:n_probes, :] * 1.1)]
        v_range = [np.min(x[n_probes:, :] * 1.1), np.max(x[n_probes:, :] * 1.1)]
        fig.update_layout(xaxis=dict(title='X',
                                     range=x_range),
                          yaxis=dict(title='dX/dt',
                                     range=v_range))

        return fig
#
# class RotorNL(SysNL):
#
#     def __init__(self,rotor,
#                  n_harm=10,nu=1,N=2,cp=1e-4,C=None,
#                  x_eq0=None, x_eq1=None, sp=0, n_harm=10, nu=1, N=1, cp=1e-4):
#
#         self.rotor = rotor
#
#         self.M = self.rotor.M()
#         self.cp = cp
#
#         # if C is None:
#         #     self.C = K * cp
#         # else:
#         #     self.C = C
#         # self.K = K
#
#         self.Snl = Snl
#         self.beta = beta
#         self.alpha = alpha
#         self.n_harm = n_harm
#         self.nu = nu
#         self.N = N
#         if self.alpha != 0:
#             self.x_eq = np.sqrt(-self.beta / self.alpha)
#         else:
#             self.x_eq = 0
#         self.ndof = len(self.M)
#         # self.K_lin = self.K - self.Snl * 2 * self.beta
#
#         self.dof_nl = []
#         self.base_dof = []
#         for i in range(len(self.Snl)):
#             if self.Snl[i, i] != 0:
#                 if self.K(0)[i, i] == 0:
#                     self.dof_nl.append(i)
#                 else:
#                     self.base_dof.append(i)
#
#         Z = np.zeros((self.ndof, self.ndof))
#         I = np.eye(self.ndof)
#
#         self.Minv = np.vstack([np.hstack([I, Z]),
#                                np.hstack([Z, la.inv(M)])])
#         # self.A_lin = np.vstack(
#         #     [np.hstack([Z, I]),
#         #      np.hstack([la.solve(-self.M, self.K_lin), la.solve(-self.M, self.C)])])
#         # self.A = np.vstack(
#         #     [np.hstack([Z, I]),
#         #      np.hstack([la.solve(-self.M, self.K), la.solve(-self.M, self.C)])])
#         self.full_orbit = False
#
#
#
#     def C(self, speed):
#
#
#     def K(self, speed):
#
#     def K_lin(self, speed):
#
#     def A_lin(self, speed):
#
#     def A(self, speed):
