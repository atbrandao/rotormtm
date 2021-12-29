import numpy as np
import scipy.linalg as la
from scipy.optimize import newton, least_squares, fsolve
import plotly.graph_objects as go
#
# m0 = 1
# m1 = 0.5
#
# w0 = 10
#
# x_eq = 0.1
# w1 = 10
#
# k0 = w0**2 * m0
# beta = -1/2 * w1**2 * m1
# alpha = -beta / x_eq**2
#
# M = np.array([[m0 , 0],
#               [0 , m1]])
#
# K_lin = np.array([[k0 , 0],
#                   [0 , 0]])
#
# Snl = np.array(np.array([[1 , -1],
#                          [-1 , 1]]))
#
# K = np.array([[k0 + beta , -beta],
#               [-beta , beta]])
#
# cp = 1e-4
# C = cp * K


def poincare_section(x, t, omg, n_points=10):
    dt = t[1] - t[0]
    T = 2 * np.pi / omg
    N = int(np.ceil(T / dt))
    dt2 = T / N
    t2 = np.arange(t[0], t[-1], dt2)

    if len(x.shape) == 1:
        x2 = np.interp(t2, t, x)
        pc = x2[::N]
    else:
        x2 = np.zeros((x.shape[0],len(t2)))
        for i in range(len(x[:, 1])):
            x2[i, :] = np.interp(t2, t, x[i, :])
        pc = x2[:, ::N]

    if n_points < pc.shape[1]:
        pc = pc[-n_points:]

    return pc

class Sys_NL:

    def __init__(self,M,K,Snl,beta,alpha,n_harm=10,nu=1,N=2,cp=1e-4):

        self.M = M
        self.cp = cp
        self.C = K * cp
        self.K = K
        self.Snl = Snl
        self.beta = beta
        self.alpha = alpha
        self.n_harm = n_harm
        self.nu = nu
        self.N = N
        self.x_eq = np.sqrt(-self.beta/self.alpha)
        self.ndof = len(self.K)
        self.K_lin = self.K - self.Snl * 2 * self.beta

        self.dof_nl = []
        for i in range(len(self.Snl)):
            if self.K[i, i] == 0 and self.Snl[i, i] != 0:
                self.dof_nl.append(i)

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


    def H(self,omg):

        M = self.M
        A = self.A_lin

        N = len(M)

        H = np.linalg.inv(1.j * omg * np.eye(2 * N) - A)

        return H

    def A_hb(self,omg):

        Z = np.zeros(self.K.shape)
        Z2 = np.vstack([Z,Z])

        A_out = np.vstack([np.hstack([self.K]+[Z]*(2*self.n_harm))] + \
                          [np.hstack([Z2]*(1+2*i)+[np.vstack([np.hstack([self.K-((i+1)*(omg/self.nu))**2*self.M, -(i+1)*omg/self.nu*self.C]),
                                                               np.hstack([(i+1)*omg/self.nu*self.C, self.K-((i+1)*(omg/self.nu))**2*self.M])])]+ \
                                     [Z2]*(2*self.n_harm-2*(i+1))) for i in range(self.n_harm)])

        return A_out

    def t(self,omg):

        w0 = omg / self.nu
        t = np.linspace(0, 2 * np.pi / w0, self.N * self.n_harm)
        t = t.reshape((len(t), 1))

        return t

    def gamma(self, omg):

        id_n = np.eye(self.ndof)
        w0 = omg/self.nu
        wt = w0 * self.t(omg)
        # print(wt)
        gamma = np.hstack([np.kron(id_n,np.cos(0*wt))] + [np.hstack([np.kron(id_n,np.sin((i+1)*wt)),
                                                                     np.kron(id_n,np.cos((i+1)*wt))]) for i in range(self.n_harm)])

        return gamma

    def f_nl(self,x):

        id_N = np.eye(self.N*self.n_harm)
        Snl2 = np.vstack(
            [np.hstack([id_N*self.Snl[i,j] for j in range(len(self.Snl))]) for i in range(len(self.Snl))])

        f_nl = self.beta * Snl2 @ x + self.alpha * (Snl2 @ x)**3

        return f_nl

    def df_dx(self,x):

        id_N = np.eye(self.N * self.n_harm)
        Snl2 = np.vstack(
            [np.hstack([id_N * self.Snl[i, j] for j in range(len(self.Snl))]) for i in range(len(self.Snl))])

        # Pela definição: df_dx = - d(f_nl)_dx
        df_dx = - (self.beta * Snl2 + 3 * self.alpha * ((Snl2 @ x)**2 * np.eye(len(x))) @ Snl2)

        return df_dx

    def h(self,z,*args):

        z = z.reshape((len(z),1))

        omg = args[0]
        if len(args) == 1:
            f_omg = {}
        else:
            f_omg = args[1]

        b_f = np.zeros((self.ndof*(2*self.n_harm+1),1))
        for f in f_omg:
            b_f[self.ndof + 2 * (self.nu - 1) * self.ndof + f] = np.imag(f_omg[f])
            b_f[2 * self.ndof + 2 * (self.nu - 1) * self.ndof + f] = np.real(f_omg[f])
        # print(b_f)
        g = self.gamma(omg)
        x = g @ z
        b = b_f - la.pinv(g) @ self.f_nl(x)
        h = self.A_hb(omg) @ z - b

        return h.reshape(len(h))

    def dh_dz(self,z,*args):

        omg = args[0]

        g = self.gamma(omg)
        x = g @ z
        db_dz = la.pinv(g) @ self.df_dx(x) @ g
        dh_dz = self.A_hb(omg) - db_dz

        return dh_dz

    def z0(self,omg,f_omg):

        H = self.H(omg)

        F = np.zeros((2*self.ndof,1))
        for f in f_omg:
            F[self.ndof+f] = f_omg[f]
        x = H @ self.Minv @ F

        z0 = np.zeros((self.ndof*(2*self.n_harm+1),1))
        z0[self.dof_nl] = self.x_eq
        for i, y in enumerate(x[:self.ndof]):
            z0[self.ndof + 2 * (self.nu - 1) * self.ndof + i] = np.imag(y)
            z0[2 * self.ndof + 2 * (self.nu - 1) * self.ndof + i] = np.real(y)

        return z0.reshape(len(z0))

    def solve_hb(self, f, omg, z0=None, full_output=False, method=None):

        if z0 is None:
            z0 = self.z0(omg=omg, f_omg=f)
        if method is None:
            res = fsolve(func=self.h, x0=z0, fprime=self.dh_dz, args=(omg, f), full_output=full_output)
        else:
            res = least_squares(fun=self.h, x0=z0, jac=self.dh_dz, args=(omg, f), xtol=1e-9)

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
        x = self.gamma(omg) @ root.reshape((len(root), 1))
        x = x.reshape(len(x))
        x = x.reshape((self.ndof,len(x)//self.ndof))[:,:-1]
        if full_output:
            return x, res
        else:
            return x

    def RK4_NL(self, B, x, dt):

        x = np.reshape(x, (len(x), 1))
        Z = np.zeros((self.ndof, self.ndof))

        A = self.A
        alpha = self.alpha
        beta = self.beta
        Minv = self.Minv
        Snl2 = np.vstack([np.hstack([Z, Z]),
                          np.hstack([-self.Snl, Z])])

        aux1 = np.zeros((len(B), 1)).astype(type(B[0, 0]))
        aux1[:, 0] = B[:, 0]
        aux2 = np.zeros((len(B), 1)).astype(type(B[0, 0]))
        aux2[:, 0] = B[:, 1]
        aux3 = np.zeros((len(B), 1)).astype(type(B[0, 0]))
        aux3[:, 0] = B[:, 2]

        k1 = A @ x + aux1 + Minv @ (beta * Snl2 @ x + alpha * (Snl2 @ x) ** 3)
        x1 = x + k1 * dt / 2
        k2 = A @ x1 + aux2 + Minv @ (beta * Snl2 @ x1 + alpha * (Snl2 @ x1) ** 3)
        x2 = x + k2 * dt / 2
        k3 = A @ x2 + aux2 + Minv @ (beta * Snl2 @ x2 + alpha * (Snl2 @ x2) ** 3)
        x3 = x + k3 * dt
        k4 = A @ x3 + aux3 + Minv @ (beta * Snl2 @ x3 + alpha * (Snl2 @ x3) ** 3)

        return np.reshape(x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), (len(x), 1))

    def solve_transient(self, f, t, omg, x0):

        x = np.zeros((2*self.ndof,len(t)))
        x[:,0] = x0[:,0]

        for i in range(1,len(t)):
            dt = t[i] - t[i-1]

            F = np.zeros((self.ndof, 3))
            for n in f:
                F[n, 0] = np.real(f[n]) * np.cos(omg * t[i-1]) + np.imag(f[n]) * np.sin(omg * t[i-1])
                F[n, 1] = np.real(f[n]) * np.cos(omg * t[i-1]+dt/2) + np.imag(f[n]) * np.sin(omg * t[i-1]+dt/2)
                F[n, 2] = np.real(f[n]) * np.cos(omg * t[i-1]+dt) + np.imag(f[n]) * np.sin(omg * t[i-1]+dt)

            B_aux = self.Minv @ np.vstack([0*F, F])
            # print(B_aux)
            x[:,i] = self.RK4_NL(B_aux,x[:,i-1],dt)[:,0]

        return x



#
# n_harm = 40
# nu = 4
# N = 8 # sinal no tempo terá comprimento N*n_harm
#
# S = Sys(M=M,C=C,K=K_lin,Snl=Snl,beta=beta,alpha=alpha,n_harm=n_harm,nu=nu,N=N)
#
# omg = 9
#
# f = {0: 1e-2}

# print(S.gamma(omg).shape)
# print(S.z0(omg=omg,f_omg=f,dof_nl=[1]))

# root = newton(func=S.h, x0=S.z0(omg=omg,f_omg=f,dof_nl=[1]), fprime=S.dh_dz, args=(omg,f),maxiter=200)
# x1_rms = []
# x2_rms = []
# x1_rms_rk = []
# x2_rms_rk = []
# omg_arr = np.arange(1,20,.1)
# for omg in omg_arr:
#     print(omg)
#     root = fsolve(func=S.h, x0=S.z0(omg=omg,f_omg=f,dof_nl=[1]), fprime=S.dh_dz, args=(omg,f))
#     x = S.gamma(omg) @ root.reshape((len(root), 1))
#     x = x.reshape(len(x))
#     x1_rms.append(np.sqrt(np.sum(x[:len(x)//S.ndof]**2))/S.t(omg)[-1,0])
#     x2_rms.append(np.sqrt(np.sum((x[len(x) // S.ndof:] - x_eq) ** 2)) / S.t(omg)[-1,0])
#     tf = 100
#     dt = 0.01
#     x = S.solve_transient(f,np.arange(0,tf,dt),omg,np.array([[0],[x_eq],[0],[0]]))
#     x1_rms_rk.append(np.sqrt(np.sum((x[0,int((tf/2)/dt):] - np.mean(x[0,int((tf/2)/dt):])) ** 2)) / (tf/2))
#     x2_rms_rk.append(np.sqrt(np.sum((x[1,int((tf/2)/dt):] - np.mean(x[1,int((tf/2)/dt):])) ** 2)) / (tf/2))
#
# fig = go.Figure(data=[go.Scatter(x=omg_arr,y=np.log10(np.array(x1_rms)),name='DoF 1'),
#                       go.Scatter(x=omg_arr,y=np.log10(np.array(x2_rms)),name='DoF 2')])
# fig.write_html('FRF.html')
#
# fig = go.Figure(data=[go.Scatter(x=omg_arr,y=np.log10(np.array(x1_rms)),name='DoF 1'),
#                       go.Scatter(x=omg_arr,y=np.log10(np.array(x2_rms)),name='DoF 2'),
#                       go.Scatter(x=omg_arr,y=np.log10(np.array(x1_rms_rk)),name='DoF 1 RK'),
#                       go.Scatter(x=omg_arr,y=np.log10(np.array(x2_rms_rk)),name='DoF 2 RK')])
# fig.write_html('FRF_rk_comp.html')
#
# # root = least_squares(fun=S.h, x0=S.z0(omg=omg,f_omg=f,dof_nl=[1]), jac=S.dh_dz, args=(omg,f))
#
# print(root)
# print(S.z0(omg=omg,f_omg=f,dof_nl=[1]))
#
# x1 = S.gamma(omg) @ root.reshape((len(root), 1))
# x1 = x1.reshape(len(x1))
# fig = go.Figure(data=[go.Scatter(x=S.t(omg).reshape(len(x1)//S.ndof),y=x1[:len(x1)//S.ndof]),
#                       go.Scatter(x=S.t(omg).reshape(len(x1)//S.ndof),y=x1[len(x1)//S.ndof:]-x_eq)])
# fig.write_html('waveform.html')
#
# fig = go.Figure(data=[go.Scatter(x=np.arange(0,tf,dt),y=x[0,:]),
#                       go.Scatter(x=np.arange(0,tf,dt),y=x[1,:]-x_eq)])
# fig.write_html('waveform_rk.html')
#
# print('')