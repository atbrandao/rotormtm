import numpy as np
import scipy.linalg as la
from scipy.optimize import newton, least_squares, fsolve
import plotly.graph_objects as go

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

cp = 1e-4
C = cp * K

class Sys:

    def __init__(self,M,C,K,Snl,beta,alpha,n_harm=10,nu=1,N=2):

        self.M = M
        self.C = C
        self.K = K
        self.Snl = Snl
        self.beta = beta
        self.alpha = alpha
        self.n_harm = n_harm
        self.nu = nu
        self.N = N
        self.x_eq = np.sqrt(-self.beta/self.alpha)
        self.ndof = len(self.K)


    def A_lin(self):

        M = self.M
        K = self.K - self.Snl * 2 * self.beta
        C = self.C

        N = len(M)

        Z = np.zeros((N, N))
        I = np.eye(N)

        A = np.vstack(
            [np.hstack([Z, I]),
             np.hstack([la.solve(-M, K), la.solve(-M, C)])])



        return A

    def calc_H(self,omg):

        M = self.M
        A = self.A_lin()

        N = len(M)
        Z = np.zeros((N, N))
        M_inv = np.vstack([np.hstack([Z, Z]),
                           np.hstack([Z, la.inv(M)])])

        H = np.linalg.inv(1.j * omg * np.eye(2 * N) - A)

        return H, M_inv

    def A_hb(self,omg):

        Z = np.zeros(self.K.shape)
        Z2 = np.vstack([Z,Z])

        A_out = np.vstack([np.hstack([self.K]+[Z]*(2*self.n_harm))] + \
                          [np.hstack([Z2]*(1+2*i)+[np.vstack([np.hstack([self.K-(i+1)*(omg/self.nu)**2*self.M, -(i+1)*omg/self.nu*self.C]),
                                                               np.hstack([(i+1)*omg/self.nu*self.C, self.K-(i+1)*(omg/self.nu)**2*self.M])])]+ \
                                     [Z2]*(2*self.n_harm-2*(i+1))) for i in range(self.n_harm)])

        return A_out

    def t(self,omg):

        w0 = omg / self.nu
        wmax = self.n_harm * w0
        t = np.linspace(0, 2 * np.pi / w0, self.N * self.n_harm)
        t = t.reshape((len(t), 1)).reshape((len(t), 1))

        return t

    def gamma(self, omg):

        id_n = np.eye(self.ndof)
        w0 = omg/self.nu
        wt = w0 * self.t(omg)
        # print(wt)
        gamma = np.hstack([np.kron(id_n,np.cos(0*wt))] + [np.hstack([np.kron(id_n,np.sin((i+1)*wt)),
                                                                     np.kron(id_n,np.cos((i+1)*wt))]) for i in range(n_harm)])

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
        df_dx = - (self.beta * Snl2 + 3 * self.alpha * (Snl2 @ x)**2 * np.eye(len(x)))

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

    def z0(self,omg,f_omg,dof_nl=[]):

        H, Minv = self.calc_H(omg)

        F = np.zeros((2*self.ndof,1))
        for f in f_omg:
            F[self.ndof+f] = f_omg[f]
        x = H @ Minv @ F

        z0 = np.zeros((self.ndof*(2*self.n_harm+1),1))
        z0[dof_nl] = self.x_eq
        for i, y in enumerate(x[:self.ndof]):
            z0[self.ndof + i] = np.imag(y)
            z0[2 * self.ndof + i] = np.real(y)

        return z0.reshape(len(z0))


n_harm = 20
nu = 1
N = 4 # sinal no tempo terá comprimento N*n_harm

S = Sys(M=M,C=C,K=K_lin,Snl=Snl,beta=beta,alpha=alpha,n_harm=n_harm,nu=nu,N=N)

omg = 9

f = {0: 2e-2}

# print(S.gamma(omg).shape)
# print(S.z0(omg=omg,f_omg=f,dof_nl=[1]))

# root = newton(func=S.h, x0=S.z0(omg=omg,f_omg=f,dof_nl=[1]), fprime=None, args=(omg,f),maxiter=200)
x1_rms = []
x2_rms = []
omg_arr = np.arange(1,20,.1)
for omg in omg_arr:
    print(omg)
    root = fsolve(func=S.h, x0=S.z0(omg=omg,f_omg=f,dof_nl=[1]), fprime=S.dh_dz, args=(omg,f))
    x = S.gamma(omg) @ root.reshape((len(root), 1))
    x = x.reshape(len(x))
    x1_rms.append(np.sqrt(np.sum(x[:len(x)//S.ndof]**2))/(len(x)//S.ndof))
    x2_rms.append(np.sqrt(np.sum((x[len(x) // S.ndof:] - x_eq) ** 2)) / (len(x) // S.ndof))

fig = go.Figure(data=[go.Scatter(x=omg_arr,y=np.log10(np.array(x1_rms)),name='DoF 1'),
                      go.Scatter(x=omg_arr,y=np.log10(np.array(x2_rms)),name='DoF 2')])
fig.write_html('FRF.html')

# root = least_squares(fun=S.h, x0=S.z0(omg=omg,f_omg=f,dof_nl=[1]), jac=S.dh_dz, args=(omg,f))

print(root)
print(S.z0(omg=omg,f_omg=f,dof_nl=[1]))

x = S.gamma(omg) @ root.reshape((len(root), 1))
x = x.reshape(len(x))
fig = go.Figure(data=[go.Scatter(x=S.t(omg).reshape(len(x)//S.ndof),y=x[:len(x)//S.ndof]),
                      go.Scatter(x=S.t(omg).reshape(len(x)//S.ndof),y=x[len(x)//S.ndof:]-x_eq)])
fig.write_html('teste.html')