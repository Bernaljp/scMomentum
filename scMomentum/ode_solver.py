import numpy as np
import scipy.integrate as solve
import sdeint

class ODESolver:
    def __init__(self, W, gamma, I, k, n, solver=None, clipping=False):
        self.W = np.array(W)
        self.gamma = np.array(gamma)
        self.I = np.array(I)
        self.k = np.array(k)
        self.n = n
        self.clip = clipping
        self.solver = solve.odeint if solver is None else solver

    @staticmethod
    def sigmoid(x, k, n):
        # Avoid division by zero and ensure numerical stability
        x_safe = np.maximum(x, 1e-5)
        return x_safe**n / (x_safe**n + k**n)

    def ode_system(self):
        def f(y, t):
            s = self.sigmoid(y, self.k, self.n)
            dydt = self.W @ s - self.gamma * y + self.I
            return np.where(np.logical_and(y <= 0, dydt < 0), 0, dydt) if self.clip else dydt
        return f

    def simulate_system(self, x0, tf, n_steps, noise=0):
        t = np.linspace(0, tf, n_steps)
        f = self.ode_system()
        
        if noise == 0:
            sol = self.solver(f, x0, t)
        else:
            def G(x, t): return noise * np.eye(len(x0))
            sol = sdeint.itoint(f, G, x0, t)

        return sol.T if isinstance(self.W, np.ndarray) else sol.T[0]
