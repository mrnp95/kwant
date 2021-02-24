import numpy as np


def vector_poten(A0, omega, t):
    """
    Vector potential.
    A0 = c*E_0 / omega
    """
    a = np.zeros((2))
    a[0] = A0 * np.cos(omega * t)
    a[1] = -A0 * np.sin(omega * t)
    return a


def k_PS(A, omega, k, t):
    """
    Peierls substitution!
    A = e*E_0 / omega
    """
    k[0] = k[0] - A * np.cos(omega * t)
    k[1] = k[1] + A * np.sin(omega * t)
    return k


def g_K(A, omega, k, t, tau):
    """
    Diagonal elements of the TS Hamiltonian.
    """
    k_ps = k_PS(A, omega, k, t)
    c1 = np.array([-np.sqrt(3) / 2, -0.5])
    c2 = np.array([np.sqrt(3) / 2, -0.5])
    c3 = np.array([0., 1.])
    return -tau * (np.exp(1j * np.dot(k_ps, c1)) + np.exp(1j * np.dot(k_ps, c2)) + np.exp(1j * np.dot(k_ps, c3)))


def Epsilon_K(g):
    e = np.zeros((2, 2))
    e[0, 0] = np.abs(g)
    e[1, 1] = -np.abs(g)
    return e


def berry_conn(A, omega, k, t, tau):
    """
    Berry conenction.
    """
    conn = 0.5 * np.ones((2, 2))
    dt = 2 * np.pi / (omega * 2000)
    g = g_K(A, omega, k, t, tau)
    g_dt = g_K(A, omega, k, t + dt, tau)
    d_Phi = 1j * (np.log(g_dt / np.abs(g_dt)) - np.log(g / np.abs(g))) / dt

    return d_Phi * conn


def zak_Phase(A, omega, k, tau):
    """
    Zak phase.
    """
    dt = 2 * np.pi / (omega * 2000)
    phase = 0.
    for t in np.linspace(0., 2 * np.pi / omega, 2000):
        b_conn = berry_conn(A, omega, k, t, tau)
        phase = phase - dt * b_conn[0, 0]
    phase = phase * omega
    return phase


def eps_WSL_n(A, k, tau, omega, n, zak_phase, sign):
    dt = 2 * np.pi / (omega * 2000)
    e_bar = 0.0
    for t in np.linspace(0., 2 * np.pi / omega, 2000):
        e_bar = e_bar - dt * sign * np.abs(g_K(A, omega, k, t, tau))
    e_bar = e_bar * omega / (2 * np.pi)
    return e_bar + omega * (n + zak_phase / (2 * np.pi))

A = 0.8
omega = 0.05
tau = 1.0
n = 3


energy = []
kx = []
ky = []
zak = []

for i in np.linspace(-np.pi, np.pi, 300):
    for j in np.linspace(-np.pi, np.pi, 300):
        k = np.array([i, j])
        z_phase = np.abs(zak_Phase(A, omega, k, tau))
        zak.append(z_phase)
        e = eps_WSL_n(A, k, tau, omega, n, 0., +1)
        energy.append(e)
        print(e)
        if np.abs(e) <= 0.001:
            kx.append(i)
            ky.append(j)
np.savetxt('./zak.txt', zak)
np.savetxt('./e.txt', energy)
np.savetxt('./kx.txt', kx)
np.savetxt('./ky.txt', ky)