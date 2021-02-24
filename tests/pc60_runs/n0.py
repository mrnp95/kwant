import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


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
    dt = 2 * np.pi / (omega * 50000)
    g = g_K(A, omega, k, t, tau)
    g_dt = g_K(A, omega, k, t + dt, tau)
    d_Phi = 1j * (np.log(g_dt / np.abs(g_dt)) - np.log(g / np.abs(g))) / dt

    return d_Phi * conn


def zak_Phase(A, omega, k, tau):
    """
    Zak phase.
    """
    dt = 2 * np.pi / (omega * 50000)
    phase = 0.
    for t in np.linspace(0., 2 * np.pi / omega, 50000):
        b_conn = berry_conn(A, omega, k, t, tau)
        phase = phase - dt * b_conn[0, 0]
    phase = phase * omega
    return phase


def eps_WSL_n(A, k, tau, omega, n, zak_phase, sign):
    dt = 2 * np.pi / (omega * 50000)
    e_bar = 0.0
    for t in np.linspace(0., 2 * np.pi / omega, 50000):
        e_bar = e_bar - dt * sign * np.abs(g_K(A, omega, k, t, tau))
    e_bar = e_bar * omega / (2 * np.pi)
    return e_bar + omega * (n + zak_phase / (2 * np.pi))


def plot_surface_3d(n, nx, ny, quantity, name):
    """
    Make 3D surface plot of BZ.
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.linspace(-np.pi, np.pi, nx)
    Y = np.linspace(-np.pi, np.pi, ny)
    X, Y = np.meshgrid(X, Y)
    Z = np.ones((nx, ny))
    count = 0
    for i in range(nx):
        for j in range(ny):
            Z[i, j] = Z[i, j] * quantity[count]
            count += 1

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(np.min(quantity) - 0.02, np.max(quantity) + 0.02)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title(name + "_n=" + str(n))
    plt.savefig('./' + name + "_n_" + str(n) + ".pdf")
    #plt.show()


A = 0.8
omega = 0.005
tau = 0.01
n = 0.0


energy = []
kx = []
ky = []
zak = []
zak_real = []
nx = 10
ny = 10

for i in np.linspace(-np.pi, np.pi, nx):
    for j in np.linspace(-np.pi, np.pi, ny):
        k = np.array([i, j])
        z_phase = zak_Phase(A, omega, k, tau)
        zak.append(z_phase)
        zak_real.append(np.real(z_phase))
        e = eps_WSL_n(A, k, tau, omega, n, np.real(z_phase), +1.)
        energy.append(e)
        print(e)
        if np.abs(e) <= 0.005:
            kx.append(i)
            ky.append(j)
np.savetxt('./out/zak.txt', zak)
np.savetxt('./out/e.txt', energy)
np.savetxt('./out/kx.txt', kx)
np.savetxt('./out/ky.txt', ky)

# Plotting

plot_surface_3d(n, nx, ny, energy, "Energy")
plot_surface_3d(n, nx, ny, zak_real, "Zak_real")

print("Delta_E= ",np.max(energy)-np.min(energy))
print("Delta_Z= ",np.max(zak_real)-np.min(zak_real))