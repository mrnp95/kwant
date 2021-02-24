import kwant
from kwant import kpm

# plotting, etc. tools
import numpy as np
import holoviews as hv

hv.notebook_extension()


def make_system(a, t, W, L, H, dis_amp):
    lat = kwant.lattice.cubic(a)
    #     lat = kwant.lattice.general([(a, 0, 0), (0, a, 0), (0, 0, a)])
    #     syst = kwant.Builder(kwant.TranslationalSymmetry(lat.vec((L, 0, 0)), lat.vec((0, W, 0))))
    syst = kwant.Builder()
    # Definign spatial dependent disorder term
    # def disorder(site, dis_amp):
    #     (x, y, z) = site.pos
    #     if (L - L/2) / 2 < x < (L + L/2) / 2:
    #         return np.random.uniform(low = - dis_amp / 2., high = dis_amp / 2.)
    #     else:
    #         return 0.

    #     def onsite(site, dis_amp):
    #         return disorder(site, dis_amp)

    # Adding onsite and hopping interactions to the lattice
    for x in range(L):
        for y in range(W):
            for z in range(H):
                # if (L - L / 2) / 2 < x < (L + L / 2) / 2 and (W - W / 2) / 2 < y < (W + W / 2) / 2 and \
                #         (H - H / 2) / 2 < z < (H + H / 2) / 2:
                #     syst[(lat(x, y, z))] = np.random.uniform(low=- dis_amp / 2., high=dis_amp / 2.)
                # else:
                #     syst[(lat(x, y, z))] = 0.
                syst[(lat(x, y, z))] = np.random.uniform(low=- dis_amp / 2., high=dis_amp / 2.)

    for x in range(L):
        for z in range(H):
            syst[(lat(x, 0, z), lat(x, W-1, z))] = -t

    for y in range(W):
        for z in range(H):
            syst[(lat(0, y, z), lat(L - 1, y, z))] = -t

    for x in range(L):
        for y in range(W):
            syst[(lat(x, y, 0), lat(x, y, H - 1))] = -t
    syst[lat.neighbors()] = -t

    # syst = kwant.wraparound.wraparound(syst)
    return syst


def plot_spectrum(spectrum):
    """
    plot the spectrum, and if the densities are imaginary, plot the imaginary part as well.
    """
    e, d = spectrum()
    d = np.real_if_close(d)
    the_plot = hv.Curve((e, d.real))
    if type(d) is complex:
        the_plot = the_plot * hv.Curve((e, d.imag))

    the_plot.redim(x='$e$', y='$\\rho$')
    hv.save(the_plot, 'the_plot.png')

    return the_plot


def main():
    a = 1
    t = 1.
    W = 50
    L = 50
    H = 50
    dis_amp = 3 * t

    syst = make_system(a, t, W, L, H, dis_amp)

    # Check that the system looks as intended.
    # kwant.plot(syst, site_lw=0.01, site_size=0.1, hop_lw=0.05)

    # Finalize the system.
    fsyst = syst.finalized()

    spectrum = kpm.SpectralDensity(fsyst)

    # spectrum.add_moments(num_moments=2048)
    # spectrum.add_vectors(num_vectors=10)

    plot_spectrum(spectrum)


# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()
    print("Done!!")


