import emcee
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from multiprocessing import Pool
from lenstronomy.Cosmo.micro_lensing import einstein_radius
from parameter_manager import ParameterManager
import astropy.units as u


with open('ellip_table_tu.npy', 'rb') as f:
    theta_E0 = np.load(f)
    u0 = np.load(f)
    table = np.load(f)

interpolator = RegularGridInterpolator((u0, theta_E0), table, bounds_error=False, fill_value=None)

pc2km = u.pc.to(u.km)
sec2day = u.s.to(u.day)
arcsec2rad = u.arcsec.to(u.rad)

def get_data(t, t0, M, v, d_l, d_s, u_min):
    """
    get curve at a given parameter.
    unit:
        t, t0: day
        M: M_sol,
        v: km/s,
        d_l, d_s: pc
    """
    theta_E = einstein_radius(M, d_l, d_s)
    t_E = theta_E*arcsec2rad*d_l*pc2km/v*sec2day
    u = np.sqrt(u_min*u_min + ((t-t0)/t_E)**2)
    new_x = np.stack((u, np.ones_like(u)*theta_E), axis=-1)
    y = interpolator(new_x)
    if np.isnan(y).any():
        print(t, t0, M, v, d_l, d_s, u_min)
        import sys
        sys.exit(1)
    return y

t_obs, e_obs, err_obs = np.load('mock2.npy')

def ln_like(**kwargs):
    res = get_data(t_obs, **kwargs)
    chi_2 = (((res - e_obs)/err_obs)**2).sum()
    return -chi_2


if __name__ == '__main__':
    Nwalkers = 40
    Nstep = 5000
    Npro = 40

    pool = Pool(Npro)


    manager = ParameterManager.from_yaml('config.yaml', ln_like, True)
    init_state = manager.random_init_state(Nwalkers)

    Ndim = manager.ndim
    backend = emcee.backends.HDFBackend('backend_free_dl.h5')
    backend.reset(Nwalkers, Ndim)

    sampler = emcee.EnsembleSampler(Nwalkers, Ndim, manager.lnlikely, pool, backend=backend)
    sampler.run_mcmc(init_state, Nstep, progress=True)
