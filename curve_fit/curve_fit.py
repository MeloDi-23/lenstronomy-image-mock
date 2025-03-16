""""
Uses the old interpolation table.
"""

import emcee
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from multiprocessing import Pool
table = np.load('ellip_table.npy')

with open('table_arguments.npy', 'rb') as f:
    t = np.load(f)
    M = np.load(f)
    v = np.load(f)


interpolator = RegularGridInterpolator((M, v, t), table, bounds_error=False)
def get_data(t0, M0, v0):
    abs_t = np.abs(t0)          # assume t0 is from -t_E to t_E, the curve is symetrical.
    new_x = np.stack((np.ones_like(t0)*M0, np.ones_like(t0)*v0, abs_t), axis=-1)
    return interpolator(new_x)

t_obs, e_obs, err_obs = np.load('mock2.npy')
def ln_prior(arg):
    if M[0] < arg[0] < M[-1] and v[0] < arg[1] < v[-1]:
        return 0
    return -np.inf

def ln_like(arg):
    if np.isneginf(ln_prior(arg)):
        return -np.inf
    res = get_data(t_obs, *arg)
    chi_2 = (((res - e_obs)/err_obs)**2).sum()
    return -chi_2


Nwalkers = 40
Nstep = 5000
Npro = 40
start_point = np.c_[
    np.random.uniform(M[0]+1e-5, M[-1]-1e-5, Nwalkers), 
    np.random.uniform(v[0]+1e-5, v[-1]-1e-5, Nwalkers)]

pool = Pool(Npro)

backend = emcee.backends.HDFBackend('backend.h5')
backend.reset(Nwalkers, 2)
sampler = emcee.EnsembleSampler(Nwalkers, 2, ln_like, pool, backend=backend)
sampler.run_mcmc(start_point, Nstep, progress=True)
