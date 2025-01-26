import numpy as np
import matplotlib.pyplot as plt
from lenstronomy.Cosmo.micro_lensing import einstein_radius
from calc_ellipticity_webb import ellipticity,ellipticity_image_mock
from astropy import units as u
deltaPix = 0.03
psf = 0.08

u0 = np.linspace(0.05, 2.0, 200)
theta_E0 = np.linspace(0.01, 0.05, 100)

uu, tt = np.meshgrid(u0, theta_E0, indexing='ij')
ellips = np.zeros((len(u0), len(theta_E0)))
err = np.zeros((len(u0), len(theta_E0)))

        
import tqdm
with tqdm.tqdm(total=len(u0)*len(theta_E0)) as tbar:
    for i in range(len(u0)):
        for j in range(len(theta_E0)):
            u = uu[i, j]
            theta_E = tt[i, j]
            ellip, err_ellip = ellipticity(deltaPix, psf, u, theta_E=theta_E, use_galsim=True)
            ellips[i,j] = ellip
            err[i,j] = err_ellip
            tbar.update(1)

with open('ellip_table_webb.npy', 'wb') as f:
    np.save(f, theta_E0)
    np.save(f, u0)
    np.save(f, ellips)
    np.save(f, err)