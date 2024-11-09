import numpy as np
import matplotlib.pyplot as plt
from lenstronomy.Cosmo.micro_lensing import einstein_radius
from calc_ellipticity import ellipticity,ellipticity_image_mock
from astropy import units as u
deltaPix = 0.03
psf = 0.08

d_s = 8000
d_l = 4000              # pc
u_min = 0.2
d_lu = d_l*u.pc

tE = 390*u.day          # a characteristic time scale, arbitrary.

print(f't_E = {tE}')


t = np.linspace(0, tE.value*1.5, 70)

d_s = 8000
d_l = 4000                                      # pc
M0 = np.linspace(50, 800, 50)                  # M_sol
v0 = np.linspace(200, 800, 50)      # km/s

MM, vv = np.meshgrid(M0, v0, indexing='ij')
ellips = np.zeros((len(M0), len(v0), len(t)))

# i= 30
# j = 0
# m = MM[i, j]
# v = vv[i, j]
# theta_E = einstein_radius(m, d_l, d_s)     # arcsec
# b = theta_E*u_min                             # We fix b/theta_E, so that the peak amplification is the same
# beta = np.sqrt(b**2 + (t*u.day* v*u.km/u.s/ d_lu * u.rad).to(u.arcsec).value**2) / theta_E
# ellip, err_ellip = ellipticity(deltaPix, psf, beta, theta_E=theta_E)
# ellips[i,j,:] = ellip
# print(m, v, theta_E)

# with open('log', 'w') as f:
#     np.savetxt(f, beta)
#     np.savetxt(f, ellip)

        
import tqdm
with tqdm.tqdm(total=len(M0)*len(v0)) as tbar:
    for i in range(len(M0)):
        for j in range(len(v0)):
            m = MM[i, j]
            v = vv[i, j]
            theta_E = einstein_radius(m, d_l, d_s)     # arcsec
            b = theta_E*u_min                             # We fix b/theta_E, so that the peak amplification is the same
            beta = np.sqrt(b**2 + (t*u.day* v*u.km/u.s/ d_lu * u.rad).to(u.arcsec).value**2) / theta_E
            ellip, err_ellip = ellipticity(deltaPix, psf, beta, theta_E=theta_E)
            ellips[i,j,:] = ellip

            tbar.update(1)

np.save('ellip_table', ellips)
with open('table_arguments.npy', 'wb') as f:
    np.save(f, t)
    np.save(f, M0)
    np.save(f, v0)