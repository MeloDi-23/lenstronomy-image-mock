import numpy as np
import matplotlib.pyplot as plt
from lenstronomy.Cosmo.micro_lensing import einstein_radius
from calc_ellipticity_webb import ellipticity, ellipticity_image_mock
from astropy import units as u
deltaPix = 0.031
psf = 0.08      # useless in this case

d_s = 8000
d_l = 4000              # pc
M0 = 200                # M_sol
u_min = 0.2
velocity = 300          # km/s
snr = 100
save_file = 'mock3_webb'

velocity = velocity*u.km/u.s
theta_E = einstein_radius(M0, d_l, d_s)     # arcsec
d_lu = d_l*u.pc
tE = (theta_E*u.arcsec/(velocity/d_lu*u.rad)).to(u.day)

print(f't_E = {tE}')
t = np.linspace(-1.2*tE.value, 1.2*tE.value, 50)
b = theta_E*u_min                                 # let's assume the nearest position is 0.5 theta_E
beta = np.sqrt(b**2 + (t*u.day*velocity/d_lu*u.rad).to(u.arcsec).value**2) / theta_E

Amp = (beta*beta+1)/(beta*np.sqrt(beta*beta + 4))
ellip, err_ellip = ellipticity(deltaPix, psf, beta, theta_E=theta_E, use_galsim=True)

plt.show()

t_obs = np.linspace(-1.2*tE.value, 1.2*tE.value, 20)
b = theta_E*u_min                                 # let's assume the nearest position is 0.5 theta_E
x = (t_obs*u.day*velocity/d_lu*u.rad).to(u.arcsec).value
e_obs = np.zeros_like(t_obs)
err_obs = np.zeros_like(t_obs)
for i in range(len(t_obs)):
    res = ellipticity_image_mock(deltaPix, psf, 0, 0, x[i], b, theta_E, 100, True, True, snr, 1)
    e_obs[i] = res['e']
    err_obs[i] = res['e_err']


plt.errorbar(t_obs, e_obs, err_obs, fmt='o')
plt.plot(t, ellip, '--')

plt.title(rf'SNR={snr}, $\Delta m = 5$')
plt.savefig('mock_image')

np.save(save_file, [t_obs, e_obs, err_obs])