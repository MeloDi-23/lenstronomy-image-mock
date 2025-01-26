## characteristic value

1000 M_sun
velocity~200 km/s
t_E ~ month

## A better Interpolation table
Come up with a table, that giving theta_E and t_E, u_min, it can generate a curve.
e.g.: choose theta_E - u table
then for a giving t_E, u_min, t0, u=sqrt(u_min^2 + ((t-t0)/t_E)^2), you can interpolate the theta_E-u diagram to get a result.

u range: 0.1 ~ 1.5
theta_E range(depend on the scale you care about): 0.008(M=200, D_l=6k, D_s=8k) ~ 0.056(M=1000, D_l=2k, D_s=8k)

Note that larger M and smaller D_l gives larger theta_E

**this table rely on the choice of psf and delta_pix.**

Try other psf: eg: Kolmogorov psf profile.
0.005
JWST PSF
PRL