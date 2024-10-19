import numpy as np
import galsim
from lenstronomy.Cosmo.micro_lensing import einstein_radius
from tqdm import tqdm
from calc_moments import calc_ellip as _calc_e_for_image

def ellipticity(deltaPix, fwhm, beta, 
                d_l=None, d_s=None, M_0=None, theta_E=None, 
                mean_value=True, verbose=False, use_galsim=True, add_noise=False, snr=100, sky_level=0.0):
    try:
        if d_l is None:
            assert d_s is None and M_0 is None
            assert theta_E is not None
        else:
            assert d_s is not None and M_0 is not None
            assert theta_E is None
            theta_E = einstein_radius(M_0, d_l, d_s)
    except AssertionError:
        raise ValueError('d_s, d_l, M_0 should be set to None, or theta_E should be set to None')
    
    if np.isscalar(beta) and not np.isscalar(theta_E):
        theta_E = np.array(theta_E, copy=False)
        beta = beta*np.ones_like(theta_E)
    elif np.isscalar(theta_E) and not np.isscalar(beta):
        beta = np.array(beta, copy=False)
        theta_E = theta_E*np.ones_like(beta)
    else:
        beta = np.array(beta, copy=False)
        theta_E = np.array(theta_E, copy=False)
        if beta.shape != theta_E.shape:
            raise ValueError("beta and theta_E must be of the same shape. beta has shape {}, but theta_E has shape {}".format(beta.shape, theta_E.shape))
    iter = enumerate(zip(beta.flat, theta_E.flat))
    if verbose:
        iter = tqdm(iter, total=beta.size)
    result = np.zeros(beta.size, float)
    err = np.zeros(beta.size, float)

    Nmeans = 20 if mean_value else 1
    for i, (b, theta) in iter:
        result[i], err[i] = ellipticity_single(deltaPix, fwhm, b, theta, Nmeans, use_galsim, add_noise, snr, sky_level)
    return result.reshape(beta.shape), err.reshape(beta.shape)


def ellipticity_single(deltaPix, fwhm, beta, theta_E, N_mean, use_galsim, add_noise, snr, sky_level):
    res = np.zeros(N_mean, dtype=float)
    figsize = (fwhm+theta_E)*5
    num_pix = int(figsize/deltaPix)
    amp = 10
    psf = galsim.Gaussian(fwhm=fwhm, flux=1)
    mymodel = SimpleImageModel(deltaPix, num_pix, num_pix, (0, 0), psf)

    theta = np.random.uniform(0, 2*np.pi, N_mean)
    x = np.cos(theta)*beta*theta_E
    y = np.sin(theta)*beta*theta_E

    if use_galsim:
        ellip = lambda image: galsim.hsm.FindAdaptiveMom(image).observed_shape.e
    else:
        ellip = lambda image: _calc_e_for_image(image.array)[2]

    
    for i in range(N_mean):
        x_ps, y_ps = np.random.uniform(0, deltaPix, 2)
        x_l = x_ps + x[i]
        y_l = y_ps + y[i]
        image = mymodel.mock_image(
            theta_E, x_l, y_l, x_ps, y_ps, amp
        )
        if add_noise:
            addPoissonNoiseSNR(image, snr, sky_level)
            
        try:
            res[i] = ellip(image)
        except galsim.errors.GalSimHSMError as exp:
            print(f'{exp}:')
            res[i] = np.nan
            continue
    return np.nanmean(res), np.nanstd(res, ddof=1)


def addPoissonNoiseSNR(image, snr, sky_level):
    # add poisson noise to the image. rescale the image value first, so that it meets the target SNR.
    refined = np.maximum(image.array, 0)    # some times the value of pixel is negative, because of fft.
    w = refined
    snr_old = (w*refined).sum() / np.sqrt((w*w*(refined + sky_level)).sum())
    ratio = snr / snr_old
    ratio = ratio * ratio

    image *= ratio
    sky_level *= ratio
    image.addNoise(galsim.PoissonNoise(sky_level=sky_level))

    refined = np.maximum(image.array, 0)
    w = refined
    return (w*refined).sum() / np.sqrt((w*w*refined).sum())


class SimpleImageModel:
    def __init__(self, pixel_size, nx, ny, origin, psf):
        self.pix_size = pixel_size
        self.origin = origin
        self.nx = nx
        self.ny = ny
        self.psf = psf
    def mock_image(self, theta_E, x_l, y_l, x_ps, y_ps, amp):
        beta = np.sqrt((x_l-x_ps)**2+(y_l-y_ps)**2)
        theta_p = (beta + np.sqrt(beta*beta+4*theta_E*theta_E))/2
        theta_n = (beta - np.sqrt(beta*beta+4*theta_E*theta_E))/2
        x = (theta_p/theta_E)**4
        A_p = x/(x-1)*amp
        x = (theta_n/theta_E)**4
        A_n = -x/(x-1)*amp
        assert A_p >= 0 and A_n >= 0

        x_p = theta_p/beta*(x_ps - x_l) + x_l
        y_p = theta_p/beta*(y_ps - y_l) + y_l
        x_n = theta_n/beta*(x_ps - x_l) + x_l
        y_n = theta_n/beta*(y_ps - y_l) + y_l

        p = galsim.DeltaFunction(flux=A_p)
        n = galsim.DeltaFunction(flux=A_n)

        con_p = galsim.Convolve([self.psf, p])
        con_n = galsim.Convolve([self.psf, n])

        image = con_p.drawImage(nx=self.nx, ny=self.ny, scale=self.pix_size, offset=(x_p/self.pix_size, y_p/self.pix_size))
        image = con_n.drawImage(image, scale=self.pix_size, add_to_image=True, offset=(x_n/self.pix_size, y_n/self.pix_size))
        return image