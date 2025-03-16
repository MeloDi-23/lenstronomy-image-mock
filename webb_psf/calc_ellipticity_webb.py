import numpy as np
import galsim
from lenstronomy.Cosmo.micro_lensing import einstein_radius
from tqdm import tqdm
from calc_moments import calc_ellip as _calc_e_for_image    # this can be changed
import webbpsf
nrc = webbpsf.NIRCam()
psf = nrc.calc_psf()
pixel_size = psf[1].header['PIXELSCL']
image = galsim.Image(psf[1].data)

base_shear = image.FindAdaptiveMom().observed_shape

galsim_psf = galsim.InterpolatedImage(image, scale=pixel_size)


def ellipticity(deltaPix, fwhm, beta, 
                d_l=None, d_s=None, M_0=None, theta_E=None, 
                mean_value=True, verbose=False, flux=100, use_galsim=True, add_noise=False, snr=100, sky_level=0.0):
    """
    calculate the ellipticity with given microlensing parameters.
    you can only define the d_l+d_+M_0, or theta_E.
    If theta_E or beta is array-like, they must be able to be cast to the same shape.
    use_galsim: means using adaptive momentum method, and non galsim method is to calculate 
        momentum directly without any weight. adaptive momentum method sometimes raises 
        exception when the Object is too small. To avoid that, you should not set the PSF 
        too small compared to the pixel size.
    add_noise: determines whether to add poisson noise. the image and sky_level is scaled 
        in order to meet the snr. when you use non-galsim method and add a lot of possion noise, 
        this code will perform poor.
    """
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
        result[i], err[i] = ellipticity_single(deltaPix, fwhm, b, theta, Nmeans, flux, use_galsim, add_noise, snr, sky_level)
    return result.reshape(beta.shape), err.reshape(beta.shape)


def ellipticity_single(deltaPix, fwhm, beta, theta_E, N_mean, flux, use_galsim, add_noise, snr, sky_level):
    res = np.zeros(N_mean, dtype=float)
    figsize = (fwhm+theta_E)*5
    num_pix = int(figsize/deltaPix)

    mymodel = SimpleImageModel(deltaPix, num_pix, num_pix, (0, 0), galsim_psf)

    theta = np.random.uniform(0, 2*np.pi, N_mean)
    x = np.cos(theta)*beta*theta_E
    y = np.sin(theta)*beta*theta_E

    if use_galsim:
        def ellip(image):
            shear = image.FindAdaptiveMom().observed_shape
            return (shear-base_shear).e
    else:
        ellip = lambda image: _calc_e_for_image(image).e

    
    for i in range(N_mean):
        x_ps, y_ps = np.random.uniform(0, deltaPix, 2)
        x_l = x_ps + x[i]
        y_l = y_ps + y[i]
        image = mymodel.mock_image(
            theta_E, x_l, y_l, x_ps, y_ps, flux
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

def ellipticity_image_mock(deltaPix, fwhm, x_ps, y_ps, x_l, y_l, theta_E, flux, use_galsim, add_noise, snr, sky_level):
    """
    returns the real image of mock with noise, estimate the ellipticity and error.
    There is a problem here: Should the sky level and time be the same over a series of observations?
    """
    figsize = (fwhm+theta_E)*6
    num_pix = int(figsize/deltaPix)

    mymodel = SimpleImageModel(deltaPix, num_pix, num_pix, (0, 0), galsim_psf)
    
    image = mymodel.mock_image(
        theta_E, x_l, y_l, x_ps, y_ps, flux
    )
    ratio = 1
    if add_noise:
        ratio = addPoissonNoiseSNR(image, snr, sky_level)

    if use_galsim:
        def findMom(image):
            mom = image.FindAdaptiveMom()
            shear = mom.observed_shape
            return (shear-base_shear).e, mom.moments_sigma
        # def findMom(image):
        #     mom = image.FindAdaptiveMom()
        #     return mom.observed_shape.e, mom.moments_sigma
    else:
        def findMom(image):
            res = _calc_e_for_image(image)
            return res.e, res.moments_sigma

    e, sig = findMom(image)
    if add_noise:
        Ne = ratio*flux
        sky = ratio*sky_level
        R2 = 1
        sigma_sky = sig/(R2*Ne)*np.sqrt(4*np.pi*sky)
        sigma_star = 1/R2 * np.sqrt(64/(27*Ne))
        sigma = np.sqrt(sigma_sky**2 + sigma_star**2)
    else:
        sigma = 0
    

    return {'e': e, 'e_err': sigma, 'image': image}


def _addPoissonNoiseSNR_1(image, snr, sky_level):
    # add poisson noise to the image. rescale the image value first, so that it meets the target SNR.
    brightness = image.array           # some times the value of pixel is negative, because of fft.
    w = brightness
    snr_old = (w*brightness).sum() / np.sqrt((w*w*(brightness + sky_level)).sum())
    ratio = snr / snr_old
    ratio = ratio * ratio

    image *= ratio
    sky_level *= ratio
    image.addNoise(galsim.PoissonNoise(sky_level=sky_level))

    return ratio

def _addPoissonNoiseSNR_2(image, snr, sky_level):
    brightness = image.array           # some times the value of pixel is negative, because of fft.
    psf = brightness.copy()                    # use surface brightness to estimate psf.
    psf /= psf.sum()
    sharpness = (psf*psf).sum()
    snr_old = brightness.sum() / (np.sqrt(brightness.sum() + sky_level/sharpness))
    ratio = snr / snr_old
    ratio = ratio * ratio

    image *= ratio
    sky_level *= ratio
    image.addNoise(galsim.PoissonNoise(sky_level=sky_level))

    return ratio


addPoissonNoiseSNR = _addPoissonNoiseSNR_2


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