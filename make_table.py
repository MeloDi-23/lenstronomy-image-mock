import numpy as np
import matplotlib.pyplot as plt
import galsim
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.psf import PSF
from lenstronomy.Util.util import make_grid
from lenstronomy.Data.pixel_grid import PixelGrid

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--thetaE', default=0.42, type=float)
parser.add_argument('--deltaPix', default=0.1, type=float)
parser.add_argument('--fwhm', default=0.7, type=float)
parser.add_argument('--galsim', action='store_true')


args = parser.parse_args()

theta_E = args.thetaE
deltaPix = args.deltaPix
fwhm = args.fwhm

ellip_est = 'galsim' if args.galsim else 'di'

figsize = (fwhm+theta_E)*3
num_pix = int(figsize/deltaPix)

lens_model = LensModel(lens_model_list=['POINT_MASS'])                      # point mass lens
kwargs_lens = [{'theta_E': theta_E, 'center_x': 0, 'center_y': 0}]

# lightModel_source = LightModel(light_model_list=['GAUSSIAN'])
# kwargs_light = [{'amp': 100, 'center_x': 0, 'center_y': 0, 'sigma': fwhm/100}]

lightModel_source = LightModel(light_model_list=['ELLIPSOID'])
kwargs_light = [{'amp': 1, 'radius': 1.16311e-6 /2*10, 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0}]

kwargs_psf = {'psf_type': 'GAUSSIAN',   # type of PSF model (supports 'GAUSSIAN' and 'PIXEL')
              'fwhm': fwhm,             # full width at half maximum of the Gaussian PSF (in angular units)
              'pixel_size': deltaPix    # angular scale of a pixel (required for a Gaussian PSF to translate the FWHM into a pixel scale)
             }
psf = PSF(**kwargs_psf)

# setup the keyword arguments to create the Data() class #
x0 = make_grid(num_pix, deltaPix)[0][0]
ra_at_xy_0 = dec_at_xy_0 = x0 # coordinate in angles (RA/DEC) at the position of the pixel edge (0,0)
print('ra, dec range: [-{0}, {0}]'.format(-x0))
transform_pix2angle = np.array([[1, 0], [0, 1]]) * deltaPix  # linear translation matrix of a shift in pixel in a shift in coordinates
kwargs_pixel = {'nx': num_pix, 'ny': num_pix,  # number of pixels per axis
                'ra_at_xy_0': ra_at_xy_0,  # RA at pixel (0,0)
                'dec_at_xy_0': dec_at_xy_0,  # DEC at pixel (0,0)
                'transform_pix2angle': transform_pix2angle}

pixel_grid = PixelGrid(**kwargs_pixel)

# define the numerics #
supersampling_factor = 5
kwargs_numerics = {'supersampling_factor': supersampling_factor, # super sampling factor of (partial) high resolution ray-tracing
                        'compute_mode': 'regular', # 'regular' or 'adaptive'
                        'supersampling_convolution': True,  # bool, if True, performs the supersampled convolution (either on regular or adaptive grid)
                        'supersampling_kernel_size': None,  # size of the higher resolution kernel region (can be smaller than the original kernel). None leads to use the full size
                        'flux_evaluate_indexes': None,  # bool mask, if None, it will evaluate all (sub) pixels
                        'supersampled_indexes': None,  # bool mask of pixels to be computed in supersampled grid (only for adaptive mode)
                        'compute_indexes': None,  # bool mask of pixels to be computed the PSF response (flux being added to). Only used for adaptive mode and can be set =likelihood mask.
                        'point_source_supersampling_factor': 3,  # int, supersampling factor when rendering a point source (not used in this script)
                       }

imageModel = ImageModel(data_class=pixel_grid, psf_class=psf, lens_model_class=lens_model,
                                    source_model_class=lightModel_source,
                                    point_source_class=None, 
                                    kwargs_numerics=kwargs_numerics)

image = imageModel.image(kwargs_lens = [{'theta_E': theta_E, 'center_x': theta_E/2, 'center_y': 0}], kwargs_source=kwargs_light)

plt.imshow(image)
plt.savefig(f'./image.pdf')

# Nbin = 30

# r = np.linspace(0, theta_E*0.5, Nbin)
# Nsamp = 100
# sample = np.zeros((Nbin, Nsamp), float)

# import random
# import tqdm
# for i in tqdm.trange(Nsamp):
#     thetas = np.random.uniform(0, 2*np.pi, Nbin)
#     rand_offset_1 = np.random.uniform(-deltaPix, deltaPix, Nbin)
#     rand_offset_2 = np.random.uniform(-deltaPix, deltaPix, Nbin)
#     for j in range(Nbin):
#         theta = thetas[j]
#         x_s = rand_offset_1[j]
#         y_s = rand_offset_2[j]
#         x_l = x_s + r[j]*np.cos(theta)
#         y_l = y_s + r[j]*np.sin(theta)

#         image = imageModel.image(kwargs_lens = [{'theta_E': theta_E, 'center_x':x_l, 'center_y': y_l}], kwargs_source = [{'amp': 100, 'center_x': x_s, 'center_y': y_s, 'sigma': fwhm/100}])
#         res = galsim.hsm.FindAdaptiveMom(galsim.Image(image))
#         e = res.observed_shape.e
#         sample[j, i] = e


# estimate = sample.mean(axis=1)
# err = sample.std(axis=1)

# np.savetxt('./e-r.txt', [estimate, err])
















num = 30
xx = np.linspace(-theta_E*0.5, theta_E*0.5, num)
xx, yy = np.meshgrid(xx, xx)
e = np.zeros_like(xx)

from calc_ellip import calc_ellip

if args.galsim:
    def e_func(image):
        res = galsim.hsm.FindAdaptiveMom(galsim.Image(image))
        return res.observed_shape.e 
else:
    e_func = lambda image: calc_ellip(image)[-1]

import tqdm
pbar = tqdm.tqdm(num*num)
for i in range(num):
    for j in range(num):
        imageModel = ImageModel(data_class=pixel_grid, psf_class=psf, lens_model_class=lens_model,
                                source_model_class=lightModel_source,
                                point_source_class=None)

        image = imageModel.image(kwargs_lens = [{'theta_E': theta_E, 'center_x': xx[i,j], 'center_y': yy[i,j]}], kwargs_source=kwargs_light)
        
        e[i,j] = e_func(image)
        pbar.update(1)
pbar.close()

np.savetxt(f'e/e_{ellip_est}_deltaPix={deltaPix:.2f}.txt', e)
