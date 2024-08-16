import numpy as np
import matplotlib.pyplot as plt

theta_E = 0.18/2            # unit: arcsec
deltaPix = 0.074

fwhm = 0.15
num_pix = 30
from lenstronomy.LensModel.lens_model import LensModel
lens_model = LensModel(lens_model_list=['POINT_MASS'])                      # point mass lens
kwargs_lens = [{'theta_E': theta_E, 'center_x': 0, 'center_y': 0}]


from lenstronomy.Data.psf import PSF
kwargs_psf = {'psf_type': 'GAUSSIAN',  # type of PSF model (supports 'GAUSSIAN' and 'PIXEL')
              'fwhm': fwhm,  # full width at half maximum of the Gaussian PSF (in angular units)
              'pixel_size': deltaPix  # angular scale of a pixel (required for a Gaussian PSF to translate the FWHM into a pixel scale)
             }
psf = PSF(**kwargs_psf)

from lenstronomy.Util.util import make_grid, array2image
from lenstronomy.Data.pixel_grid import PixelGrid

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
from lenstronomy.PointSource.point_source import PointSource

# unlensed source positon #
point_source_model_list = ['SOURCE_POSITION']
pointSource = PointSource(point_source_type_list=point_source_model_list, lens_model=lens_model, fixed_magnification_list=[True])
kwargs_ps = [{'ra_source': 0, 'dec_source': 0, 'source_amp': 100}]
from lenstronomy.ImSim.image_model import ImageModel
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

import galsim

fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharex=False, sharey=False)
x = [0, 0.02, 0.05]
fig.suptitle(f'$\\theta_E={theta_E}$, pixel size={deltaPix:.3f}, fwhm = {fwhm}')

imageModel = ImageModel(data_class=pixel_grid, psf_class=psf, lens_model_class=lens_model,
                        source_model_class=lightModel_source,
                        point_source_class=None,
                        kwargs_numerics=kwargs_numerics)


image = imageModel.image(kwargs_lens = [{'theta_E': 0}], kwargs_source=kwargs_light)

i = 3
axes[i].imshow(image, origin='lower')
    # axes[i].matshow(image)
axes[i].set_xticks([])
axes[i].set_yticks([])

axes[i].set_xlabel('ra')
axes[i].set_ylabel('dec')
axes[i].set_title(f'UnLensed')

for i in range(3):
    image = imageModel.image(kwargs_lens = [{'theta_E': theta_E, 'center_x': x[i], 'center_y': 0}],
                            kwargs_source=kwargs_light)
    galsim.hsm.FindAdaptiveMom(image)
    axes[i].matshow(image, origin='lower')
    axes[i].set_xticks([])
    axes[i].set_yticks([])

    axes[i].set_xlabel('ra')
    axes[i].set_ylabel('dec')
    axes[i].set_title(f'$\\beta={x[i]}$')
fig.tight_layout()
plt.savefig('./demo')


