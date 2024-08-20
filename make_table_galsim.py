import numpy as np
import matplotlib.pyplot as plt
import galsim
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.psf import PSF
from lenstronomy.Util.util import make_grid
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Cosmo.micro_lensing import einstein_radius
import argparse

with open('make_table.log', 'a+') as f:
    f.write(' '.join(sys.argv)+'\n')

parser = argparse.ArgumentParser()
parser.add_argument('--d_l', default=100, type=float, help='distance of lens [pc]')
parser.add_argument('--d_s', default=200, type=float, help='distance of source [pc]')
parser.add_argument('--M0', default=1e3, type=float, help='mass of lens [M_sol]')
parser.add_argument('--deltaPix', default=0.2, type=float, help='resolution of image [arcsec]')
parser.add_argument('--fwhm', default=0.7, type=float, help='fwhm of PSF [arcsec]')
parser.add_argument('--out', '-o', help='output file name')
parser.add_argument('--pickle', '-p', action='store_true', help='pickle the parameters')
parser.add_argument('--mean', '-m', action='store_true', help='calculate mean value across different values')

args = parser.parse_args()

theta_E = einstein_radius(args.M0, args.d_l, args.d_s)
deltaPix = args.deltaPix
fwhm = args.fwhm
ellip_est = 'galsim'
figsize = (fwhm+theta_E)*5
num_pix = int(figsize/deltaPix)

print(f'theta_E = {theta_E}')

lens_model = LensModel(lens_model_list=['POINT_MASS'])                      # point mass lens
kwargs_lens = [{'theta_E': theta_E, 'center_x': 0, 'center_y': 0}]

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


from lenstronomy.PointSource.point_source import PointSource
# unlensed source positon #
point_source_model_list = ['SOURCE_POSITION']
pointSource = PointSource(
    point_source_type_list=point_source_model_list, lens_model=lens_model, fixed_magnification_list=[True],
    kwargs_lens_eqn_solver={'search_window': -x0*2, 'min_distance': deltaPix/100, 'precision_limit': deltaPix/1000})
kwargs_ps = [{'ra_source': 0, 'dec_source': 0, 'source_amp': 100}]


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
                                    point_source_class=pointSource, kwargs_numerics=kwargs_numerics)
image = imageModel.image(kwargs_lens = [{'theta_E': theta_E, 'center_x': 0.8*theta_E, 'center_y': 0}], kwargs_ps=kwargs_ps)

# imageModel = ImageModel(data_class=pixel_grid, psf_class=psf, lens_model_class=lens_model,
#                                     source_model_class=lightModel_source,
#                                     point_source_class=None, 
#                                     kwargs_numerics=kwargs_numerics)

# image = imageModel.image(kwargs_lens = [{'theta_E': theta_E, 'center_x': theta_E/2, 'center_y': 0}], kwargs_source=kwargs_light)

plt.imshow(image)
plt.savefig(f'./image.pdf')


num = 30
xx = np.linspace(-theta_E*0.6, theta_E*0.6, num)
xx, yy = np.meshgrid(xx, xx)
e = np.zeros_like(xx)

# from calc_ellip import calc_ellip

# if args.galsim:
#     def e_func(image):
#         try:
#             res = galsim.hsm.FindAdaptiveMom(galsim.Image(image))
#         except galsim.errors.GalSimHSMError:
#             print_err()


#         return res.observed_shape.e 
# else:
#     e_func = lambda image: calc_ellip(image)[-1]

import sys
def print_err(ps, l, image):
    print('argument:', file=sys.stderr)
    print(f'{args}', file=sys.stderr)
    print(f'ps={ps}, len={l}', file=sys.stderr)
    np.save('problem_image.npy', image)

Nrand = 10
import tqdm
pbar = tqdm.tqdm(total=num*num)


for i in range(num):
    for j in range(num):
        if args.mean:
            ee = np.zeros(Nrand)
            for k in range(Nrand):
                x_ps, y_ps = np.random.uniform(0, deltaPix, 2)
                x_l = x_ps + xx[i,j]
                y_l = y_ps + yy[i,j]
                image = imageModel.image(
                    kwargs_lens = [{'theta_E': theta_E, 'center_x': x_l, 'center_y':y_l}], 
                    kwargs_ps = [{'ra_source': x_ps, 'dec_source': y_ps, 'source_amp': 100}])
                try:
                    res = galsim.hsm.FindAdaptiveMom(galsim.Image(image))
                except galsim.errors.GalSimHSMError as exp:
                    print(f'{exp}:')
                    print_err((x_ps, y_ps), (x_l, y_l), image)
                    ee[k] = np.nan
                    continue
                ee[k] = res.observed_shape.e
            e[i,j] = np.nanmean(ee)
        else:
            x_ps, y_ps = 0, 0
            x_l = x_ps + xx[i,j]
            y_l = y_ps + yy[i,j]
            image = imageModel.image(
                kwargs_lens = [{'theta_E': theta_E, 'center_x': x_l, 'center_y':y_l}], 
                kwargs_ps = [{'ra_source': x_ps, 'dec_source': y_ps, 'source_amp': 100}])
            try:
                res = galsim.hsm.FindAdaptiveMom(galsim.Image(image))
            except galsim.errors.GalSimHSMError as exp:
                print(f'{exp}:')
                print_err((x_ps, y_ps), (x_l, y_l), image)
                e[i,j] = np.nan
                continue
            e[i,j] = res.observed_shape.e
        pbar.update(1)
pbar.close()

if args.out:
    output = f'./e/{args.out}'
else:
    output = f'./e/e_galsim_deltaPix={deltaPix:.2f}'
print(f'writing to {output}')
np.savetxt(output, e)

if args.pickle:
    import pickle
    print(f'writing to {output}')

    pickle.dump(
        {'d_l': args.d_l, 'd_s': args.d_s, 
         'M0': args.M0, 
         'deltaPix': deltaPix, 'fwhm': fwhm, 
         'theta_E': theta_E, 
         'e': e, 'xy_range': [-theta_E*0.6, theta_E*0.6]
        }, output+'_pickle')


