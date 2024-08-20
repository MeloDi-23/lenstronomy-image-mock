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

parser = argparse.ArgumentParser()
parser.add_argument('--d_l', default=100, type=float, help='distance of lens [pc]')
parser.add_argument('--d_s', default=200, type=float, help='distance of source [pc]')
parser.add_argument('--M0', default=1e3, type=float, help='mass of lens [M_sol]')
parser.add_argument('--deltaPix', default=0.2, type=float, help='resolution of image [arcsec]')
parser.add_argument('--fwhm', default=0.7, type=float, help='fwhm of PSF [arcsec]')
parser.add_argument('--out', '-o', help='output file name')
parser.add_argument('--pickle', '-p', action='store_true', help='pickle the parameters')

args = parser.parse_args()

import sys
with open('make_e_r.log', 'a+') as f:
    f.write(' '.join(sys.argv)+'\n')


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
    kwargs_lens_eqn_solver={'search_window': -x0*2, 'min_distance': deltaPix/50, 'precision_limit': deltaPix/1000})
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


def print_err(ps, l, image):
    print('argument:', file=sys.stderr)
    print(f'{args}', file=sys.stderr)
    print(f'ps={ps}, len={l}', file=sys.stderr)
    # np.save('problem_image.npy', image)

Nbin = 30
r = np.linspace(theta_E*0.01, theta_E*1, Nbin)
Nsamp = 100

import multiprocessing as mp
sender, recev = mp.Pipe()

def calculate_each_samp(i):
    ee = np.zeros(Nbin)
    thetas = np.random.uniform(0, 2*np.pi, Nbin)
    rand_offset_1 = np.random.uniform(-deltaPix, deltaPix, Nbin)
    rand_offset_2 = np.random.uniform(-deltaPix, deltaPix, Nbin)
    for j in range(Nbin):
        theta = thetas[j]
        x_s = rand_offset_1[j]
        y_s = rand_offset_2[j]
        x_l = x_s + r[j]*np.cos(theta)
        y_l = y_s + r[j]*np.sin(theta)

        image = imageModel.image(
            kwargs_lens = [{'theta_E': theta_E, 'center_x':x_l, 'center_y': y_l}], 
            kwargs_ps = [{'ra_source': x_s, 'dec_source': y_s, 'source_amp': 100}])
        try:
            res = galsim.hsm.FindAdaptiveMom(galsim.Image(image))
        except galsim.errors.GalSimHSMError as er:
            print(er)
            print_err([x_s, y_s], [x_l, y_l], image)
            e = np.nan
        else:
            e = res.observed_shape.e
        ee[j] = e
    sender.send(1)
    return ee


import tqdm
pbar = tqdm.tqdm(total=Nsamp)
def process():
    while True:
        if recev.recv():
            pbar.update(1)
        else:
            # pbar.close()
            return

bar = mp.Process(target=process)
bar.start()

pool = mp.Pool(20)
sample = np.vstack(pool.map(calculate_each_samp, range(Nsamp), chunksize=int(Nsamp/20)))
sender.send(0)
bar.join()
assert sample.shape == (Nsamp, Nbin)

estimate = np.nanmean(sample, axis=0)
err = np.nanstd(sample, axis=0, ddof=1)


if args.out:
    output = f'./e-r/{args.out}'
else:
    output = f'./e-r/e_{ellip_est}_deltaPix={deltaPix:.2f}'

np.savetxt(output, [estimate, err])
# if args.pickle:
#     import pickle
#     with open(output+'_pickle', 'wb') as f:
#         pickle.dump(
#             {'d_l': args.d_l, 'd_s': args.d_s, 
#             'M0': args.M0, 
#             'deltaPix': deltaPix, 'fwhm': fwhm, 
#             'theta_E': theta_E, 
#             'e': estimate, 'r_range': r
#             }, f)

