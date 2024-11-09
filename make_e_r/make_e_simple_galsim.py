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
import sys
with open('make_e_simp.log', 'a+') as f:
    f.write(' '.join(sys.argv)+'\n')

parser = argparse.ArgumentParser()
parser.add_argument('--d_l', type=float, help='distance of lens [pc]')
parser.add_argument('--d_s', type=float, help='distance of source [pc]')
parser.add_argument('--M0', type=float, help='mass of lens [M_sol]')
parser.add_argument('--theta_E', type=float, help='Einstein angular radius [arcsec]')
parser.add_argument('--deltaPix', default=0.2, type=float, help='resolution of image [arcsec]')
parser.add_argument('--fwhm', default=0.7, type=float, help='fwhm of PSF [arcsec]')
parser.add_argument('--out', '-o', help='output file name')
parser.add_argument('--pickle', '-p', action='store_true', help='pickle the parameters')

args = parser.parse_args()

if args.theta_E:
    theta_E = args.theta_E
else:
    theta_E = einstein_radius(args.M0, args.d_l, args.d_s)
deltaPix = args.deltaPix
fwhm = args.fwhm
figsize = (fwhm+theta_E)*5
num_pix = int(figsize/deltaPix)

print(f'theta_E = {theta_E}')

psf = galsim.Gaussian(fwhm=fwhm, flux=1)
amp = 10

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


mymodel = SimpleImageModel(deltaPix, num_pix, num_pix, (0, 0), psf)

num = 30
rp = np.linspace(theta_E*1e-3, theta_E*0.6, num)
e = np.zeros_like(rp)

import sys
def print_err(ps, l, image):
    print('argument:', file=sys.stderr)
    print(f'{args}', file=sys.stderr)
    print(f'ps={ps}, len={l}', file=sys.stderr)
    np.save('problem_image.npy', image)

Nrand = 100
import multiprocessing as mp

sender, recev = mp.Pipe()

def mock_and_e(radius):
    ee = np.zeros(Nrand)
    theta = np.random.uniform(0, 2*np.pi, Nrand)
    x = np.cos(theta)*radius
    y = np.sin(theta)*radius
    for k in range(Nrand):        
        x_ps, y_ps = np.random.uniform(0, deltaPix, 2)
        x_l = x_ps + x[k]
        y_l = y_ps + y[k]
        image = mymodel.mock_image(
            theta_E, x_l, y_l, x_ps, y_ps, amp
        )
        try:
            res = galsim.hsm.FindAdaptiveMom(image)
            ee[k] = res.observed_shape.e
        except galsim.errors.GalSimHSMError as exp:
            print(f'{exp}:')
            print_err((x_ps, y_ps), (x_l, y_l), image)
            ee[k] = np.nan
            continue
    sender.send(1)
    return np.nanmean(ee), np.nanstd(ee, ddof=1)

import tqdm

def process(total):
    pbar = tqdm.tqdm(total=total)
    while True:
        if recev.recv():
            pbar.update(1)
        else:
            pbar.close()
            return

bar = mp.Process(target=process, args=(num,))
bar.start()
pool = mp.Pool(20)
result = pool.map(mock_and_e, rp, chunksize=10)
sender.send(0)
bar.join()
e = np.array(result)
if args.out:
    output = f'./e-r/{args.out}'
else:
    output = f'./e-r/e_simple_deltaPix={deltaPix:.2f}'
print(f'writing to {output}')
np.savetxt(output, np.concatenate((rp.reshape(-1, 1), e), axis=1))

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


