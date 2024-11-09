import numpy as np
import galsim
class ShapeData:
    def __init__(self, e1, e2, M):
        self.e1 = e1
        self.e2 = e2
        self.e = np.sqrt(e1*e1+e2*e2)
        self.moments_sigma = M
def calc_ellip(image):
    image = image.array
    s_x, s_y = image.shape
    XX, YY = np.meshgrid(np.arange(s_x), np.arange(s_y))

    x0 = (image*XX).sum()/image.sum()
    y0 = (image*YY).sum()/image.sum()

    total = image.sum()
    Ixx = ((XX-x0)*(XX-x0)*image).sum() / total
    Iyy = ((YY-y0)*(YY-y0)*image).sum() / total
    Ixy = ((YY-y0)*(XX-x0)*image).sum() / total

    e1 = (Ixx-Iyy)/(Ixx+Iyy)
    e2 = 2*Ixy/(Ixx+Iyy)
    M = (Ixx*Iyy - Ixy*Ixy)**(1/4)

    return ShapeData(e1, e2, M)
