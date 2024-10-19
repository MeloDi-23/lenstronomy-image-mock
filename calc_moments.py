import numpy as np
def calc_ellip(image):
    s_x, s_y = image.shape

    XX, YY = np.meshgrid(np.arange(s_x), np.arange(s_y))

    x0 = (image*XX).sum()/image.sum()
    y0 = (image*YY).sum()/image.sum()

    Ixx = ((XX-x0)*(XX-x0)*image).sum()
    Iyy = ((YY-y0)*(YY-y0)*image).sum()
    Ixy = ((YY-y0)*(XX-x0)*image).sum()

    e1 = (Ixx-Iyy)/(Ixx+Iyy)
    e2 = 2*Ixy/(Ixx+Iyy)

    return e1, e2, np.sqrt(e1*e1+e2*e2)

