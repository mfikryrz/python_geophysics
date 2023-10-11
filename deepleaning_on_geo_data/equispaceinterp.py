import numpy as np

def equispace2d(x,y,n):
    xd = np.diff(x)
    yd = np.diff(y)
    dist = np.sqrt(xd**2+yd**2)
    u = np.cumsum(dist)
    u = np.hstack([[0],u])

    t = np.linspace(0,u.max(),n)
    xn = np.interp(t, u, x)
    yn = np.interp(t, u, y)
    return xn, yn

def equispace3d(x,y,z,n):
    xd = np.diff(x)
    yd = np.diff(y)
    zd = np.diff(z)

    dist = np.sqrt(xd**2+yd**2+zd**2)

    u = np.cumsum(dist)
    u = np.hstack([[0],u])

    t = np.linspace(0,u.max(),n)
    xn = np.interp(t, u, x)
    yn = np.interp(t, u, y)
    zn = np.interp(t, u, z)
    return np.vstack((xn, yn, zn)).T

