'''reference: https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html'''
import numpy as np
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axes3d

def plt_tri_surface():
    r = np.linspace(0, 1, 10)[:,np.newaxis]
    theta = np.linspace(0, 2*np.pi, 50)[np.newaxis]
    x = (r*np.cos(theta)).reshape(-1)
    y = (r*np.sin(theta)).reshape(-1)
    z = np.sin(-x*y)

    hFig = plt.figure()
    hAxes = hFig.add_axes([0,0,1,1], projection='3d')
    hAxes.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
    hAxes.set_title('Tri-surface')
    hFig.show()


if __name__=='__main__':
    plt_tri_surface()
