'''reference: https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import coolwarm
from mpl_toolkits.mplot3d import Axes3D, axes3d


def plt_line(N0=200):
    theta = np.linspace(-4*np.pi, 4*np.pi, N0)
    z = np.linspace(-2, 2, N0)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    hFig = plt.figure()
    hAxes = hFig.add_axes([0,0,1,1], projection='3d')
    hAxes.plot(x, y, z, label='parametric curve')
    hAxes.legend()
    hAxes.set_title('plt_line')
    hFig.show()


def plt_scatter(N0=200):
    hFig = plt.figure()
    hAxes = hFig.add_axes([0,0,1,1], projection='3d')
    x,y,z = np.random.normal(0,1,[N0,]), np.random.normal(0,1,[N0,]), np.random.normal(0,1,[N0,])
    hAxes.scatter(x, y, z, c='r', marker='o', label='class1')
    x,y,z = np.random.normal(1,1,[N0,]), np.random.normal(1,1,[N0,]), np.random.normal(1,1,[N0,])
    hAxes.scatter(x, y, z, c='b', marker='^', label='class2')
    hAxes.legend()
    hAxes.set_title('plt_scatter')
    hFig.show()


def plt_surface():
    x,y = np.meshgrid(np.linspace(-5,5), np.linspace(-5,5))
    z = np.sin(np.sqrt(x**2 + y**2))

    hFig = plt.figure()
    hAxes = hFig.add_axes([0,0,0.93,1], projection='3d')
    hSurf = hAxes.plot_surface(x, y, z, cmap=coolwarm, linewidth=0, antialiased=True)
    hFig.colorbar(hSurf, shrink=0.5, aspect=5)
    hAxes.set_title('plt_surface')
    hFig.show()


def plt_surface01():
    theta = np.linspace(0, np.pi, 100)[np.newaxis]
    phi = np.linspace(0, 2*np.pi, 100)[:,np.newaxis]

    x = 10 * np.sin(theta) * np.cos(phi)
    y = 10 * np.sin(theta) * np.sin(phi)
    z = 10 * np.cos(theta) * np.ones_like(phi)

    hFig = plt.figure()
    hAxes = hFig.add_axes([0,0,1,1], projection='3d')
    hAxes.plot_surface(x, y, z)
    hAxes.set_title('plt_surface01')
    hFig.show()


def plt_wireframe():
    x, y, z = axes3d.get_test_data(0.05)

    hFig = plt.figure()
    hAxes = hFig.add_axes([0,0,1,1], projection='3d')
    hAxes.plot_wireframe(x, y, z, rstride=10, cstride=10)
    hAxes.set_title('plt_wireframe')
    hFig.show()


def plt_contour():
    x, y, z = axes3d.get_test_data(0.05)

    hFig = plt.figure()
    hAxes = hFig.add_axes([0,0,1,1], projection='3d')
    _ = hAxes.contour(x, y, z, cmap=coolwarm)
    hAxes.set_title('plt_contour')
    hFig.show()


if __name__=='__main__':
    plt_line()
    plt_scatter()
    plt_surface()
    plt_surface01()
    plt_wireframe()
    plt_contour()
