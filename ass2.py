import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import numba as nb
import math

# set constants
R = 1
Q = 10
REZ=101

charge_density = lambda i,j:math.exp(-(i**2+j**2)/(2*R**2)) * Q/REZ**2

def calc_field(r,x,y,z,q):
    # calculate the electric field at position r, we omit the 4 pi epsilon_0 for simplicity
    dx=r[0]-x
    dy=r[1]-y
    dz=r[2]-z
    abs_d=np.sqrt(dx**2+dy**2+dz**2)
    Ex=np.sum(dx/abs_d**3*q)
    Ey=np.sum(dy/abs_d**3*q)
    Ez=np.sum(dz/abs_d**3*q)
    return np.array([Ex,Ey,Ez])

def calc_fake_field(r,x,y,z,q):
    # for quiver plot purposes
    dx=r[0]-x
    dy=r[1]-y
    dz=r[2]-z
    abs_d=np.sqrt(dx**2+dy**2+dz**2)
    Ex=np.sum(dx/abs_d**2*q)
    Ey=np.sum(dy/abs_d**2*q)
    Ez=np.sum(dz/abs_d**2*q)
    return np.array([Ex,Ey,Ez])


def calc_potential(r,x,y,z,q):
    dx=r[0]-x
    dy=r[1]-y
    dz=r[2]-z
    abs_d=np.sqrt(dx**2+dy**2+dz**2)
    return np.sum(q/abs_d)


# we approximate the charge of the plane with a whole bunch of points (onehundred points as in lecture)
arr=np.linspace(-R,R,REZ)
x_grid,y_grid=np.meshgrid(arr,arr)
x,y=np.ravel(x_grid),np.ravel(y_grid) # flatten these guys
z=0*x # z coord is zero in the x,y plane

# the question gives us how the charge decays, so it doesn't really matter what my base charge is so long as python can handle it with float64. Note that it goes from e^0 to e^-1/sqrt 2 so in the middle the charge density is order Q/len x, and in the corers its like 0.1 Q/len x
q=np.array([charge_density(i,j) for i,j in zip(x,y)])
print('the total charge is {}'.format(np.sum(q)))

# now we define the y,z plane, with sidelength 6R, so 'radius 3R', lets call the coords u,v,w instead of x,y,z so as not to mix them up
arr = np.linspace(-2.3*R,2.3*R,REZ)
v_grid,w_grid=np.meshgrid(arr,arr)
v,w=np.ravel(v_grid),np.ravel(w_grid)
u=0.5*R/REZ*np.ones(len(v)) # this is so that things don't blow up to infinity, no colliding points!


# calculate the potential at all points and store these numbers in pot array
pot = np.array([calc_potential(r,x,y,z,q) for r in zip(u,v,w)])
pot = np.reshape(pot,(REZ,REZ))
# trace print("here is part of the potential {}".format(pot[50][20:45]))
# trace input("\n\n...\n\n")
plt.figure()
plt.contourf(pot,levels=1000)
plt.colorbar()
plt.title("POTENTIAL PLOT\nGAUSSIAN DISTRIBUTED CHARGE CROSS SECTION")
plt.savefig("potential_plot_guassian_distrubuted_charge_cross_section.png")
plt.show()


# calculate the electric field at all points and store these numbers in multidimensional array
REZ_QUIVER=24
arr = np.linspace(-2.3*R,2.3*R,REZ_QUIVER)
v_grid,w_grid=np.meshgrid(arr,arr)
v,w=np.ravel(v_grid),np.ravel(w_grid)
u=0.5*R/REZ*np.ones(len(v)) # prevent blow up

#e_field = np.array([calc_field(r,x,y,z,q)[1:] for r in zip(u,v,w)]) # we omit the x direction by indexing [1:] cause we in y,z plane (or we omit u cause we in v,w plane)
e_field = np.array([calc_fake_field(r,x,y,z,q)[1:] for r in zip(u,v,w)]) # this fake field is for quiver plotting purposes, so that the quiver plot looks nice to a human

# for nice colormap arrows, we also calculate the potential at thie meshgrid
pot = np.array([calc_potential(r,x,y,z,q) for r in zip(u,v,w)]).reshape(REZ_QUIVER,REZ_QUIVER)

fv = np.reshape(e_field.T[0],(REZ_QUIVER,REZ_QUIVER))
fw = np.reshape(e_field.T[1],(REZ_QUIVER,REZ_QUIVER))

# quiver plot
plt.figure()
print("shape of quiver is {}".format(e_field.T.shape))
plt.quiver(fv,fw,pot)
plt.title("E FIELD PLOT\nGAUSSIAN DISTRIBUTED CHARGE DENSITY CROSS SECTION")
plt.savefig("e_field_plot_gaussian_distributed_charge_cross_section.png")
plt.show()



