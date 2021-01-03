"""calculate the vector potential anywhere in apce given an arbitrary current loop.
We approximate the loop with a bunch of line segments, then add up all the vector potentials of all the line segments, which we found the formula for in problem 2"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sin = math.sin
cos = math.cos
PI = math.pi
ln = math.log
sqrt = math.sqrt
dot = np.dot
# I = mu0 = 4PI = 1

def cylindrical_to_cartesian(z,r,theta):
	return r*cos(theta),r*sin(theta),z # x,y,z

def polar_to_cartesian(r,theta,phi):
	return r*sin(theta)*cos(phi) , r*sin(theta)*sin(phi) , r*cos(theta) # x,y,z

def circle_loop(r=1,n=50): # returns circle loop in x,y plane
	theta = np.linspace(0,2*PI,n+1) 
	loopx = [r*cos(t) for t in theta] + [r*cos(theta[0])]
	loopy = [r*sin(t) for t in theta] + [r*sin(theta[0])]
	loopz = [0 for i in range(n+2)]
	return loopx,loopy,loopz 

def long_line_segment(z=100):
	return [0,0],[0,0],[-z,z]

def line(z=100):
	return np.zeros(20),np.zeros(20),np.linspace(-z,z,20)

def square_loop(r=1):
	return circle_loop(r,n=4)

def pentagon_loop(r=1):
	return circle_loop(r,n=5)

def hexagon_loop(r=1):
	return circle_loop(r,n=6)

def make_cube_grid(d=0.1,s=5): # n is number of points in grid, s is length of cube
	xarr,yarr,zarr = np.meshgrid(np.arange(-s,s,d),
			      	     np.arange(-s,s,d),
			   	     np.arange(-s,s,d))
	return xarr,yarr,zarr 

def norm(vec):# takes numpy vector
	return sqrt(sum([i**2 for i in vec]))

def crossp(a,b): # a and b must be 3d vectors
	x = a[1]*b[2] - a[2]*b[1]
	y = a[2]*b[0] - a[0]*b[2]
	z = a[0]*b[1] - a[1]*b[0]
	return np.array([x,y,z])

def get_vec_pot_at(loopx,loopy,loopz,x,y,z): # returns A, the vector potential at x,y,z
	"""A from line segment"""
	def vec_pot_line_seg(p1,p2,x,y,z):# p1 and p2 (3-vec) are the ends of line segments and x,y,z is where you are atm, 
		# define conveniant local coords, zhat, yhat, xhat is their cross product
		# print("p1,p2,x,y,z\t{},{},{},{},{}".format(p1,p2,x,y,z)) # trace
		zhat = (p2-p1) / norm(p2-p1)
		yvec = np.array([x,y,z]) - p1
		try:
			yvec = yvec - yvec * dot(yvec,zhat) / norm(yvec)
			yhat = yvec / norm(yvec) 
		except:
			print("problem!!, exception caught in vec_pot_line_seg")
			yhat = np.array([0.0,1.0,0.0]) # temporary
		xhat = crossp(yhat,zhat)
		
		# find the vector potential from equation in question 2, it's in the zhat direction
		# print("xhat,yhat,zhat\t{},{},{}".format(xhat,yhat,zhat)) # trace
		p1xyz = p1 - np.array([x,y,z])
		x1,y1,z1 = dot(p1xyz,xhat),dot(p1xyz,yhat),dot(p1xyz,zhat)
		p2xyz = p2 - np.array([x,y,z])
		x2,y2,z2 = dot(p2xyz,xhat),dot(p2xyz,yhat),dot(p2xyz,zhat) # if all went well x1 and x2 should be 0
		# print("x1 and x2 should  e zero : x1:{},x2:{}".format(x1,x2)) # trace
		try:
			a_seg = ln(z2 + sqrt(z2**2 + y2**2)) - ln(z1 + sqrt(z1**2 + y1**2))
		except:
			print("error encountered, printing\nz2,z1 should be positive:\n{}\n{}".format(z2,z1))
			print("set a_seg to zero")
			a_seg = 0.0
		return a_seg * zhat
		# returns 3 vector

	a_field = np.array([0.0,0.0,0.0])
	for i in range(len(loopx)-1):
		p1 = np.array([loopx[i],loopy[i],loopz[i]])
		p2 = np.array([loopx[i+1],loopy[i+1],loopz[i+1]])
		a_field += vec_pot_line_seg(p1,p2,x,y,z)
	return a_field	



def get_b(loop,x,y,z,d=0.001):
	loopx,loopy,loopz = loop()
	a000 = get_vec_pot_at(loopx,loopy,loopz,x,y,z)
	a100 = get_vec_pot_at(loopx,loopy,loopz,x+d,y,z)
	a010 = get_vec_pot_at(loopx,loopy,loopz,x,y+d,z)
	a001 = get_vec_pot_at(loopx,loopy,loopz,x,y,z+d)
	bx = (a010[2] - a000[2])/d - (a001[1] - a000[1])/d
	by = (a001[0] - a000[0])/d - (a100[2] - a100[2])/d
	bz = (a100[1] - a000[1])/d - (a010[0] - a000[0])/d
	
	cutoff = 7
	if norm([bx,by,bz]) < cutoff: # if it's not too big
		return np.array([bx,by,bz])
	return np.zeros(3)


def plot_loop(loopx,loopy,loopz):
	fig = plt.figure()
	ax = fig.gca(projection='3d')

	ax.plot(loopx,loopy,loopz)
	plt.show()
	return	

def calculate_potential_field(loopx,loopy,loopz,xarr,yarr,zarr):
	# loopx,loopy,loopz should be same shape
	# xarr, yarr, zarr should be same shape
	# returns uarr,varr,warr - the quiver plot xyz, same shape as (x|y|z)arr (tanjent vectors)
	shape = xarr.shape
	if yarr.shape!=shape or zarr.shape!=shape:raise Exception("The shapes of grid are not the same! incompatible!")
	if len(loopx)!= len(loopy) or len(loopy)!= len(loopz):raise Exception("Loops not good.")
	uarr,varr,warr = np.zeros(shape),np.zeros(shape),np.zeros(shape); 
	for i in range(shape[0]):
		for j in range(shape[1]):
			for k in range(shape[2]):
				uvw = get_vec_pot_at(loopx,loopy,loopz,xarr[i,j,k],yarr[i,j,k],zarr[i,j,k])
				uarr[i,j,k],varr[i,j,k],warr[i,j,k] = uvw[0],uvw[1],uvw[2]
	return uarr,varr,warr

def calculate_magnetic_field(loop,xarr,yarr,zarr):
	shape = xarr.shape
	uarr,varr,warr = np.zeros(shape),np.zeros(shape),np.zeros(shape); 
	for i in range(shape[0]):
		for j in range(shape[1]):
			for k in range(shape[2]):
				uvw = get_b(loop,xarr[i,j,k],yarr[i,j,k],zarr[i,j,k])
				uarr[i,j,k],varr[i,j,k],warr[i,j,k] = uvw[0],uvw[1],uvw[2]
	return uarr,varr,warr
	
def quiver_plot(xarr,yarr,zarr,uarr,varr,warr):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.quiver(xarr,yarr,zarr,uarr,varr,warr,length=0.2,normalize=True)
	plt.show()
	return

if __name__ == "__main__":
#	xarr,yarr,zarr = make_cube_grid(s=2)
	configurations = {circle_loop.__name__:circle_loop,
			square_loop.__name__:square_loop,
			long_line_segment.__name__:long_line_segment,
			pentagon_loop.__name__:pentagon_loop,
			hexagon_loop.__name__:hexagon_loop,
			line.__name__:line}
#	loopx,loopy,loopz = circle_loop(n=4)
#	loopx,loopy,loopz = long_line_segment()
#	get_vec_pot_at(loopx,loopy,loopz,xarr,yarr,zarr)

	print("Here are the loops I made:")	
	print(configurations.keys())
	print("What kind of loop do you want?")
	loopname = input("enter loopname: ")
	loop = configurations[loopname]
	loopx,loopy,loopz = loop()
	choice = 0
	while choice not in [1,2]:
		choice = int(input("do you want quiver plots (2) or get a specific field (1)?: ")) 
	if choice ==1:
		print("\nNow enter the specific coordinates you want me to evaluate the vector potential and magnetic field at?")
		x = float(input("enter x coord: "))
		y = float(input("enter y coord: "))
		z = float(input("enter z coord: "))

		vector_potential = get_vec_pot_at(loopx,loopy,loopz,x,y,z)
	
		print("\nThe vector potential at {},{},{} is \n{}".format(x,y,z,vector_potential))	
		b = get_b(loop,x,y,z)
		print("\nThe magnetic field at {},{},{} is\n{}".format(x,y,z,b))
	
	elif choice==2:
		# quiver plot magnetic field
		s,d=0.7,0.2
		xarr,yarr,zarr = make_cube_grid(s=s,d=d)
		mode=input("display mag field (b) or potential (a):")
		if mode == 'm':
			uarr,varr,warr = calculate_magnetic_field(loop,xarr,yarr,zarr)
			cmap='Greens'
			label='Mag field'
		else:
			loopx,loopy,loopz = loop()
			cmap='Reds'
			label='Mag potential'
			uarr,varr,warr = calculate_potential_field(loopx,loopy,loopz,xarr,yarr,zarr)
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		#o = np.sqrt(uarr**2 + varr**2 + zarr**2)
		#o = np.random.random(uarr.shape)
		c = np.hypot(np.hypot(uarr,varr),warr)
		if loop.__name__=="line":q=ax.quiver(xarr,yarr,zarr,uarr,varr,warr,cmap='Greens',alpha=0.5,length=2.0,lw=2)
		else: q=ax.quiver(xarr,yarr,zarr,uarr,varr,warr,cmap=cmap,alpha=0.5,length=0.05,lw=2,label=label)
		#q.set_array(np.random.rand(np.prod(xarr.shape)))
		q.set_array(c.flatten())
		
		if loop.__name__=="line":plt.plot([0,0],[0,0],[-s,s],color='b')
		else:plt.plot(loopx,loopy,loopz,color='b')
		plt.legend()	
		plt.show()


#	s,d=2,0.4
#	xarr,yarr,zarr = np.meshgrid(np.arange(-s,s,d),
#				np.arange(-s,s,d),
#				np.arange(-s,s,d))
#	u,v,w = [],[],[]
#	for i in np.arange(-s,s,d):
#		for j in np.arange(-s,s,d):
#			karr = []
#			for k in np.arange(-s,s,d):
#				# strangely i=y,j=x,k=z
#				vec_pot = get_vec_pot_at(loopx,loopy,loopz,j,i,k)
#				karr.append(vec_pot)
#			jarr.append(karr)
#		iarr.append(jarr)
	
#	# test the quiver plot
#	fig = plt.figure()
#	ax = fig.gca(projection='3d')

#	# makegrid
#	x,y,z = np.meshgrid(np.arange(-0.8,1,0.3),
#					np.arange(-0.8,1,0.3),
#					np.arange(-0.8,1,0.3))	 	
	
#	# make the direction data for arrows
#	u = np.sin(PI*x) * np.cos(PI*y) * np.cos(np.pi * z)
#	v = -np.cos(PI*x) * np.sin(PI*y) * np.cos(PI*z)
#	w = (np.sqrt(2./3.)*np.cos(np.pi*x) * np.cos(np.pi*y) * np.sin(np.pi*z))

#	ax.quiver(x,y,z,u,v,w,length=0.1,normalize=True)
#	plt.show()




