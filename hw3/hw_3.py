# Homework 3 for PHYS 239
# 
#history
#20161024 dino first built
##############################################


# import some libraries for calculating and plotting firgures
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
x0 = float(-10000*5.29e-11) # in unit of Bohr radius
b = float(800*5.29e-11)         # in unit of Bohr radius
Z = float(82)        # atomic number 
v0 = float(1e6)     # in unit of m/s
nstep = int(5000)
dt = -float(x0/v0/nstep)     # in steps of 2000 (total length 2000*e-8)


# Now calculate the interation force via Coulomb's Law


x = np.zeros((nstep, 2))
r = np.zeros((nstep, 1))
v = np.zeros((nstep, 2))
a = np.zeros((nstep, 2))
t = np.zeros((nstep, 1))
x[0] = np.array([x0, b])
t[0] = 0
print(x[0])
r = [np.sqrt(np.sum(x[0]**2))]
v = [np.array([v0,0])]
print(r[0])
print(v[0])

def _acceleration(x,r):
    p = -253.2638423 * Z / r**2  *  x/r
    return p

a = [_acceleration(x[0],r[0])]
print _acceleration(x[0],r[0])

print(type(a[0]))
#a = [np.array( Ze2 / r / r / me * x/r )]


# Now add time variation
for i in xrange (1,nstep):
    x[i] = x[i-1] + v[i-1] * dt + 1./2 * a[i-1] * dt**2
    #x.append(x[i-1] + v[i-1] * dt + 0.5 * a[i-1] * dt**2)
    #r[i] = np.sqrt(np.sum(x[i]**2))
    r.append(np.sqrt(np.sum(x[i]**2)))
    #v[i] = v[i-1]
    v.append(v[i-1] + a[i-1] * dt)
    #a[i] = _acceleration(x[i-1],r[i-1])
    a.append(_acceleration(x[i],r[i]))
    t[i] = t[i-1] + dt
    
x = np.array(x)
r = np.array(r)
v = np.array(v)
a = np.array(a)
t = np.array(t)
#t = [np.array(i*dt for i in xrange (0,20001))]


#length=2000
#x = np.zeros((2000,1))

print x[2][0]
print x[2]

z = x*1e10


plt.figure()
plt.title( r"$ Path of the Electron $  ")
#plt.plot(x[:][0],x[:,1])
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(z[:,0], z[:, 1], s=1)
plt.xlim(np.min(x[:][0]), np.max(x[:,0]))
#plt.xlim(5.29e-09, 5.292645e-09)
plt.ylim(np.min(x[:,1]), np.max(x[:,1]))   
plt.axis('equal')
#plt.xlim(-2000,2000)
#plt.ylim(-2000,2000)
plt.legend()
plt.show()

#Plot x-t
plt.figure()
plt.title( "Position(x) vs. time plot" )
plt.xlabel('t')
plt.ylabel('x')
plt.plot(t,z[:,0])
plt.show()

#Plot y-t
plt.figure()
plt.title( "Position(y) vs. time plot" )
plt.xlabel('t')
plt.ylabel('y')
plt.plot(t,z[:,1])
plt.show()

#Plot Vx-t
plt.figure()
plt.title( "Velocity(x) vs. time plot" )
plt.xlabel('t')
plt.ylabel('Vx')
plt.plot(t,v[:,0])
plt.show()

#Plot Vy-t
plt.figure()
plt.title( "Velocity(y) vs. time plot" )
plt.xlabel('t')
plt.ylabel('Vy')
plt.plot(t,v[:,1])
plt.show()

#Plot Ax-t
plt.figure()
plt.title( "Acceleration(x) vs. time plot" )
plt.xlabel('t')
plt.ylabel('Ax')
plt.plot(t,a[:,0])
plt.show()

#Plot Ay-t
plt.figure()
plt.title( "Acceleration(y) vs. time plot" )
plt.xlabel('t')
plt.ylabel('Ay')
plt.plot(t,a[:,1])
plt.show()

#Fourier transform of acceleration 
freq = np.fft.fftfreq(nstep,dt)
axfft = np.fft.fft(a[:,0])
ayfft = np.fft.fft(a[:,1])
afft = np.abs(axfft)**2 + np.abs(ayfft)**2
#freq, afft = freq[:1],afft[:1]

#plot power spectrum
plt.figure()
plt.title("Power Spectrum vs. freq")
plt.xlabel('freq')
plt.ylabel('power')
plt.xlim(0,np.max(freq))
plt.plot(freq,afft)
plt.show()
