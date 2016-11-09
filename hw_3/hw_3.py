# Homework 3 for PHYS 239
# 
#history
#20161024 dino first built
#20161108 dino finish for the v-b plot
##############################################


# import some libraries for calculating and plotting firgures
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
x0 = float(-5000*5.29e-11) # in unit of Bohr radius
b = float(800*5.29e-11)         # in unit of Bohr radius
Z = float(82)        # atomic number 
v0 = float(1e6)     # in unit of m/s
nstep = int(5000)
dt = -2. * float(x0/v0/nstep)     # in steps of 2000 (total length 2000*e-8)


# Now calculate the interation force via Coulomb's Law


x = np.zeros((nstep, 2))
r = np.zeros((nstep, 1))
v = np.zeros((nstep, 2))
a = np.zeros((nstep, 2))
t = np.zeros((nstep, 1))
x[0] = np.array([x0, b])
t[0] = 0
r = [np.sqrt(np.sum(x[0]**2))]
v = [np.array([v0,0])]

def _acceleration(x,r):
    p = -253.2638423 * Z / r**2  *  x/r
    return p

a = [_acceleration(x[0],r[0])]
#print _acceleration(x[0],r[0])

# Now add time variation
for i in xrange (1,nstep):
    x[i] = x[i-1] + v[i-1] * dt + 1./2 * a[i-1] * dt**2
    r.append(np.sqrt(np.sum(x[i]**2)))
    v.append(v[i-1] + a[i-1] * dt)
    a.append(_acceleration(x[i],r[i]))
    t[i] = t[i-1] + dt
    
x = np.array(x)
r = np.array(r)
v = np.array(v)
a = np.array(a)
t = np.array(t)

z = x*1e10

plt.figure()
plt.title("Path of the Electron",size=20)
#plt.scatter(x[:,0],x[:,1],s=1)
plt.xlabel(r'x ($10^{-10} m$)',size=14)
plt.ylabel(r'y ($10^{-10} m$)',size=14)
plt.scatter(z[:,0], z[:, 1], s=1)
plt.xlim(np.min(x[:,0]), np.max(x[:,0]))
plt.ylim(np.min(x[:,1]), np.max(x[:,1]))   
plt.axis('equal')
plt.legend()
plt.show()

#Plot x-t
plt.figure()
plt.title( r"Position(x) vs. time plot",size=20 )
plt.xlabel(r't ($s$)',size=14)
plt.ylabel(r'x ($10^{-10} m$)',size=14)
plt.plot(t,z[:,0])
plt.show()

#Plot y-t
plt.figure()
plt.title( "Position(y) vs. time plot",size=20 )
plt.xlabel(r't ($s$)',size=14)
plt.ylabel(r'y ($10^{-10} m$)',size=14)
plt.plot(t,z[:,1])
plt.show()

#Plot Vx-t
plt.figure()
plt.title( r"Velocity(x) vs. Time",size=20 )
plt.xlabel(r'$t (s)$',size=14)
plt.ylabel(r'$V_{x}$ $(m/s)$',size=14)
plt.plot(t,v[:,0])
plt.show()

#Plot Vy-t
plt.figure()
plt.title( "Velocity(y) vs. Time",size=20 )
plt.xlabel(r'$t (s)$',size=14)
plt.ylabel(r'$V_{y}$ $(m/s)$',size=14)
plt.plot(t,v[:,1])
plt.show()

#Plot Ax-t
plt.figure()
plt.title( "Acceleration(x) vs. Time",size=20 )
plt.xlabel(r'$t (s)$',size=14)
plt.ylabel(r'$A_{x}$ $(m/s^{2})$',size=14)
plt.plot(t,a[:,0])
plt.show()

#Plot Ay-t
plt.figure()
plt.title( "Acceleration(y) vs. Time",size=20 )
plt.xlabel(r'$t (s)$',size=14)
plt.ylabel(r'$A_{y}$ $(m/s^{2})$',size=14)
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
plt.title("Power Spectrum",size=20)
plt.xlabel(r'frequency ($Hz$)',size=14)
plt.ylabel(r'power ($a_{x}^{2} + a_{y}^{2}$)',size=14)
plt.xlim(0,np.max(freq)/50)
plt.plot(freq,afft)
plt.show()

#pmax = [freq,afft]
#p = np.array(pmax)
#p[:,np.max(afft)]
def _maxfreq(b,v00):
    #x = np.zeros((nstep, 2))
    #r = np.zeros((nstep, 1))
    #v = np.zeros((nstep, 2))
    #a = np.zeros((nstep, 2))
    #t = np.zeros((nstep, 1))
    x = [np.array([x0, b])]
    t = 0
    r = [np.sqrt(np.sum(x[0]**2))]
    v = [np.array([v00,0])]
    a = [np.array(_acceleration(x[0],r[0]))]
    for i in xrange (1,nstep):
        x.append(x[i-1] + v[i-1] * dt + 1./2 * a[i-1] * dt**2)
        r.append(np.sqrt(np.sum(x[i]**2)))
        v.append(v[i-1] + a[i-1] * dt)
        a.append(_acceleration(x[i],r[i]))
        t = t + dt

    x = np.array(x)
    r = np.array(r)
    v = np.array(v)
    a = np.array(a)
    t = np.array(t)
    axfft = np.fft.fft(a[:,0])
    ayfft = np.fft.fft(a[:,1])
    afft = np.abs(axfft)**2 + np.abs(ayfft)**2
    return np.abs(freq[np.argmax(afft)])

#print _maxfreq(b=800*5.29e-11,v00=1e6)

#set up a range of impact parameters and initial velocities
bmin = 200*5.29e-11
bmax = 1000*5.29e-11
v0min = 6e5
v0max = 1e6
n = 50
bs  = np.logspace(np.log10(bmin),np.log10(bmax),n)
v0s = np.logspace(np.log10(v0min),np.log10(v0max),n)
fmax = np.full([n,n],np.nan)

for i in range (n):
    print "Now it's", i , "time"
    for j in range (n):
        fmax[i,j] = _maxfreq(bs[i],v0s[j])

bv = [np.log10(min(v0s)),np.log10(max(v0s)),np.log10(min(bs)),np.log10(max(bs))]
      
plt.figure()
plt.title('Peak Frequencies',size=20)
plt.xlabel(r'Initial Velocities $v_{0}$ $log{v_{0}}$ ($m/s$)',size=14)
plt.ylabel(r'Impact Parameters b $log{b}$ ($m$)',size=14)
plt.imshow(np.log10(fmax),origin='lower',extent=bv,aspect="auto")
plt.colorbar()

