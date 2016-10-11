########################################################################
# homework code for PHYS 239
# 20161008 dino first built
########################################################################

#########################
# We start with problem 1"

print "############## For the Problem 1: ###################"

pc = float( 3.085e18 ) #define the unit parsec
D = float( 100 * pc )  #set up the cloud of depth D
n = float(1)           #set up the number density n

# define column density CD
CD = n * D

print "  "
print 'The column density = ', CD, 'cm^(-2).'
print "  "

# define optical depth: od1, od2, od3
od1 = float( 1e-3 ) #optical depth for a: 10^(-3)
od2 = float( 1 )    #optical depth for b: 1
od3 = float( 1e3 )  #optical depth for c: 10^3

# define the function for cross sections for given optical depth
def _cs( od, n, D):
    return float( od / n / D );

print "  "
print ' The cross section for optical depth', od1, 'is', _cs(od1, n, D), 'cm^(-2).'
print ' The cross section for optical depth', od2, 'is', _cs(od2, n, D), 'cm^(-2).'
print ' The cross section for optical depth', od3, 'is', _cs(od3, n, D), 'cm^(-2).'
print "  "

#########################
#Now for problem 2
print "############## For the Problem 2: ###################"
print "## Please see the function _intensity in the code ###"



# import some libraries for calculating and plotting firgures
import numpy as np
import matplotlib.pyplot as plt

#define the function intensity with the following parameters:
#       initial intensity I0, source function S, D, total integration number 
#       nstep, number density n, and cross section cs

def _intensity ( I0, S, D, nstep, n, cs ):
    ds = D / nstep # distance travelled per step
    I = np.full_like( freq, I0) #define an array for I, the same as that for frequency
    for i in range(nstep):
        I += ( -I + S ) * n * cs * ds # set up for loop of differential equation I
    return I ;
    
print "  "
print "def _intensity ( I0, S, D, nstep, n, cs ):"
print "    ds = D / nstep "
print "    I = np.full_like( freq, I0) "
print "    for i in range(nstep):"
print "        I += ( -I + S ) * n * cs * ds"
print "    return I ;"
print "  "

# Now setup a spectral line with gaussian line shape
print "############## For the Problem 3: ###################"

# define the function for cross section as a fuction of frequency "freq",
#        centered at freq0 with maximum cross section "amp",
#        and the FWHM for the gaussian is "width"
def _gaussian_line ( freq, freq0, width, amp):
    return amp * np.exp( - ( freq - freq0)**2 / ( 2 * width**2) );

print "###The function for cross section is shown below: ###"
print "  "
print "def _gaussian_line ( freq, freq0, width, amp):"
print "    return amp * np.exp( - ( freq - freq0)**2 / ( 2 * width**2) );"
print "  "

###############################################
# Now get the plots for cross sections of the given three optical depths
print "The three plots for the derived cross sections in problem 1:"

# Setup the initial values

freq0 = 230 # GHz
nstep = 10000
width = 0.5 # GHz, i.e. 500 MHz
res = 1000 # spectral resolution

# define the frequency range with the spectral resolution "res"
freq = np.linspace( freq0 - 10 * width, freq0 + 10 * width, res )

# plot y-axis in the log scale because it's much easier to view them

x = freq
y1 = np.log10( _gaussian_line( freq, freq0, width, _cs(od1, n, D)) )
y2 = np.log10( _gaussian_line( freq, freq0, width, _cs(od2, n, D)) )
y3 = np.log10( _gaussian_line( freq, freq0, width, _cs(od3, n, D)) )

plt.figure()
plt.title(r"The Cross Sections $\sigma_{\nu}$")
plt.plot(x, y1, 'r-', label= r"$\tau = 10^{-3} $")
plt.plot(x, y2, 'b-', label= r"$\tau = 1 $")
plt.plot(x, y3, 'g-', label= r"$\tau = 10^3 $")
plt.xlabel('Frequencies (GHz)')
plt.ylabel('Cross Section (log10 * cm^(-2))')
plt.legend()
plt.show()

print "  "
####################################################
#Now plot for problem 4
print "############## For the Problem 4: ###################"

# for (a) choose I0=3, S=3 and od=1000
print "  "
print 'The case (a):'

od = float(1000) #optical depth approaches infinity
I0 = float(3) #in Jy
S = float(5) #in Jy
I = _intensity ( I0, S, D, nstep, n, _cs( od, n, D) )
plt.figure()
plt.title( r"$ \tau_{\nu} (D) >> 1$  ")
plt.plot(freq, [I0] * res, 'b--', label=r"$I(0)$")
plt.plot(freq, [S] * res, 'g--', label=r"$S_{\nu}$")
plt.plot(freq, I, 'r-', label=r"$I_{\nu}$")
plt.xlabel('Frequency (GHz)')
plt.ylabel('Intensity (Jy)')
plt.ylim(-0.5,10)
plt.legend()
plt.show()

# for (b) choose I0=0, S=5 and od=0.5
print "  "
print 'The case (b):'

od = float(0.5)
I0 = float(0) #in Jy
S = float(5) #in Jy
amp = _cs(od, n, D)
cs = _gaussian_line( freq, freq0, width, amp )
I = _intensity ( I0, S, D, nstep, n, cs )
plt.figure()
plt.title(r"$ I_{\nu} (0)=0$"" $and$ " r"$\tau_{\nu} (D) < 1 $")
plt.plot(freq, [I0] * res, 'b--', label=r"$I(0)$")
plt.plot(freq, [S] * res, 'g--', label=r"$S_{\nu}$")
plt.plot(freq, I, 'r-', label=r"$I_{\nu}$")
plt.xlabel('Frequency (GHz)')
plt.ylabel('Intensity (Jy)')
plt.ylim(-0.5,10)
plt.legend()
plt.show()

# for (c) choose I0=3, S=5 and od=0.5
print "  "
print 'The case (c):'

od = float(0.5)
I0 = float(3) #in Jy
S = float(5) #in Jy
amp = _cs(od, n, D)
cs = _gaussian_line( freq, freq0, width, amp)
I = _intensity ( I0, S, D, nstep, n, cs )
plt.figure()
plt.title(r"$ I_{\nu} (0)<S_{\nu}$"" $and$ " r"$\tau_{\nu} (D) < 1 $")
plt.plot(freq, [I0] * res, 'b--', label=r"$I(0)$")
plt.plot(freq, [S] * res, 'g--', label=r"$S_{\nu}$")
plt.plot(freq, I, 'r-', label=r"$I_{\nu}$")
plt.xlabel('Frequency (GHz)')
plt.ylabel('Intensity (Jy)')
plt.ylim(-0.5,10)
plt.legend()
plt.show()

# for (d) choose I0=7, S=5 and od=0.5
print "  "
print 'The case (d):'

od = float(0.5)
I0 = float(7) #in Jy
S = float(5) #in Jy
amp = _cs(od, n, D)
cs = _gaussian_line( freq, freq0, width, amp )
I = _intensity ( I0, S, D, nstep, n, cs )
plt.figure()
plt.title(r"$ I_{\nu} (0)>S_{\nu}$"" $and$ " r"$\tau_{\nu} (D) < 1 $")
plt.plot(freq, [I0] * res, 'b--', label=r"$I(0)$")
plt.plot(freq, [S] * res, 'g--', label=r"$S_{\nu}$")
plt.plot(freq, I, 'r-', label=r"$I_{\nu}$")
plt.xlabel('Frequency (GHz)')
plt.ylabel('Intensity (Jy)')
plt.ylim(-0.5,10)
plt.legend()
plt.show()

# for (e) choose I0=3, S=5 and od=0.5
print "  "
print 'The case (e):'

od = float(0.5)
I0 = float(3) #in Jy
S = float(5) #in Jy
amp = _cs(od, n, D) * 10
cs = _gaussian_line( freq, freq0, width, amp )
I = _intensity ( I0, S, D, nstep, n, cs )
plt.figure()
plt.title(r"$ I_{\nu} (0)<S_{\nu}$"" $and$ " r"$\tau_{\nu} (D) < 1 $"" $while$ " r"$\tau_{\nu,0}(D)>1$")
plt.plot(freq, [I0] * res, 'b--', label=r"$I(0)$")
plt.plot(freq, [S] * res, 'g--', label=r"$S_{\nu}$")
plt.plot(freq, I, 'r-', label=r"$I_{\nu}$")
plt.xlabel('Frequency (GHz)')
plt.ylabel('Intensity (Jy)')
plt.ylim(-0.5,10)
plt.legend()
plt.show()

# for (f) choose I0=7, S=5 and od=0.5
print "  "
print 'The case (f):'

od = float(0.5)
I0 = float(7) #in Jy
S = float(5) #in Jy
amp = _cs(od, n, D) * 10
cs = _gaussian_line( freq, freq0, width, amp )
I = _intensity ( I0, S, D, nstep, n, cs )
plt.figure()
plt.title(r"$ I_{\nu} (0)>S_{\nu}$"" $and$ " r"$\tau_{\nu} (D) < 1 $"" $while$ " r"$\tau_{\nu,0}(D)>1$")
plt.plot(freq, [I0] * res, 'b--', label=r"$I(0)$")
plt.plot(freq, [S] * res, 'g--', label=r"$S_{\nu}$")
plt.plot(freq, I, 'r-', label=r"$I_{\nu}$")
plt.xlabel('Frequency (GHz)')
plt.ylabel('Intensity (Jy)')
plt.ylim(-0.5,10)
plt.legend()
plt.show()
