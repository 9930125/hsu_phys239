#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 00:52:52 2016

@author: dinohsu
"""
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt

#define constants in SI units
e = float(1.6e-19)
m_e = float(9.11e-31)
c = float(3e8)
k = float(1.38e-23)
h = float(6.62e-34)
L_sun = float(3.828e26)
A = float(1e-10) 

#Set up the parameters for M82
D = float(3.63e6 * 3.0856e16)

#read the M82 data first
def readdat():
    dat = np.loadtxt("m82spec.dat",skiprows=1)
    plt.figure()
    plt.loglog(dat[:, 0], dat[:, 1])
    plt.xlabel(r'Wavelength ($\mu m$)', size=16)
    plt.ylabel(r'$L_\nu$ ($L_\odot / Hz$)', size=16)
    plt.title('M82 Observed spectrum', size=20)
    return dat

print readdat()

#for the starlight
#I adopt the 1999 dataset on the stardust99 website
def starl_emi():
    dat = np.loadtxt("fig7c.dat.txt",skiprows=3)
    starl = 10**(dat[:,-5]) / L_sun * dat[:,0]**2 * 10**(-12) *1e-2 / c #convert in the unit of L_sun
    starl = starl.reshape(-1,1)
    wavel = dat[:,0] * 1e-4
    wavel = wavel.reshape(-1,1)
    plt.figure()
    plt.loglog(wavel,starl)
    plt.xlabel(r'Wavelength ($\mu m)', size=16)
    plt.ylabel(r'$L_\nu$ ($L_\odot / Hz$)', size=16)
    plt.title('Star light', size=20)
    return np.concatenate((wavel, starl), axis=1)
    
starl = starl_emi()
    
#for the dust emission
def bb_fn(T,nu):
    return 2*h*nu**3/c**2/(np.exp(h*nu/k/T-1))

#parameters for the dust emission
Q_abs = 
a = 
rho = 
k_nu = float(0.75*Q_abs/a/rho)
M_dust = float(7.5e6*1.9891e30)
T_dust = 30
    
def dust_rad(T,nu):
    return M_dust * k_nu * bb_fn(T,nu) / D**2

#for the synchrotron radiation, refer to eq. 6.36 in the textbook
def sync_rad(w,C,p,B,sina):
    C = float(11.51253504*1e29)
    return C * np.sqrt(3)* e**3 * C * B * sina /2 /np.pi /m_e /c**2 /(p+1) \
           * gamma(p/4 + 19/12) * gamma(p/4 - 1/12) \
           * (m_e * c * w / (3*e*B*sina))**((1-p)/2) *1e-11

#parameters for synchrotron radiation
B = float(200e-6)
sina = float(np.sqrt(2))
p = float(2.25)
C = float(1)

wavel = np.loadtxt("m82spec.dat",skiprows=1)
wavel = wavel[:,0]
freq = c * 1e6 / wavel
#freq = freq[:,0]
plt.figure()
plt.loglog(wavel,sync_rad(freq,C,p,B,sina))
plt.xlabel(r'Wavelength ($\mu m$)', size=16)
plt.ylabel(r'$L_\nu$ ($L_\odot / Hz$)', size=16)
plt.title('Synchrotron radiation', size=20)
plt.show()

           
#for the free-free emission
#I refer to eq 5.14a and Fig 5.2 to determine which function I am going to use
#I use Large-angle region, Gaunt gactor ~1
#C4 = float(32*np.pi* e**6/3/m_e/c**3)
def ff_rad(T,nu,Z, n_e, n_i):
    C = float(11.51253504*1e28)
    return C * 68e-39 * Z**2 * n_e * n_i * T**(-0.5) *np.exp(-(h * nu /k /T)) *1e17

    #    return C4 * T**(-0.5) * Z**2 * n_e * n_i *np.exp(-(h*nu/k/T))
    
#parameters for free-free emission
T = float(1e2)
Z = float(1)
n_e = float(210e-9)
n_i = float(210e-9)

plt.figure()
plt.loglog(wavel,ff_rad(T,freq,Z, n_e, n_i))
plt.xlabel(r'Wavelength ($\mu m$)', size=16)
plt.ylabel(r'$L_\nu$ ($L_\odot / Hz$)', size=16)
plt.title('Thermal Free-Free Emission', size=20)
plt.show()


#Put all the plots together
dat = np.loadtxt("m82spec.dat",skiprows=1)
plt.figure()
plt.loglog(dat[:, 0], dat[:, 1],label="M82 data")
plt.loglog(starl[:,0],starl[:,1],label="star light")
plt.loglog(wavel,sync_rad(freq,C,p,B,sina),label="synchrotron radiation")
plt.loglog(wavel,ff_rad(T,freq,Z, n_e, n_i),label="free-free emission")
plt.xlabel(r'Wavelength ($\mu$ m)', size=16)
plt.ylabel(r'$L_\nu$ ($L_\odot / Hz$)', size=16)
plt.legend(fontsize=10, loc=4)
plt.xlim([np.min(dat[:, 0]), np.max(dat[:, 0])])
plt.ylim(ymin=1e-10)
plt.title('Final Spectra', size=20)
plt.show()

    

