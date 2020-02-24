# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:06:26 2020

@author: lilon
This program tries to fit the germanium quenching factor with the Lindhard model using MCMC technique.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm #for histogram fitting
#TUNL data only
RecoilE=np.array([0.815,0.931,1.114,1.745,1.978,1.991,2.316,2.370,4.205,4.895])
RecoilEerr=np.array([0.085,0.098,0.116,0.128,0.145,0.206,0.171,0.248,0.297,0.360])
QF=np.array([15.8,14.9,14.9,16.7,16.8,17.6,17.4,17.9,19.7,20.4])
QFerr=np.array([1.0,0.9,0.7,1.2,1.0,1.1,0.9,1.2,0.7,0.8])
#world data
JonesE=np.array([0.254,0.960,1.040,1.120,1.220,1.350,1.480,1.620,1.750])
JonesQF=np.array([15.4 ,17.6 ,21.5 ,24.6 ,22.3 ,25.9 ,22.6 ,26.9 ,18.6 ])
JonesQFerr=np.array([2.2 ,4.2 ,1.9 ,1.8 ,1.6 ,1.5 , 1.4 ,1.2 , 1.1 ])

BarbeauE=np.array([0.647 ,0.737 ,0.988 ,1.223  ])
BarbeauQF=np.array([17.3 ,18.5 , 19.1 ,19.8 ])
BarubeauQFerr=np.array([2.0,2.7,2.6,1.8])

TexonoE=np.array([1.365 ,3.218 ,3.613 , 7.923 ,8.269 ,12.096 ,19.634 ,22.706 ,29.270 ,34.027 ,35.523 , 48.914 ,63.071 ,68.695 ,86.877 ,87.511 ,92.939 ])
TexonoQF=np.array([21.2 ,24.0 ,24.0 ,26.5 ,25.0 ,25.9 ,28.1 ,29.3 ,29.5 ,29.8 ,31.2 ,31.7 ,33.3 ,34.5 , 33.7 ,36.2 ,35.1 ])
TexonoQFerr=np.array([3.1 ,2.0 ,1.8 , 1.9 ,1.8 ,1.2 ,1.1 , 1.5 ,1.7 , 1.0 , 1.1 ,1.5 , 1.2 , 1.6 , 2.0 ,1.6 ,1.4 ])

MessesE=np.array([2.709 ,5.189 ,8.791 ,13.204 ,16.051 ,21.503 ,26.724 ,32.392 ,37.276 ])
MessesQF=np.array([29.6 ,25.0 ,24.1 ,26.3 ,24.4 ,24.9 , 27.5 , 27.1 ,28.9 ])
MessesQFerr=np.array([11.3 ,6.1 , 0.5 , 0.5 ,0.5 , 0.8 ,1.7 ,0.0 ,0.3 ])

ChasmanE=np.array([10.0 ,17.6 ,19.4 ,22.5 ,24.1 ,29.8 ])
ChasmanQF=np.array([20 ,23 ,24 , 25 , 27 ,28 ])
ChasmanQFerr=np.array([2,3,3,2,2,2])

SattlerE=np.array([21.4 ,26.8 ,32.2 ,32.2 ,37.5 ,42.9 ,48.2 ,54.3 ,80.4 ])
SattlerQF=np.array([14.9 , 17.5 , 21.5 , 22.4 , 22.3 , 23.4 , 24.9 , 23.5 , 28.1 ])
SattlerQFerr=np.array([5.7 ,2.7 , 2.2 , 2.2 ,3.0 ,2.7 ,2.3 ,2.1 , 1.5 ])

SimonE=np.array([94,80,76])
SimonQF=np.array([34.7,32.8,30.1])
SimonQFerr=np.array([0.8,0.9,4.0])

ShuttE=np.array([17.6 ,22.5 ,27.4 ,35.0 , 45.1 ,55.1 , 70.0 ])
ShuttQF=np.array([25.1 , 27.8 ,28.4 , 30.7 , 31.6 ,33.2 ,34.2 ])
ShuttQFerr=np.array([0.9 ,0.6 , 0.6 ,0.3 , 0.6 , 1.0 , 1.0 ])

BaudisE=np.array([59.1 ,68.2 ,75.3 ,77.0 ,79.4 ,83.4 , 85.8 ,88.4 ,93.1 ,93.8 ])
BaudisQF=np.array([32.0 ,32.1 ,31.8 ,33.6 ,32.6 ,33.4 , 34.9 ,33.9 ,34.2 , 35.0 ])
BaudisQFerr=np.array([1.9 ,1.8 , 2.1 ,1.0 ,1.9 , 2.0 ,1.0 , 2.1 , 1.6 ,1.0 ])
#constants for lindhard QF from Lewin and Smith
Z=32 #Ge atomic number
A=72.6 #Ge
def LindhardQF(er,paras):
    f=paras
    eps=11.5*er*pow(Z,-7.0/3)
    ge=3*pow(eps,0.15)+0.7*pow(eps,0.6)+eps
    k=f*pow(Z,2.0/3)*pow(A,-0.5)
    return k*ge/(1+k*ge)

def LindhardQF_adiabaticcorrected(er,paras):
    f,ksi=paras
    Fac=1-np.exp(-er/ksi)
    return LindhardQF(er,f)*Fac

def LindhardQFtoP(er,paras):
    return 100*LindhardQF_adiabaticcorrected(er,paras)

def log_prior(paras):
    f,ksi=paras
    if 0<f<0.2 and 0<ksi<2:
        return 0.0
    return -np.inf

def log_likelihood(paras,x,y,yerr):
    #f,ksi=paras
    model=LindhardQFtoP(x,paras)
    sigma2=yerr**2
    return -0.5*np.sum((y-model)**2/sigma2+np.log(sigma2))

def log_probability(paras,x,y,yerr):
    lp=log_prior(paras)
    if not np.isfinite(lp):
        return -np.inf
    return lp+log_likelihood(paras,x,y,yerr)

import emcee
np.random.seed(1) #set the random seed for this program
nwalkers,ndim=32,2
f=np.random.uniform(0 ,0.2,nwalkers)
ksi=np.random.uniform(0 ,2,nwalkers)
paras=np.stack((f,ksi))
paras=paras.T
print(paras.shape)
sampler=emcee.EnsembleSampler(nwalkers,ndim,log_probability,args=(RecoilE,QF,QFerr))
sampler.run_mcmc(paras,5000,progress=True) #5000 steps

fig, axes = plt.subplots(2, figsize=(10, 5), sharex=True)
samples = sampler.get_chain()
labels = ["f", r"$\xi$"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:,:,i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
plt.savefig("MC chain plot Lindhard Paras.pdf")

tau = sampler.get_autocorr_time()
print("steps to forget ",tau) #get steps to forget the start

flat_samples = sampler.get_chain(discard=100,thin=10,flat=True)
print(flat_samples.shape)
arrf=np.array(flat_samples[:,0])
(fmu, fsigma) = norm.fit(arrf)
print("the mean and sigma of f is:",fmu,fsigma)

plt.figure(30)
n,bins,patches=plt.hist(arrf, 50, density=1, facecolor='b', alpha=0.5)
plt.xlabel("f")
plt.ylabel("Probability")
#plt.xlim(0.1,0.14)
plt.title("Histogram of Lindhard f")
plt.savefig("histo_Lindhardf.pdf")

import corner
cornerfig = corner.corner(
    flat_samples, labels=["f",r"$\xi$"],quantiles=[0.16,0.5,0.84],show_titles=True
)
plt.savefig("cornerplot Lindhard Paras.pdf")

plt.figure(40)
x0=np.linspace(0,6,100)
meanpara=np.mean(flat_samples,axis=0)
plt.plot(x0,LindhardQFtoP(x0,[fmu,0.21]),color='r',linestyle='-',alpha=1,label=r"k=0.142,$\xi$=0.21 TUNL")
plt.plot(x0,LindhardQFtoP(x0,[fmu+fsigma,0.21-0.15]),color='r',linestyle='--',alpha=1)
plt.plot(x0,LindhardQFtoP(x0,[fmu-fsigma,0.21+0.12]),color='r',linestyle='--',alpha=1)
plt.plot(x0,LindhardQFtoP(x0,[0.133,0.0000001]),color='#d00d5b',linestyle='-',alpha=1,label=r"k=0.157,$\xi$=0 Standard Lindhard")
plt.plot(x0,LindhardQFtoP(x0,[0.151,0.16]),color='#ffa24f',linestyle='-',alpha=1,label=r"k=0.1789,$\xi$=0.16 B.J.Scholz")
plt.legend(loc='lower right')

plt.errorbar(RecoilE, QF, xerr=RecoilEerr,yerr=QFerr, marker='D',color='#4a6bf5',linestyle='None', capsize=0)
plt.xlim(0, 6)
plt.xticks(np.linspace(0,6,7))
plt.xlabel("Nuclear recoil energy (keVnr)")
plt.ylabel("Quenching factor (\%)")
#plt.show()
plt.savefig("Ge QF and MCMC fit.pdf")

plt.figure(50)
x0=np.linspace(0,105,1050)
meanpara=np.mean(flat_samples,axis=0)
plt.plot(x0,LindhardQFtoP(x0,[fmu,0.21]),color='r',linestyle='-',alpha=1,label=r"k=0.142,$\xi$=0.21 TUNL")
plt.plot(x0,LindhardQFtoP(x0,[fmu+fsigma,0.21-0.15]),color='r',linestyle='--',alpha=1)
plt.plot(x0,LindhardQFtoP(x0,[fmu-fsigma,0.21+0.12]),color='r',linestyle='--',alpha=1)
plt.plot(x0,LindhardQFtoP(x0,[0.133,0.0000001]),color='#d00d5b',linestyle='-',alpha=1,label=r"k=0.157,$\xi$=0 Standard Lindhard")
plt.plot(x0,LindhardQFtoP(x0,[0.151,0.16]),color='#ffa24f',linestyle='-',alpha=1,label=r"k=0.1789,$\xi$=0.16 B.J.Scholz")
plt.errorbar(RecoilE, QF, xerr=RecoilEerr,yerr=QFerr, marker='D',color='#4a6bf5',linestyle='None', capsize=0,label="TUNL 2020")        
plt.errorbar(JonesE, JonesQF,yerr=JonesQFerr, marker='D',color='k',linestyle='None', capsize=0,label="World data")        
plt.errorbar(BarbeauE, BarbeauQF,yerr=BarubeauQFerr, marker='D',color='k',linestyle='None', capsize=0) 
plt.errorbar(TexonoE, TexonoQF,yerr=TexonoQFerr, marker='D',color='k',linestyle='None', capsize=0) 
plt.errorbar(MessesE, MessesQF,yerr=MessesQFerr, marker='D',color='k',linestyle='None', capsize=0) 
plt.errorbar(ChasmanE, ChasmanQF,yerr=ChasmanQFerr, marker='D',color='k',linestyle='None', capsize=0) 
plt.errorbar(SattlerE, SattlerQF,yerr=SattlerQFerr, marker='D',color='k',linestyle='None', capsize=0) 
plt.errorbar(SimonE, SimonQF,yerr=SimonQFerr, marker='D',color='k',linestyle='None', capsize=0) 
plt.errorbar(ShuttE, ShuttQF,yerr=ShuttQFerr, marker='D',color='k',linestyle='None', capsize=0) 
plt.errorbar(BaudisE, BaudisQF,yerr=BaudisQFerr, marker='D',color='k',linestyle='None', capsize=0) 

plt.legend(loc='lower right')


plt.xlim(0, 100)
plt.xticks(np.linspace(0,100,11))
plt.xlabel("Nuclear recoil energy (keVnr)")
plt.ylabel("Quenching factor (\%)")
#plt.show()
plt.savefig("Ge QF and MCMC fit 0-100keV.pdf")
