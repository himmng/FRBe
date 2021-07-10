from frb import Frb
from multiprocessing import Pool
import os
import numpy as np
import time
import emcee
os.environ["OMP_NUM_THREADS"] = "1"
path = './'

nsteps = 2000
nwalkers = 100
alpha = np.random.uniform(-20., 20., nwalkers)
ebar = np.random.uniform(0.01, 10., nwalkers)
gama = np.random.uniform(0., 6., nwalkers)
initial = np.array([alpha, ebar, gama])
initial = initial.T
ndim = initial[0].shape

# defining the data for different models (No-Sc, Sc-I, Sc-II), event rates, DM cases

sc_mo = ['nosc', 'scone', 'sctwo'] # scattering models
sm = [5,6,7]  # scattering model index in the data-set (5==> nosc, 6==> scone, 7==> sctwo)
ra_mo = ['cer', 'sfr'] # rate model
dm_mo = ['dm50', 'dmrand'] # DM cases

#using the specific case
sc_mo = sc_mo[2]
sm = sm[2]
ra_mo = ra_mo[0]
dm_mo = dm_mo[0]

mcmc_filename = 'joint{0:s}_{1:s}{2:s}.h5'.format(sc_mo, ra_mo, dm_mo) # MCMC filename

#checking for the case
print(mcmc_filename)


#simulation data load#
argc = np.loadtxt(path + 'sim_data/chime/{0:s}{1:s}.dat'.format(ra_mo, dm_mo), usecols=[0, 1, 2, 4, sm, 8], unpack=True)
argp = np.loadtxt(path + 'sim_data/parkes/{0:s}{1:s}.dat'.format(ra_mo, dm_mo), usecols=[0, 1, 2, 4, sm, 8], unpack=True)
arga = np.loadtxt(path + 'sim_data/askap/{0:s}{1:s}.dat'.format(ra_mo, dm_mo), usecols=[0, 1, 2, 4, sm, 8], unpack=True)
argu = np.loadtxt(path + 'sim_data/utmost/{0:s}{1:s}.dat'.format(ra_mo, dm_mo), usecols=[0, 1, 2, 4, sm, 8], unpack=True)

#observational data load#
Nac = np.loadtxt(path + 'obs_data/chime/Nac')
Nap = np.loadtxt(path + 'obs_data/parkes/Nap')
Nas = np.loadtxt(path + 'obs_data/askap/Nas')
Nau = np.loadtxt(path + 'obs_data/utmost/Nau')

def log_prior(theta):
    alpha, Ebar, gama = theta
    if -20. <= alpha <= 20. and 0.01 <= Ebar <= 10.0 and 0.0 <= gama <= 6.:
        return 0.0
    return -np.inf

def like_chime(theta,):
    muc = Frb(name='chime', path=path)
    muc = muc.mu(*theta, *argc)
    return -(np.sum(np.dot((muc - Nac).T, (muc - Nac))))

def like_parkes(theta,):
    mup = Frb(name='parkes', path=path)
    mup = mup.mu(*theta, *argp)
    return -(np.sum(np.dot((mup - Nap).T, (mup - Nap))))

def like_utmost(theta,):
    muu = Frb(name='utmost', path=path)
    muu = muu.mu(*theta, *argu)
    return -(np.sum(np.dot((muu - Nau).T, (muu - Nau))))

def like_askap(theta,):
    mua = Frb(name='askap', path=path)
    mua = mua.mu(*theta, *arga)
    return -(np.sum(np.dot((mua - Nas).T, (mua - Nas))))

def joint_prob(theta,):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + like_chime(*theta, *argc) \
           + like_parkes(*theta, *argp) \
           + like_askap(*theta, *arga) \
           + like_utmost(*theta, *argu)

def prob(theta, name):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    if name=='chime':
        return lp + like_chime(*theta, *argc)
    elif name == 'parkes':
        return lp + like_parkes(*theta, *argp)
    elif name == 'askap':
        return lp + like_askap(*theta, *arga)
    elif name == 'utmost':
        return lp + like_utmost(*theta, *argu)
    else:
        raise NameError('use correct name, available instrument names are parkes, chimes, utmost, askap')


# MCMC functions
backend = emcee.backends.HDFBackend(mcmc_filename)
def joint_run():
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, joint_prob, pool=pool, backend=backend)
        start = time.time()
        sampler.run_mcmc(initial, nsteps, progress=True)
        end = time.time()
    print('time taken is %s mins.' % np.around(np.float(end - start) / 60.))

def run():
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, prob, pool=pool, backend=backend)
        start = time.time()
        sampler.run_mcmc(initial, nsteps, progress=True)
        end = time.time()
    print('time taken is %s mins.' % np.around(np.float(end - start) / 60.))

