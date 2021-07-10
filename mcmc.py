from frb import Frb
from multiprocessing import Pool
import os
import numpy as np
import time
import emcee

os.environ["OMP_NUM_THREADS"] = "1"
path = './'

nsteps = 200
nwalkers = 6
alpha = np.random.uniform(-20., 20., nwalkers)
ebar = np.random.uniform(0.01, 10., nwalkers)
gama = np.random.uniform(0., 6., nwalkers)
initial = np.array([alpha, ebar, gama])
initial = initial.T
ndim = 3

# defining the data for different models (No-Sc, Sc-I, Sc-II), event rates, DM cases

sc_mo = ['nosc', 'scone', 'sctwo']  # scattering models
sm = [5, 6, 7]  # scattering model index in the data-set (5==> nosc, 6==> scone, 7==> sctwo)
ra_mo = ['cer', 'sfr']  # rate model
dm_mo = ['dm50', 'dmrand']  # DM cases

# using the specific case
sc_mo = sc_mo[2]
sm = sm[2]
ra_mo = ra_mo[0]
dm_mo = dm_mo[0]

mcmc_filename = 'joint_{0:s}_{1:s}{2:s}.h5'.format(sc_mo, ra_mo, dm_mo)  # MCMC filename

# checking for the case
print(mcmc_filename)

# simulation data load#
argc = np.loadtxt(path + 'sim_data/chime/{0:s}_{1:s}.dat'.format(ra_mo, dm_mo), usecols=[0, 1, 2, 4, sm, 8],
                  unpack=True)
argp = np.loadtxt(path + 'sim_data/parkes/{0:s}_{1:s}.dat'.format(ra_mo, dm_mo), usecols=[0, 1, 2, 4, sm, 8],
                  unpack=True)
arga = np.loadtxt(path + 'sim_data/askap/{0:s}_{1:s}.dat'.format(ra_mo, dm_mo), usecols=[0, 1, 2, 4, sm, 8],
                  unpack=True)
argu = np.loadtxt(path + 'sim_data/utmost/{0:s}_{1:s}.dat'.format(ra_mo, dm_mo), usecols=[0, 1, 2, 4, sm, 8],
                  unpack=True)

# observational data load#
Nac = np.loadtxt(path + 'obs_data/chime/Nac')
Nap = np.loadtxt(path + 'obs_data/parkes/Nap')
Nas = np.loadtxt(path + 'obs_data/askap/Nas')
Nau = np.loadtxt(path + 'obs_data/utmost/Nau')


def loss_chime(theta, ):
    chime = Frb(name='chime', path=path)
    muc = chime.mu(*theta, *argc)
    return -(np.sum(np.dot((muc - Nac).T, (muc - Nac))))


def loss_parkes(theta, ):
    parkes = Frb(name='parkes', path=path)
    mup = parkes.mu(*theta, *argp)
    return -(np.sum(np.dot((mup - Nap).T, (mup - Nap))))


def loss_utmost(theta, ):
    utmost = Frb(name='utmost', path=path)
    muu = utmost.mu(*theta, *argu)
    return -(np.sum(np.dot((muu - Nau).T, (muu - Nau))))


def loss_askap(theta, ):
    askap = Frb(name='askap', path=path)
    mua = askap.mu(*theta, *arga)
    return -(np.sum(np.dot((mua - Nas).T, (mua - Nas))))


def joint_prob(theta, ):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loss_chime(theta, ) \
           + loss_parkes(theta, ) \
           + loss_askap(theta, ) \
           + loss_utmost(theta, )


def log_prior(theta):
    alpha, Ebar, gama = theta
    if -20. <= alpha <= 20. and 0.01 <= Ebar <= 10.0 and 0.0 <= gama <= 6.:
        return 0.0
    return -np.inf


def prob(theta, name):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    if name == 'chime':
        return lp + loss_chime(theta, )
    elif name == 'parkes':
        return lp + loss_parkes(theta, )
    elif name == 'askap':
        return lp + loss_askap(theta, )
    elif name == 'utmost':
        return lp + loss_utmost(theta, )
    else:
        raise NameError('use correct name, available instrument names are parkes, chimes, utmost, askap')


# MCMC functions
backend = emcee.backends.HDFBackend(mcmc_filename)


def run():
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, prob, pool=pool, backend=backend, args=[name])
        start = time.time()
        sampler.run_mcmc(initial, nsteps, progress=True)
        end = time.time()
    print('time taken is %s mins.' % np.around(np.float(end - start) / 60.))


def joint_run():
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, joint_prob, pool=pool, backend=backend)
        start = time.time()
        sampler.run_mcmc(initial, nsteps, progress=True)
        end = time.time()
    print('time taken is %s mins.' % np.around(np.float(end - start) / 60.))


name = 'chim'
run()
