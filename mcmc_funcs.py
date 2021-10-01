from frb import Frb
from multiprocessing import Pool
import os
import numpy as np
import emcee

os.environ["OMP_NUM_THREADS"] = "1"
path = './'

class MCMC(object):

    def __init__(self, initial, nsteps, mcmc_filename):

        self.initial = initial
        self.nsteps = nsteps
        self.nwalkers, self.ndim = initial.shape()
        self.filename = mcmc_filename

    def loss_like(self, name, method):
        frb = Frb(name, path)
        if method == 'use_loss':
            loss_like = frb.loss(N, mu)
        else:
            loss_like = frb.loglike(N, mu)
        return loss_like


    def prob(self, theta, name):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        
        if name == 'chime':
            chime = Frb(name, path=path)
            muc = chime.mu(*theta, *argc)
            return lp + self.loss_like(Nac, muc)
        elif name == 'parkes':
            return lp + self.loss_parkes(theta, )
        elif name == 'askap':
            return lp + self.loss_askap(theta, )
        elif name == 'utmost':
            return lp + self.loss_utmost(theta, )
        else:
            raise NameError('use correct name, available instrument names are parkes, chimes, utmost, askap')

    def joint_prob(self, theta, ):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.loss_chime(theta, ) \
               + self.loss_parkes(theta, ) \
               + self.loss_askap(theta, ) \
               + self.loss_utmost(theta, )

    def run(self, method, name):
        backend = emcee.backends.HDFBackend(self.filename)
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.prob, pool=pool,
                                            backend=backend, args=[name, method])
            sampler.run_mcmc(self.initial, self.nsteps, progress=True)


    def joint_run(self, method):
        backend = emcee.backends.HDFBackend(self.filename)
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.joint_prob,
                                            pool=pool, backend=backend, args=[method])
            sampler.run_mcmc(self.initial, self.nsteps, progress=True)



