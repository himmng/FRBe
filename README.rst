====
FRBe
====

.. image:: http://ForTheBadge.com/images/badges/made-with-python.svg
   :target: https://www.python.org/

**Fast Radio Burst Estimator: maximum likelihood estimation of FRB population
parameters from observed fluence and dispersion measure.**

FRBe models the population of Fast Radio Bursts and predicts event counts
binned in fluence and dispersion measure, for several radio telescopes with
different frequency coverage and different detection criteria. It compares
those predictions with observed catalogues to estimate the parameters of the
underlying population by maximum likelihood, and supports MCMC sampling for
posterior estimation.

Method
------

Each telescope has its own frequency range, sensitivity and selection function,
so the observed burst counts cannot be compared directly across instruments.
FRBe handles this by simulating the expected counts per telescope under a given
population model, then evaluating the likelihood of the observed data under
that model.

**Population parameters:**

- ``alpha``: spectral index
- ``Ebar`` (``E33``): mean FRB energy, in units of 10^33 J, across
  2128 to 2848 MHz in the FRB rest frame
- ``gamma``: exponent of the Schechter luminosity function

**Model variants.** The framework compares competing physical models rather
than fitting one: pulse scatter broadening increasing moderately with redshift,
increasing sharply, or absent; a comoving event rate density that is constant
with redshift against one following the cosmological star formation rate; and a
fixed against a randomly drawn host-galaxy dispersion measure.

**Estimation.** Parameters are estimated by maximum likelihood, either from the
loss function directly or from the log-likelihood, for a single telescope or
jointly across telescopes. Posterior sampling uses
`emcee <https://emcee.readthedocs.io/en/stable/>`_.

Supported telescopes: Parkes, ASKAP, CHIME, UTMOST.

Requirements
------------

::

    numpy
    scipy
    emcee
    matplotlib

Install
-------

::

    $ git clone https://github.com/himmng/FRBe.git
    $ cd FRBe
    $ python setup.py install --user

Usage
-----

Create directories for the observational and simulated data::

    $ mkdir obs_data sim_data

Predicting event counts for a given telescope::

    from frb import FRB

    # telescope name: chime, utmost, askap or parkes
    chime = FRB(name='chime', path='path_to_init')   # path to telescope parameters
                                                     # see the Config docstring

    muc = chime.mu(alpha, ebar, gamma, *args)

    # or, equivalently
    parameters = alpha, ebar, gamma
    muc = chime.mu(*parameters, *args)

    # *args are the simulation values loaded once:
    # args = [z, r, theta, dmtot, wwa, cdf]

Parameter estimation with MCMC::

    from mcmc import MCMC

    nwalkers = 6
    ndim = 3
    mcmc_filename = 'run.h5'       # must be a .h5 file

    mcmc = MCMC(nwalkers, ndim, mcmc_filename)

    # joint estimation across telescopes
    joint = mcmc.joint_run(method='use_loss')      # maximum likelihood region
    joint = mcmc.joint_run(method='use_loglike')   # log-likelihood directly

    # single telescope
    run = mcmc.run(method='use_loss', name='chime')
    run = mcmc.run(method='use_loglike', name='chime')

Citation
--------

FRBe was developed for the statistical analysis in the following papers:

    S. Bhattacharyya, H. Tiwari, S. Bharadwaj and S. Majumdar,
    *A maximum likelihood estimate of the parameters of the FRB population*,
    Monthly Notices of the Royal Astronomical Society: Letters, 2021.
    DOI: https://doi.org/10.1093/mnrasl/slab107 |
    arXiv: https://arxiv.org/abs/2109.06785

    S. Bhattacharyya, S. Bharadwaj, H. Tiwari and S. Majumdar,
    *Modelling the energy distribution in CHIME/FRB Catalog-1*,
    Monthly Notices of the Royal Astronomical Society, 2023, 522, 3349.
    DOI: https://doi.org/10.1093/mnras/stad458 |
    arXiv: https://arxiv.org/abs/2209.12961

The first paper analyses 82 non-repeating FRBs detected at Parkes, ASKAP, CHIME
and UTMOST, and finds best-fit values alpha = -1.53, Ebar_33 = 1.55 and
gamma = 0.77. The second uses 254 non-repeating low dispersion measure bursts
from CHIME/FRB Catalog-1 to derive the conditional energy distribution,
finding evidence for two distinct FRB populations.

Status
------

Under active development. The MCMC module is being extended.
