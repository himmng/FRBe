====
FRBe
====

.. image:: http://ForTheBadge.com/images/badges/made-with-python.svg
   :target: https://www.python.org/


FRBe (Fast Radio Bursts Estimator) evaluate the FRB populations and event counts over binned fluence and Dispersion Measure.
The project is under development.

Requirements
------------

::


    numpy
    scipy
    emcee
    matplotlib


::

Install
-------

::

    $ git clone https://github.com/himmng/FRBe.git
    $ cd FRBe
    $ python setup.py install

::

How to use
----------
make two different directories for observational, simulation data to put the data there
(if doesn't exist)


::

    $ mkdir obs_data sim_data

::


Using the FRB class in python:

::

    from frb import FRB

    # using for specific telescope; use e.g. chime, utmost, askap, parkes

    chime = FRB(name = 'chime', path = 'path_to_init') # provide name of telescope, path ot init (telescope paramters).
    # see docstring of Config class for help.

    # To do the prediction
    muc = chime.mu(alpha, ebar, gamma, *args)

    # OR use

    parameters = alpha, ebar, gamma
    muc = chime.mu(*parameters, *args)
    # *args; positional arguments are the simulation values which are loaded at once
    # args = [z, r, theta, dmtot, wwa, cdf]


::

Using in MCMC (coming soon)

::

    from mcmc import MCMC

    # create instances
    nwalkers = 6
    ndim = 3
    mcmc_filename = 'run.h5'
    # for specific case of (No-Sc, Sc-I, Sc-II)(cer, sfr,) and (DM50, DMrand)
    # must be a .h5 file

    mcmc = MCMC(nwalkers, ndim, filename)

    # load your dataset...

    # Uses for joint estimation

    joint = mcmc.joint_run(method = 'use_loss')
    # It will use loss function to find out the maximum likelihood region

    #OR

    joint = mcmc.joint_run(method = 'use_loglike')
    # It will use log likelihood function itself

    # Uses for specific telescope cases
    # using specific telescopes; ; use e.g. chime, utmost, askap, parkes as name
    run = mcmc.run(method = 'use_loss', name='chime')

    # OR

    run = mcmc.run(method = 'use_loglike', name='chime')






