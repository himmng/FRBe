from model import Conf
import numpy as np
import scipy.special as sc
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class Frb(object):
    """
    FRB major class that evaluates fluence, Schechter energy (Esch), binned FRB predictions (mu)
    """
    def __init__(self, name, path):
        '''
        :param name: telescope name
        :param path: provide the path to the telescope parameters data; please follow the indexing of the telescope
        parameters as mentioned in the Config Class. Also, It is recommended that
        you put all of the dataset into a same directory.
        '''

        instr = Conf(name, path)  # configuring the telescope (instrument)
        self.init =  instr.params() # telescope parameters
        self.instr = instr

    def fluence(self, z, r, theta, alpha, Eval):
        '''
        Calculate fluence for the given telescope.
        :param z: redshift
        :param r: radius
        :param theta: telescope beam angle
        :param alpha: spectral index
        :param Eval: Energy
        :return: fluence
        '''

        def phibar(z, alpha):
            return (((np.power(self.init[6], 1.0 + alpha) - np.power(self.init[5], 1.0 + alpha)) * np.power(1.0 + z, alpha))
                / ((np.power(self.init[1], 1.0 + alpha) - np.power(self.init[0], 1.0 + alpha)) * self.init[4]))

        return (0.454 * Eval * phibar(z, alpha) * self.instr.beamf(theta)) / np.power(r, 2.0)

    def esch(self, gamma, ebar, cdf):
        '''

        :param gamma: FRB gamma exponent parameter
        :param ebar: FRB everage energy
        :param cdf: cumulative distribution parameter
        :return: cumulative distribution of Schechter Function (energy-cdf)
        '''
        xx = np.arange(0.0001, 10000, 0.01)

        def func0(En):
            return (1.0 - np.exp(-En / ebar))

        def func(En):
            return (1.0 - (sc.gammainc(1.0 + gamma, En * (1.0 + gamma) / ebar) / sc.gamma(1.0 + gamma)))

        if gamma == 0.0:
            m = interp1d(x=func0(En=xx), y=xx, bounds_error=False, fill_value=10000)
            m = m(cdf)
            return m
        else:
            m = interp1d(x=func(En=xx), y=xx, bounds_error=False, fill_value=10000)
            m = m(cdf)
            return m

    def mu(self, alpha, Ebar, gama, z, r, theta, dmtot, wwa, cdf):
        '''

        :param alpha: FRB spectral index parameter
        :param Ebar: FRB average energy
        :param gama: FRB gamma exponent parameter
        :param z: FRB redshift
        :param r: radius
        :param theta: telescope beam angle
        :param dmtot: Total dispersion measure
        :param wwa: width of FRB profile
        :param cdf: cumulative distribution parameter
        :return: predicted binned FRB events
        '''
        if alpha == -1.0:
            alpha += np.random.uniform(0,0.001)
        Eval = self.esch(gama, Ebar, cdf)
        indx = np.where(Eval != 10000)[0]

        Fval = self.fluence(z, r, theta, alpha, Eval)[indx]

        cond = np.where((Fval / np.sqrt(wwa[indx])) >= self.init[9])[0]

        DMa = dmtot[cond]
        Fval = Fval[cond]

        if len(cond) == 0:
            return -np.inf

        nDMmod = ((np.log(DMa) - np.log(self.init[10])) / self.init[12])
        nFmod = ((np.log(Fval) - np.log(self.init[11])) / self.init[13])

        x0 = np.where(nDMmod < int(self.init[15]))[0]

        x1 = np.where(nDMmod[x0] >= 0.0)[0]

        y0 = np.where(nFmod[x0[x1]] < int(self.init[16]))[0]

        y1 = np.where(nFmod[x0[x1[y0]]] >= 0.0)[0]

        nDMmod = nDMmod[x0[x1[y0[y1]]]]
        nFmod =  nFmod[x0[x1[y0[y1]]]]

        mua = plt.hist2d(x=nDMmod,y=nFmod, bins=(int(self.init[15]), int(self.init[16])))[0]
        mua = (mua * self.init[14])/ len(cond)

        return mua

    def loss(self, N, mu):
        '''

        :param N: observed FRB events binned over fluence and Dispersion measure (type: array)
        :param mua: simulation predicted FRB events binned over fluence and Dispersion measure (type: array)
        :return: count loss between observation and prediction
        '''

        return -(np.sum(np.dot((mu - N).T, (mu - N))))


    def loglike(self, N, mu):
        '''

        param N: observed FRB events binned over fluence and Dispersion measure (type: array)
        :param mua: simulation predicted FRB events binned over fluence and Dispersion measure (type: array)
        :return: binned summation of log likelihood
        '''
        ll = 0
        for i in range(int(self.init[15])):
            for j in range(int(self.init[16])):
                if (mu[i][j] != 0):
                    ll += N[i, j] * np.log(mu[i, j]) - mu[i, j]

        return ll