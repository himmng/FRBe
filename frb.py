from model import Conf
import numpy as np 
from numpy import *
import scipy.special as sc
from matplotlib.pyplot import *
from scipy.interpolate import interp1d

class Frb(object):
    """
    FRB major class that evaluates fluence, Schechter energy (Esch), binned FRB predictions (mu)
    """
    def __init__(self, name, path):
        '''
        :param name: telescope name
        :param path: provide the path to the telescope parameters data;
         please follow the indexing of the telescope
        parameters as mentioned in the Config Class. Also, It is recommended that
        you put all of the dataset into a same directory.
        '''

        instrument = Conf(name, path)  # configuring the telescope (instrument)
        self.init =  instrument.params() # telescope parameters
        self.instrument = instrument

    def fluence(self, z, r, theta, alpha, Eval):
        '''
        Calculate fluence for the given telescope.
        :param z: redshift
        :param theta: telescope beam angle
        :param alpha: spectral index
        :param Eval: Energy
        :return: fluence
        '''

        def phibar(z, alpha):
            return (((power(self.init[6], 1.0 + alpha) - power(self.init[5], 1.0 + alpha)) * power(1.0 + z, alpha))
                / ((power(self.init[1], 1.0 + alpha) - power(self.init[0], 1.0 + alpha)) * self.init[4]))

        return (0.454 * Eval * phibar(z, alpha) * self.instrument.beamf(theta)) / power(r, 2.0)

    def esch(self, gamma, ebar, cdf):
        '''

        :param gamma: FRB gamma exponent parameter
        :param ebar: FRB everage energy
        :param cdf: cumulative distribution parameter
        :return: cumulative distribution of Schechter Function (energy-cdf)
        '''
        xx = np.arange(0.0001, 10000, 0.01) #this is just a x-range to interpolate the energy function 

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

    def mu(self, alpha, Ebar, gama, z, r, theta, dmtot, wwa, cdf): ##for updated results 
        #(only three parameters alpha, Ebar, gama are variable)
        '''

        :param alpha: FRB spectral index parameter
        :param Ebar: FRB average energy
        :param gama: FRB gamma exponent parameter
        :param z: FRB redshift
        :param theta: telescope beam angle
        :param dmtot: Total dispersion measure
        :param wwa: width of FRB profile
        :param cdf: cumulative distribution parameter
        :return: predicted binned FRB events
        '''
        if alpha == -1.0:
            alpha += np.random.uniform(0,0.001)
        Eval = self.esch(gama, Ebar, cdf)
        indx = np.where(Eval != 10000)[0] # where Eval is discarded

        Fval = self.fluence(z, r, theta, alpha, Eval)[indx]

        cond = np.where((Fval / np.sqrt(wwa[indx])) >= self.init[9])[0] # where Fval > Flim

        DMa = dmtot[cond]
        Fval = Fval[cond]

        if len(cond) == 0: #no predictions are found
            return 0.0

        nDMmod = ((log(DMa) - log(self.init[10])) / self.init[12])
        nFmod = ((log(Fval) - log(self.init[11])) / self.init[13])

        x0 = np.where(nDMmod < int(self.init[15]))[0] # just bypassing the loops

        x1 = np.where(nDMmod[x0] >= 0.0)[0]

        y0 = np.where(nFmod[x0[x1]] < int(self.init[16]))[0]

        y1 = np.where(nFmod[x0[x1[y0]]] >= 0.0)[0]

        nDMmod = nDMmod[x0[x1[y0[y1]]]]
        nFmod =  nFmod[x0[x1[y0[y1]]]]

        mup = np.array(hist2d(x=nDMmod,y=nFmod, bins=(int(self.init[15]), int(self.init[16])))[0])
        mup = (mup * self.init[14])/ len(cond)

        return mup

    def mua(self, alpha, Ebar, gama, z, r, theta, dmtot, wwa, cdf): ##(for old calculations)
            '''

            :param alpha: FRB spectral index parameter
            :param Ebar: FRB average energy
            :param gama: FRB gamma exponent parameter
            :param z: FRB redshift
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
            #indxx = np.where(Eval == 10000)[0]
            Fval = self.fluence(z, r, theta, alpha, Eval)[indx]

            cond = np.where((Fval / sqrt(wwa[indx])) >= self.init[9])[0]

            DMa = dmtot[cond]
            Fval = Fval[cond]

            nDMmod = ((np.log(DMa) - np.log(self.init[10])) / self.init[12])
            nFmod = ((np.log(Fval) - np.log(self.init[11])) / self.init[13])
            # justing bypassing loops
            x0 = np.where(nDMmod < int(self.init[15]))[0]

            x1 = np.where(nDMmod[x0] >= 0.0)[0]

            y0 = np.where(nFmod[x0[x1]] < int(self.init[16]))[0]

            y1 = np.where(nFmod[x0[x1[y0]]] >= 0.0)[0]
            #### till here

            nDMmod = np.int64(nDMmod[x0[x1[y0[y1]]]])
            nFmod = np.int64(nFmod[x0[x1[y0[y1]]]])

            count = np.zeros(shape=(int(self.init[15]), int(self.init[16])))
            if len(cond) == 0:  # no predictions
                return count

            for ii in range(len(nDMmod)):
                count[nDMmod[ii]][nFmod[ii]] += 1

            mua = count * self.init[14] / len(cond)

            return mua

    def loss(self, N, mu):
        '''

        :param N: observed FRB events binned over fluence and Dispersion measure (type: array)
        :param mua: simulation predicted FRB events binned over fluence and Dispersion measure (type: array)
        :return: count loss between observation and prediction
        '''

        return -(np.sum(np.dot((mu - N).T, (mu - N)))) #loss for likelihood 


    def loglike(self, N, mu):
        '''

        param N: observed FRB events binned over fluence and Dispersion measure (type: array)
        :param mua: simulation predicted FRB events binned over fluence and Dispersion measure (type: array)
        :return: binned summation of log likelihood
        '''
        ll = 0
        lm = np.ma.masked_equal(N, 0)
        lm = np.sum(lm*np.log(lm) - lm)
        for i in range(int(self.init[15])):
            for j in range(int(self.init[16])):
                if (mu[i][j] != 0): # other conditions already 
                    #consumed inside it, except where (mu[i,j] = 0); it is ruled-out.
                    ll += N[i, j] * np.log(mu[i, j]) - mu[i, j]
                    
        return ll/lm

    def data_bin(self, DMobs, Fobs):
        '''

            :param DMobs: Observed Dispersion measure
            :param Fobs: Observed Fluence measure
            :return: binned array
        '''
        Na = np.zeros(shape=(int(self.init[15]), int(self.init[16])))
        for i in range(0, int(self.init[14])):
            nDMobs = ((np.log(DMobs[i]) - np.log(self.init[10])) / self.init[12])
            nFobs = ((np.log(Fobs[i]) - np.log(self.init[11])) / self.init[13])
            if (nDMobs >= 0 and nFobs >= 0 and nDMobs < int(self.init[15]) and nFobs < int(self.init[16])):
                Na[int(nDMobs)][int(nFobs)] += 1

        return Na

    def hex_bin(self, DMobs, Fobs):
        '''

        :param DMobs: Observed Dispersion measure
        :param Fobs: Observed Fluence measure
        :return: binned array
        '''
        nDMobs = ((np.log(DMobs) - np.log(self.init[10])) / self.init[12])
        nFobs = ((np.log(Fobs) - np.log(self.init[11])) / self.init[13])
        Na = np.array(hist2d(x=nDMobs, y=nFobs, bins=(int(self.init[15]), int(self.init[16])))[0])
        return Na

