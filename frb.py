from model import Configure
import numpy as np 
import scipy.special as sc
from matplotlib.pyplot import *
from scipy.interpolate import interp1d

class Frb(object):
    """
    FRB major class that evaluates fluence, Schechter energy (E_sch), binned FRB predictions (mu_pred)
    """
    def __init__(self, name, path):
        '''
        :param name: telescope name, i.e. parkes, chime, askap, utmost
        :param path: provide the path to the telescope parameters file a.k.a init_telescope
        please follow the indexing for the telescope in the init_telescope file, [0-parkes, 1-chime, 2-askap, 3-utmost]

        parameters are mentioned in the Configure Class in model.py file,
        parameters are;
        nuA ==>Normalization Frequency Limit Lower : MHz 
        nuB ==>Normalization Frequency Limit Upper : MHz
        BW ==>Bandwidth : MHz
        nu1 ==>Lower Frequency limit : MHz
        nu2 ==>Upper Frequency limit : MHz
        Flim ==>Limiting Fluence
        DMmin ==> min DM
        Fmin ==> min Fluence
        hDM ==>divisor Despersion measure
        hF ==>divisor Fluence measure 
        nev ==>Number of FRB Events
        nDMbin ==> bins along DM axis
        nFbin ==> bins along fluence axis
        Also, It is recommended that you put all of the dataset into a same directory.
        '''

        instrument = Configure(name, path)  # configuring the telescope (instrument)
        self.tel_parm =  instrument.params() # telescope parameters
        self.instrument = instrument

    def phibar(self, z, alpha): 
            
        return (((np.power(self.tel_parm[4], 1.0 + alpha) - np.power(self.tel_parm[3], 1.0 + alpha)) 
            * np.power(1.0 + z, alpha))
                / ((np.power(self.tel_parm[1], 1.0 + alpha) - np.power(self.tel_parm[0], 1.0 + alpha))
                 * (self.tel_parm[2])))

    def fluence(self, z, r, theta, alpha, Eval):
        '''
        Calculate fluence for the given telescope.
        :param z: redshift
        :param theta: telescope beam angle
        :param alpha: spectral index
        :param Eval: Schechter Energy

        :return: fluence
        '''
        return ((453.96 * Eval * self.phibar(z, alpha) * self.instrument.beamf(theta))
         / np.power(r, 2.0))

    def esch(self, gamma, ebar, cdf):
        '''
        Schecter Energy fuction
        :param gamma: FRB gamma exponent parameter
        :param ebar: FRB everage energy
        :param cdf: cumulative distribution parameter
        :return: cumulative distribution of Schechter Function (energy-cdf)
        '''
        xx = np.arange(0.001, 100, 0.01) #this is just a x-range to interpolate the energy function 

        def func0(En):
            return (1.0 - np.exp(-En / ebar))

        def func(En):
            return sc.gammainc(1.0 + gamma, En * gamma / ebar)

        if gamma == 0.0:
            m = interp1d(x=func0(En=xx), y=xx, bounds_error=False, fill_value=10000)
            m = m(cdf)
            return m
        else:
            m = interp1d(x=func(En=xx), y=xx, bounds_error=False, fill_value=10000)
            m = m(cdf)
            return m

    def mu(self, alpha, ebar, gama, z, r, theta, dmtot, wwa, cdf): ##for updated results 
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
            alpha += np.random.uniform(0,0.1)
        Eval = self.esch(gama, ebar, cdf)
        indx = np.where(Eval != 10000)[0] # where Eval is discarded

        Fval = self.fluence(z, r, theta, alpha, Eval)[indx]

        cond = np.where((Fval / np.sqrt(wwa)) >= self.tel_parm[5])[0] # where Fval/sc_w > Flim

        DMa = dmtot[cond]
        Fval = Fval[cond]

        nDMmod = ((np.log10(DMa) - np.log10(self.tel_parm[6])) / self.tel_parm[8])
        nFmod = ((np.log10(Fval) - np.log10(self.tel_parm[7])) / self.tel_parm[9])


        a = set(np.where(nDMmod<self.tel_parm[11])[0])
        b = set(np.where(nFmod<self.tel_parm[12])[0]) 
        c = set(np.where(nDMmod>=0)[0])
        d = set(np.where(nFmod>=0)[0])
        e = np.array(list(a & b & c & d))

        mup = np.array(hist2d(x=nDMmod[e],y=nFmod[e], bins=(int(self.tel_parm[11]), int(self.tel_parm[12])))[0])
        mup = (mup * self.tel_parm[10])/ len(cond)

        return mup


    def log_post(self, N, mu, a=1, b=0.1):
        log_prob= -np.log((len(N.flatten())+1/b)**(len(N.flatten())*np.mean(N.flatten())+a-2)) \
        + (-len(N.flatten())+ 1/b) * mu + \
        + (len(N.flatten())*np.mean(N.flatten()) + a - 1)* np.log(mu) \
        - np.log(sc.gamma(len(N.flatten())*np.mean(N.flatten())+a))
        
        return np.sum(log_prob)

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
        Na = np.ma.masked_equal(N, 0)
        #lm = np.sum(Na*np.log(Na) - Na)
        for i in range(int(self.tel_parm[11])):
            for j in range(int(self.tel_parm[12])):
                if (mu[i][j] != 0): # other conditions already 
                    #consumed inside it, except where (mu[i,j] = 0); it is ruled-out.
                    ll += N[i, j] * np.log(mu[i, j]) - mu[i, j]
   
        return ll

    def like(self, N, mu):
        '''
        param N: observed FRB events binned over fluence and Dispersion measure (type: array)
        :param mua: simulation predicted FRB events binned over fluence and Dispersion measure (type: array)
        :return: binned summation of log likelihood
        '''
        N  = N.flatten('F')
        mu = mu.flatten('F')
        
        prob = (mu**N)*(np.exp(-mu))/sc.factorial(N)
        prob_max = (N**N)*(np.exp(-N))/sc.factorial(N) #when mu=N max. likelihood
        like = prob-prob_max
        like = np.sum(like)
        return like


    def data_bin(self, DMobs, Fobs):
        '''

            :param DMobs: Observed Dispersion measure
            :param Fobs: Observed Fluence measure
            :return: binned array
        '''
        #Na = np.zeros(shape=(int(self.tel_parm[11]), int(self.tel_parm[12])))

        nDMobs = ((np.log10(DMobs) - np.log10(self.tel_parm[6])) / self.tel_parm[8])
        nFobs = ((np.log10(Fobs) - np.log10(self.tel_parm[7])) / self.tel_parm[9])

        a = set(np.where(nDMobs<self.tel_parm[11])[0])
        b = set(np.where(nFobs<self.tel_parm[12])[0]) 
        c = set(np.where(nDMobs>=0)[0])
        d = set(np.where(nFobs>=0)[0])
        e = np.array(list(a & b & c & d))
        Na = np.array(hist2d(x=nDMobs[e], y=nFobs[e], bins=(int(self.tel_parm[11]), int(self.tel_parm[12])))[0])
        return Na

    def hex_bin(self, DMobs, Fobs):
        '''

        :param DMobs: Observed Dispersion measure
        :param Fobs: Observed Fluence measure
        :return: binned array
        '''
        nDMobs = ((np.log10(DMobs) - np.log10(self.tel_parm[6])) / self.tel_parm[8])
        nFobs = ((np.log10(Fobs) - np.log10(self.tel_parm[7])) / self.tel_parm[9])
        Na = np.array(hist2d(x=nDMobs, y=nFobs, bins=(int(self.tel_parm[11]), int(self.tel_parm[12])))[0])
        return Na

