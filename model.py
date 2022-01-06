import numpy as np

class Configure(object):
    """
    Configure class enables to configure the parameters and beam response for PARKES, CHIME, UTMOST, ASKAP
    telescopes
    """
    def __init__(self, name, tel_prm):
        '''
        :param name: telescope name (use amongst; parkes, chime, utmost, askap)
        :param tel_prm: telescope parameter file (init file)
        '''
        self.name = name
        self.tel_parm = tel_prm

    def params(self):
        '''
        :return: telescope parameters for given dataset (type: array)
        parameters: array( nuA, nuB, dmMW, nu, BW, nu1, nu2, theta0,
         Nch, Flim, DMmin, Fmin, hDM, hF, nev, nDMbin, nFbin)
         
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
        '''
        init = np.loadtxt(self.tel_parm)
        if self.name == 'parkes':
            return init[0]
        elif self.name == 'chime':
            return init[1]
        elif self.name == 'askap':
            return init[2]
        elif self.name == 'utmost':
            return init[3]
        else:
            raise NameError('use correct name, available instrument names are parkes, chimes, utmost, askap')

    def beamf(self, thy):
        '''
        Beam functions for PARKES, CHIME, ASKAP, UTMOST telescopes
        :param thy: beam angle (type: int, float, array)
        :return: beam response (type: float, array)
        '''
        if self.name == 'parkes': #(exp((-pow(thy,2.0))/(2*pow(0.003467,2))))
            return np.exp((-np.power(thy, 2.0)) / (2. * np.power(0.003467, 2.)))

        elif self.name == 'chime': #(pow(sinc(125.66*thy),2.0)*pow(sinc(1.963*thy),2.0))
            return np.power(np.sinc(125.66*thy),2.0)*np.power(np.sinc(1.963*thy),2.0)

        elif self.name == 'utmost':
            return np.power(np.sinc(102.37*thy),2.0)*np.power(np.sinc(40.33*thy),2.0)

        elif self.name == 'askap': #(exp((-pow(thy,2.0))/(2*pow(0.019275,2))))
            return np.exp((-np.power(thy,2.))/(2.*np.power(0.019275,2.)))
        else:
            raise NameError('use correct name, available instrument names are parkes, chimes, utmost, askap')
