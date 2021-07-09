import numpy as np

class Conf(object):
    """
    Conf class enables to configure the parameters and beam response for PARKES, CHIME, UTMOST, ASKAP
    telescopes
    """
    def __init__(self, name, path):
        '''
        :param name: telescope name (use amongst; parkes, chime, utmost, askap)
        '''
        self.name = name
        self.path = path

    def params(self):
        '''
        :return: telescope parameters for given dataset (type: array)
        parameters: array( nuA, nuB, dmMW, nu, BW, nu1, nu2, theta0,
         Nch, Flim, DMmin, Fmin, hDM, hF, nev, nDMbin, nFbin)
        '''
        init = np.loadtxt(self.path + 'init')
        if self.name == 'parkes':
            return init[0]
        elif self.name == 'chime':
            return init[1]
        elif self.name == 'utmost':
            return init[2]
        elif self.name == 'askap':
            return init[3]
        else:
            raise NameError('use correct name, available instrument names are parkes, chimes, utmost, askap')

    def beamf(self, thy):
        '''
        :param thy: beam angle (type: int, float, array)
        :return: beam response (type: float, array)
        '''
        if self.name == 'parkes':
            return np.exp((-np.power(thy, 2.0)) / (2. * np.power(0.003467, 2.)))
        elif self.name == 'chime':
            return np.power(np.sinc(125.66*thy),2.0)*np.power(np.sinc(1.963*thy),2.0)
        elif self.name == 'utmost':
            return np.power(np.sinc(102.37*thy),2.0)*np.power(np.sinc(40.33*thy),2.0)
        elif self.name == 'askap':
            return np.exp((-np.power(thy,2.0))/(2.*np.power(0.019275,2.)))
        else:
            raise NameError('use correct name, available instrument names are parkes, chimes, utmost, askap')