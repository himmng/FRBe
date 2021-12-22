import numpy as np

def init_maker(parkes, chime, askap, utmost):
    """
    Makes the telescope parameter file
    
        :return: telescope parameters for given dataset (type: array)
        parameters: array( nuA, nuB, BW, nu1, nu2,
         Flim, DMmin, Fmin, hDM, hF, nev, nDMbin, nFbin)
         
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
        
    """
    init0 = init1 = init2 = init3 = np.zeros(13)
    init0 = parkes
    init1 = chime
    init2 = askap
    init3 = utmost
    initt = np.array([init0, init1, init2, init3])
    initt = initt.reshape(4,13)
    np.savetxt('init_telescope', initt)
    print('init_telescope saved!')
    

