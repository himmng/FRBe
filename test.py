from frb import Frb
import numpy as np
d, f = np.loadtxt('obs_data/askap/obs_askap.dat', unpack=True)

hex = Frb(name='utmost', path='./').hex_bin(DMobs=d, Fobs=f)

#np.savetxt('Nas_hex', hex)
print(hex, np.sum(hex))