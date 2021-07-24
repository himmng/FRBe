from frb import Frb
import numpy as np
d, f = np.loadtxt('/home/ht/Desktop/chime/obs_askap.dat', unpack=True)

hex = Frb(name='askap', path='./').hex_bin(DMobs=d, Fobs=f)

np.savetxt('Nas_hex', hex)
print(hex, np.sum(hex))