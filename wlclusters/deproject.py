import numpy as np
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

class MyDeprojVol:
    def __init__(self, radin, radot):
        self.radin = radin
        self.radot = radot

    def deproj_vol(self):
        ri = np.copy(self.radin)
        ro = np.copy(self.radot)

        diftot = 0
        for i in range(1, len(ri)):
            dif = abs(ri[i] - ro[i-1]) / ro[i-1] * 100.
            diftot += dif
            ro[i-1] = ri[i]

        if abs(diftot) > 0.1:
            print(' DEPROJ_VOL: WARNING - abs(ri(i)-ro(i-1)) differs by', diftot, ' percent')
            print(' DEPROJ_VOL: Fixing up radii ... ')
            for i in range(1, len(ri)-1):
                dif = abs(ri[i] - ro[i-1]) / ro[i-1] * 100.
                diftot += dif

        nbin = len(ro)
        volconst = 4. / 3. * np.pi
        volmat = np.zeros((nbin, nbin))

        for iring in list(reversed(range(0, nbin))):
            volmat[iring, iring] = volconst * ro[iring]**3 * (1. - (ri[iring] / ro[iring])**2.)**1.5
            for ishell in list(reversed(range(iring+1, nbin))):
                f1 = (1. - (ri[iring] / ro[ishell])**2.)**1.5 - (1. - (ro[iring] / ro[ishell])**2.)**1.5
                f2 = (1. - (ri[iring] / ri[ishell])**2.)**1.5 - (1. - (ro[iring] / ri[ishell])**2.)**1.5
                volmat[ishell, iring] = volconst * (f1 * ro[ishell]**3 - f2 * ri[ishell]**3)

                if volmat[ishell, iring] < 0.0:
                    exit()

        return np.copy(volmat)