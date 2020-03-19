import numpy as np

# TODO: extend from sklearn
class IREOS():
    def __init__(self, data, solutions):
        self.data = data
        self.solutions = solutions

    def set_gamma_max(self, gamma_max):
        assert gamma_max > 0, "The maximum value of gamma (gamma_max) must be higher than 0"

        self.gamma_max = gamma_max

    def set_ngamma(self, n):
        assert n > 0; "Number of values to discretize gamma (n) must be higher than 0"

        self.n_gamma = n

    def set_gammas(self, gammas=None, scale=None):
        assert (gammas is not None) or (scale is not None); "Either the values (gammas) or the scale (scale) must be provided"

        if scale is not None:
            scales = {
                0: 'linear',
                1: 'quadratic',
                2: 'log',
                3: '_log'
            }
            assert (scale in scales.keys()) or (scale in scales.values()); "Scale must be 0 ('linear'), 1 ('quadratic'), 2 (logarithmic='log') or 3 (legacy log='_log')"

            if type(scale) is float:
                scale = int(scale)
            
            if type(scale) is int:
                scale = scales[scale]
            
            start = 0
            stop = self.gamma_max
            steps = self.n_gamma
            if scale == 'linear':
                self.gammas = np.linspace(0, self.gamma_max, self.n_gamma)
            elif scale == 'quadratic':
                quad_start = np.power(start, 2)
                quad_stop = np.power(stop, 2)
                self.gammas = np.sqrt(np.linspace(quad_start, quad_stop,steps))
            elif scale == 'log':
                assert start != 0, "this scale is not suitable for this range (start=0), please use the legacy log scale ('_log')"

                log_start = np.log10(start)
                log_stop = np.log10(stop)
                self.gammas = np.logspace(log_start, log_stop, steps)
            elif scale == '_log':
                lin_values = np.linspace(start, steps, steps+1)
                offset = 1 - start
                self.gammas = np.power(stop + offset, lin_values / steps) - offset

    def set_mCl(self, mCl):
        assert mCl > 0, "The maximum clump size (mCl) must be higher than 0"
        assert mCl <= self.get_noutliers(), "The maximum clump size (mCl) should not be higher than the number of outliers"

        self.mCl = mCl

    def get_noutliers(self):
        solution = self.solutions[0]
        return sum(solution > 0)
