class LSSVMClassifier(BaseEstimator, RegressorMixin):

    def __init__(self, gamma: float = 1.0, kernel: str = None,
                 c: float = 1.0, d: float = 2, sigma: float = 1.0):

        self.gamma = gamma
        self.c = c
        self.d = d
        self.sigma = sigma

        if kernel is None:
            self.kernel = 'rbf'
        else:
            self.kernel = kernel

        params = dict()

        if self.kernel == 'poly':
            params['c'] = self.c
            params['d'] = self.d

        elif self.kernel == 'rbf':
            params['sigma'] = self.sigma

        self.kernel_ = LSSVMCLassifier.__set_kernel(self.kernel, **params)
        
# model storage?
        self.x = None
        self.y = None
        self.coef_ = None
        self.intercept_ = None


    def get_params(self, deep=True):
        return {
            "c": self.c,
            "d": self.d,
            "gamma": self.gamma,
            "kernel": self.kernel,
            "sigma": self.sigma}

    def set_params(self, **parameters):

        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        params = dict()

        if self.kernel == 'poly':
            params['c'] = self.c
            params['d'] = self.d

        elif self.kernel == 'rbf':
            params['sigma'] = self.sigma

        self.kernel_ = LSSVMCLassifier.__set_kernel(self.kernel, **params)

        return self

# KERNELS @staticmethod
    def __set_kernel(name: str, **params):

        def linear(xi, xj):
            return np.dot(xi, xj.T)

        def poly(xi, xj, c=params.get('c', 1.0), d=params.get('d', 2)):
            return ((np.dot(xi, xj.T) / c) + 1) ** d

        def rbf(xi, xj, sigma=params.get('sigma', 1.0)):
            from scipy.spatial.distance import cdist

            if xi.ndim == 2 and xj.ndim == 2:
                return np.exp(-(cdist(xi, xj, 'sqeuclidean')) / (2 * sigma**2))

            else:
                return np.exp(
                    -(np.dot(xi, xi) + (xj**2).sum(axis=1)
                      - 2 * np.dot(xi, xj.T))
                    / (2 * sigma**2)
                )

        kernels = {
            'linear': linear,
            'poly': poly,
            'rbf': rbf
        }

        return kernels[name]