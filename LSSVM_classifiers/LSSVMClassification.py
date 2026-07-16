#!/usr/bin/env python
# coding: utf-8

# In[ ]:
# theory classification_                         _   _   _    _  _
#| 0          y^T            |  | b  |   | 0 |
#|                          |  |    | = |   |
#| y_N  Omega+gamma^-1 I_N  |  | a  |   | 1 |
#|_                        _|  |_  _|   |_ _|
# Vector of ones becomes labelvector y in classification

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class LSSVMClassification(BaseEstimator, ClassifierMixin):
    """
    An Least Squared Support Vector Machine (LS-SVM) classification class, built
    on the BaseEstimator and ClassifierMixin base classes of sklearn.

    Attributes:
        - gamma : the regularization hyper-parameter (float)
        - kernel: the kerneltype ('linear', 'poly', 'rbf') used     (string)
        - kernel_: the selected kernel function
        - x : the data on which the LSSVM is trained (call it support vectors)
        - y : binary class labels (-1 or +1)
        - coef_ : coefficents of the support vectors
        - intercept_ : (intercept) bias term

    """
    def __init__(self, gamma: float = 1.0, kernel: str = None, c: float = 1.0,
                 d: float = 2, sigma: float = 1.0): #stays the same for classification
        """
        Create a new LS-SVM classifier.

        Parameters:
        --------------
            - gamma: floating point value for the hyper-parameter gamma, DEFAULT=1.0
            - kernel: string indicating the kernel: {'linear','poly','rbf'}, DEFAULT='rbf'
            - the kernel parameters
                    * linear: none
                    * poly:
                        + c: float
                        scaling constant, DEFAULT=1.0
                        + d: float
                        polynomial power, DEFAULT=2
                    * rbf:
                        + sigma: float
                        scaling constant, DEFAULT=1.0
        """
        self.gamma = gamma
        self.c = c
        self.d = d
        self.sigma = sigma
        if kernel is None:
            self.kernel = 'rbf'
        else:
            self.kernel = kernel

        params = dict()
        if kernel == 'poly':
            params['c'] = c
            params['d'] = d
        elif kernel == 'rbf':
            params['sigma'] = sigma

        self.kernel_ = LSSVMClassification.__set_kernel(self.kernel, **params)

        #model parameters
        self.x = None
        self.y = None
        self.coef_ = None
        self.intercept_ = None

    def get_params(self, deep=True): #stays the same for classification
        """
            Return the hyperparameters of the LS-SVM Classificier. 

            The learned model parameters (support vectors, coefficients and
            intercept) are not included.
        """
        return {"c": self.c, "d": self.d, "gamma": self.gamma,
                "kernel": self.kernel, "sigma":self.sigma}

    def set_params(self, **parameters): #stays the same for classification
        """
            Set the parameters of the classifier. 

            This method mirrors the behaviour of __init__(), which is required
            for compatiblity with scikit-learn tools such as GridSearchCV.
            More info:  https://scikit-learn.org/stable/developers/develop.html
        """
        #print("SETTING PARAMETERS IN LSSVM:",parameters.items())

        for parameter, value in parameters.items():
            #setattr should do the trick for gamma,c,d,sigma and kernel
            setattr(self, parameter, value)
        #now also update the actual kernel
        params = dict()
        if self.kernel == 'poly':
            params['c'] = self.c
            params['d'] = self.d
        elif self.kernel == 'rbf':
            params['sigma'] = self.sigma
        self.kernel_ = LSSVMClassification.__set_kernel(self.kernel, **params)

        return self

    def set_attributes(self, **parameters): #stays the same for classification
        """
            Manually set the attributes of the model. This should generally
            not be done, except when testing some specific behaviour, or
            creating an averaged model. 

            This method is mainly intended for testing or constructing pre-trained
            models.

            Parameters are provided as a dictionary.
                - 'intercept_' : float intercept
                - 'coef_'      : float array of coefficients
                - 'support_'   : array of support vectors, in the same order sorted
                                 as the coefficients
        """
        #not the most efficient way of doing it...but sufficient for the time being
        for param, value in parameters.items():
            if param == 'intercept_':
                self.intercept_ = value
            elif param == 'coef_':
                self.coef_ = value
            elif param == 'support_':
                self.x = value

    @staticmethod
    def __set_kernel(name: str, **params): #stays the same for classification
        """
            Internal static function to set the kernel function.
            NOTE: The second "vector" xj will be the one which generally
                  contains an array of possible vectors, while xi should be a single
                  vector. Therefore, the numpy dot-product requires xj to
                  be transposed.
            The same kernel definitions are used for both regression and classification.

            The kernel returns either a scalar or a numpy nd-array of
            rank 1 (i.e. a vector), if it returns something else the result
            is wrong if xi is an array.

        """
        def linear(xi, xj):
            """
               v*v=scal (dot-product OK)
               v*m=v    (dot-product OK)
               m*m=m    (matmul for 2Dx2D, ok with dot-product)
            """
            return np.dot(xi, xj.T)

        def poly(xi, xj, c=params.get('c', 1.0), d=params.get('d', 2)):
            """
                Polynomial kernel ={1+ (xi*xj^T)/c }^d

                Parameters:
                    - c: scaling constant, DEFAULT=1.0
                    - d: polynomial power, DEFAULT=2
                    - xi and xj are numpy nd-arrays
                (cf: https://en.wikipedia.org/wiki/Least-squares_support-vector_machine )

                works on same as linear
            """
            return ((np.dot(xi, xj.T))/c  + 1)**d

        def rbf(xi, xj, sigma=params.get('sigma', 1.0)):
            """
            Radial Basis Function kernel= exp(- ||xj-xi||² / (2*sigma²))
            In this formulation, the rbf is also known as the Gaussian kernel of variance sigma²
            As the Euclidean distance is strict positive, the results of this kernel
            are in the range [0..1] (x € [+infty..0])

            Parameters:
                - sigma: scaling constant, DEFAULT=1.0
                - xi and xj are numpy nd-arrays
            (cf: https://en.wikipedia.org/wiki/Least-squares_support-vector_machine )

            Possible combinations of xi and xj:
                vect & vect   -> scalar
                vect & array  -> vect
                array & array -> array => this one requires a pair distance...
                                    which can not be done with matmul and dot

                The vectors are the rows of the arrays (Arr[0,:]=first vect)

                The squared distance between vectors= sqr(sqrt( sum_i(vi-wi)² ))
                --> sqr & sqrt cancel
                --> you could use a dot-product operator for vectors...but this
                seems to fail for nd-arrays.

            For vectors:
                ||x-y||²=sum_i(x_i-y_i)²=sum_i(x²_i+y²_i-2x_iy_i)
                --> all products between vectors can be done via np.dot: takes the squares & sum

            For vector x and array of vectors y:
                --> x²_i : these are vectors: dot gives a scalar
                --> y²_i : this should be a list of scalars, one per vector.
                            => np.dot gives a 2d array
                            => so   1) square manually (squares each element)
                                    2) sum over every row (axis=1...but only in case we
                                                           have a 2D array)
                --> x_iy_i : this should also be a list of scalars. np.dot does the trick,
                            and even gives the same result if matrix and vector are exchanged

            for array of vectors x and array of vectors y:
                --> either loop over vectors of x, and for each do the above
                --> or use cdist which calculates the pairwise distance and use that in the exp

            """
            from scipy.spatial.distance import cdist

           # print('LS_SVM DEBUG: Sigma=',sigma,'  type=',type(sigma) )
           # print('              xi   =',xi,'  type=',type(xi))
           # print('              xj   =',xj,'  type=',type(xj))


            if (xi.ndim == 2 and xi.ndim == xj.ndim): # both are 2D matrices
                return np.exp(-(cdist(xi, xj, metric='sqeuclidean'))/(2*(sigma**2)))
            elif ((xi.ndim < 2) and (xj.ndim < 3)):
                ax = len(xj.shape)-1 #compensate for python zero-base
                return np.exp(-(np.dot(xi, xi) + (xj**2).sum(axis=ax)
                                - 2*np.dot(xi, xj.T))/(2*(sigma**2)))
            else:
                message = "The rbf kernel is not suited for arrays with rank >2"
                raise Exception(message)

        kernels = {'linear': linear, 'poly': poly, 'rbf': rbf}
        if kernels.get(name) is not None:
            return kernels[name]
        else: #unknown kernel: crash and burn?
            message = "Kernel "+name+" is not implemented. Please choose from : "
            message += str(list(kernels.keys())).strip('[]')
            raise KeyError(message)

    def __OptimizeParams(self):
        # Classification: labels need to be included in kernelmatrix Ωij=yiyj*K(xi,xj) 
        Omega = np.multiply.outer(self.y, self.y) * self.kernel_(self.x, self.x) #in fit(): trainingdata saved as self.x and self.y
        #Ones = np.array([[1]]*len(self.y)) # needs to be a 2D 1-column vector, hence [[ ]]
        y = np.array([self.y]).T
        
        A_dag = np.linalg.pinv(np.block([
            [0, y.T],
            [y, Omega + self.gamma**-1 * np.eye(len(y))]
        ])) # 2D kolomvector

        B = np.concatenate((np.array([0]), np.ones(len(self.y))))
        
        solution = np.dot(A_dag, B)
        
        
        self.intercept_ = solution[0]
        self.coef_      = solution[1:] #these never change 

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the LS-SVM classifier using the training data.

        The training samples are stored as support vectors and the optimization
        problem is solved to determine the model parameters.

        We are doing Classification.
        Parameters:
        ---------------
            - X : ndarray
            2D array of training features (n_samples x n_features)
            - y : 1D vector of binary class labels (0 or 1)
        """


        if isinstance(X, (pd.DataFrame, pd.Series)): #checks if X is an instance of either types
            Xloc = X.to_numpy()
        else:
            Xloc = X

        if isinstance(y, (pd.DataFrame, pd.Series)):
            yloc = y.to_numpy()
        else:
            yloc = y

        #check the dimensionality of the input
        if (Xloc.ndim == 2) and (yloc.ndim == 1):

            self.x = Xloc
            self.y = yloc

            labels = np.unique(self.y)

            if len(labels) != 2:
                raise ValueError(
                    "LSSVMClassification supports only binary classification."
                )

            if not np.array_equal(np.sort(labels), np.array([-1, 1])):
                raise ValueError(
                    "Labels need to be -1 and 1."
                )

            self.__OptimizeParams()

        else:
            message = (
                "The fit procedure requires a 2D numpy array of features "
                "and a 1D array of targets."
            )
            raise Exception(message)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the decision values for a set of feature vectors.
        
        Parameters
        ----------
        X : ndarray
            Feature vectors to classify.

        Returns
        -------
        ndarray
            Decision values of the classifier.
        """

        Ker = self.kernel_(X, self.x)

        decision = np.dot(Ker, self.coef_ * self.y) + self.intercept_

        return decision

      
    def predict(self, X: np.ndarray)->np.ndarray:
        """
        Predict the class labels for a set of feature vectors

        Parameters:
        -----------
            - X: ndarray of feature vectors to classify (max: 2D), 1 per row if more than one.

        Returns
        --------
        ndarray
            Predicted class labels. 

        """
        Ker = self.kernel_(X, self.x) 
        #decision= np.dot(self.coef_, Ker.T) + self.intercept_
        decision = np.dot(Ker, self.coef_ * self.y) + self.intercept_
        
        
        return np.sign(decision)




