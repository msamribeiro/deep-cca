#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Theano-based Deep CCA implementation

Mostly based on the following papers:
 (1) Andrew, Galen, et al. 
    "Deep Canonical Correlation Analysis." 
     ICML (3). 2013.

 (2) Wang, Weiran, et al. 
    "Unsupervised learning of acoustic features via deep canonical correlation analysis." 
    2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2015.

Many thanks to Herman Kamper (https://github.com/kamperh) for various resources, comments, and discussions.

Author: M. Sam Ribeiro
Date: 2017

"""


import theano
import theano.tensor as T


class CCA(object):
    ''' Linear Canonical Correlation Analysis (CCA) '''

    def __init__(self, config):

        # input dimension to CCA layer
        self.in_dim1 = config.architecture1[-1]
        self.in_dim2 = config.architecture2[-1]

        # CCA dimension, i.e., number of canonical components
        self.cca_dim = config.cca_dim

        # CCA regularization 
        self.reg1 = config.cca_reg1
        self.reg2 = config.cca_reg1

        # for numerical statibility (from H. Kamper via W. Wang)
        self.eps = 1e-12


    def cca(self, data1, data2):

        n_data = data1.shape[0]

        # center the data
        data1 -= T.mean(data1, axis=0)
        data2 -= T.mean(data2, axis=0)
        data1 = data1.T
        data2 = data2.T

        # find covariance matrices
        sigma11 = (1/(n_data-1.)) * T.dot(data1, data1.T)
        sigma22 = (1/(n_data-1.)) * T.dot(data2, data2.T)
        sigma12 = (1/(n_data-1.)) * T.dot(data1, data2.T)

        # add regulatization
        sigma11 += self.reg1 * T.eye(self.in_dim1)
        sigma22 += self.reg2 * T.eye(self.in_dim2)

        # diagonalize covariance matrices to find inverses
        diag1, q1 = T.nlinalg.eigh(sigma11)
        diag2, q2 = T.nlinalg.eigh(sigma22)

        # numerical stability (from H. Kamper, via W. Wang)
        # http://stackoverflow.com/questions/20590909/returning-the-index-of-a-value-in-theano-vector
        idx = T.gt(diag1, self.eps).nonzero()[0] 
        diag1 = diag1[idx]
        q1 = q1[:, idx]
        idx = T.gt(diag2, self.eps).nonzero()[0]
        diag2 = diag2[idx]
        q2 = q2[:, idx]

        # find correlation matrix T
        sigma11_inv = T.dot(q1, T.dot(T.diag(diag1**(-0.5)), q1.T))
        sigma22_inv = T.dot(q2, T.dot(T.diag(diag2**(-0.5)), q2.T))
        T_corr = T.dot(sigma11_inv, T.dot(sigma12, sigma22_inv))

        # find the singular values of T through the eigenvalues of TT.T
        Tdiag, Tevec = T.nlinalg.eigh(T.dot(T_corr, T_corr.T))
        Tdiag = Tdiag[T.gt(Tdiag, self.eps).nonzero()[0]]
        Tdiag.sort()
        Tdiag = Tdiag[::-1]**(0.5)

        # take the top k canonical components (top k singular values)
        # here we negate corr to treat this as a minimization problem
        corr = -T.sum(Tdiag[:self.cca_dim])
        mean = T.mean(Tdiag[:self.cca_dim])

        return corr, mean

