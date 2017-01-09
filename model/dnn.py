#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Parallel network and feedforward network.

Author: M. Sam Ribeiro
Date: 2017

"""

import numpy
import theano
import theano.tensor as T

from cca_layer import CCA
from layers import HiddenLayer, ConvPoolLayer


class DNN(object):

    def __init__(self, rng, in_x, in_size, architecture, activation=T.tanh):
        ''' Single feedforward Deep Neural Network '''

        self.layers = []
        self.params = []
        self.n_layers = len(architecture)

        assert self.n_layers > 0

        self.x = in_x

        for i in xrange(self.n_layers):
            if i == 0:
                input_size = in_size
            else:
                input_size = architecture[i-1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.layers[-1].output

            hidden_layer = HiddenLayer(rng=rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=architecture[i],
                                        activation=activation)
            self.layers.append(hidden_layer)
            self.params.extend(hidden_layer.params)

        self.output = self.layers[-1].output



class ParallelDNN(object):

    def __init__(self, config, data):
        ''' Parallel DNN with CCA objective function '''

        index = T.lscalar()                             # index to a [mini]batch
        x1 = T.matrix("x1", dtype=theano.config.floatX) # view1 of the data
        x2 = T.matrix("x2", dtype=theano.config.floatX) # view2 of the data

        rng = numpy.random.RandomState(1234)

        # parallel networks
        dnn1 = DNN(rng, x1, config.x1_dim, config.architecture1)
        dnn2 = DNN(rng, x2, config.x2_dim, config.architecture2)

        # CCA objective function
        cca = CCA(config)
        cost, mean = cca.cca(dnn1.output, dnn2.output)

        params = dnn1.params + dnn2.params
        gparams = [T.grad(cost, param) for param in params]

        updates = [
            (param, param - config.learning_rate * gparam)
            for param, gparam in zip(params, gparams)
        ]

        train_set_x1, train_set_x2 = data[0]
        valid_set_x1, valid_set_x2 = data[1]
        test_set_x1,  test_set_x2  = data[2]


        self.train = theano.function(
            inputs=[index],
            outputs=[cost, mean],
            updates=updates,
            givens={
                x1: train_set_x1[index * config.batch_size: (index + 1) * config.batch_size],
                x2: train_set_x2[index * config.batch_size: (index + 1) * config.batch_size]
            }
        )

        self.valid = theano.function(
            inputs=[index],
            outputs=[cost, mean],
            givens={
                x1: valid_set_x1[index * config.batch_size:(index + 1) * config.batch_size],
                x2: valid_set_x2[index * config.batch_size:(index + 1) * config.batch_size]
            }
        )

        self.test = theano.function(
            inputs=[index],
            outputs=[cost, mean],
            givens={
                x1: test_set_x1[index * config.batch_size:(index + 1) * config.batch_size],
                x2: test_set_x2[index * config.batch_size:(index + 1) * config.batch_size]
            }
        )
