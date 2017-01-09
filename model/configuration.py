#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Deep CCA configuration

Author: M. Sam Ribeiro
Date: 2017

"""


class Config(object):

    def __init__(self):

        self.learning_rate = 0.01
        self.epochs        = 100
        self.batch_size    = 10000

        # L1 and L2 regularization -- not implemented yet
        #self.L1_reg       = 0.00
        #self.L2_reg       = 0.0001

        self.cca_dim  = 50      # number of CCA dimensions
        self.cca_reg1  = 1e-4   # CCA regularization for view 1
        self.cca_reg2  = 1e-4   # CCA regularization for view 2

        # architectures for view1 and view2
        # each list item indicates a hidden layer, each int the number of nodes
        # [1000, 1000, 1000] is 3 hidden layers with 1000 nodes each
        self.architecture1 = [1024, 512, 256, 128]
        self.architecture2 = [1024, 512, 256, 128]


    def set_data_config(self, data):
        train_set_x1, train_set_x2 = data[0]
        valid_set_x1, valid_set_x2 = data[1]
        test_set_x1,  test_set_x2  = data[2]

        self.n_train_batches = train_set_x1.get_value(borrow=True).shape[0] // self.batch_size
        self.n_valid_batches = valid_set_x1.get_value(borrow=True).shape[0] // self.batch_size
        self.n_test_batches = test_set_x1.get_value(borrow=True).shape[0]   // self.batch_size

        self.x1_dim = train_set_x1.get_value(borrow=True).shape[1]
        self.x2_dim = train_set_x2.get_value(borrow=True).shape[1]
