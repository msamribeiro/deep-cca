#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

MNIST data provider for Deep CCA.
MNIST digits are divided into left and right halves for DCCA model.
These will correspond to 2 views from the same data.

Based on code from http://deeplearning.net/tutorial

Author: M. Sam Ribeiro
Date: 2017

"""

import os
import gzip
import pickle
import logging
import theano
import numpy as np


def load_data(dataset, shared=False):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    logging.info('... loading data')

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        logging.info('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    # Load the dataset
    fid = gzip.open(dataset, 'rb')
    try:
        train_set, valid_set, test_set = pickle.load(fid, encoding='latin1')
    except:
        train_set, valid_set, test_set = pickle.load(fid)
    fid.close()

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def halve_dataset(data_xy):
        """ break MNIST matrix into right and left halves """
        data, label = data_xy
        m, n = data.shape
        left  = np.zeros((m, n/2))
        right = np.zeros((m, n/2))

        for i in range(m):
            image = data[i].reshape(28, 28)
            left[i] = image[:,:14].reshape(1, -1)
            right[i] = image[:,14:].reshape(1, -1)
        return (left, right, label)


    def shared_dataset(data_xxy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x1, data_x2, data_y = data_xxy
        shared_x1 = theano.shared(np.asarray(data_x1,
                                    dtype=theano.config.floatX),
                                    borrow=borrow)
        shared_x2 = theano.shared(np.asarray(data_x2,
                                    dtype=theano.config.floatX),
                                    borrow=borrow)
        return shared_x1, shared_x2

    train_set = halve_dataset(train_set)
    valid_set = halve_dataset(valid_set)
    test_set  = halve_dataset(test_set)

    if shared:
        train_set_x1, train_set_x2 = shared_dataset(train_set)
        valid_set_x1, valid_set_x2 = shared_dataset(valid_set)
        test_set_x1,  test_set_x2  = shared_dataset(test_set)
    else:
        train_set_x1, train_set_x2, train_set_y = train_set
        valid_set_x1, valid_set_x2, valid_set_y = valid_set
        test_set_x1,  test_set_x2,  test_set_y  = test_set

    data = [
        (train_set_x1, train_set_x2),
        (valid_set_x1, valid_set_x2),
        (test_set_x1, test_set_x2)
        ]

    return data
