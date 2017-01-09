#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Theano-based Deep Canonical Correlation Analysis (Deep CCA) on MNIST data.

Training procedure based on code from http://deeplearning.net/tutorial

Author: M. Sam Ribeiro
Date: 2017

"""

import os
import time
import logging

import numpy
import theano
import theano.tensor as T

from model.dnn import ParallelDNN
from model.configuration import Config
from data_provider import load_data


def train(cfg, model):

    logging.info('... training')

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(cfg.n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_score = numpy.inf
    best_epoch = 0
    start_time = time.time()

    epoch = 0
    done_looping = False

    train_model   = model.train
    validate_model = model.valid
    test_model = model.test

    while (epoch < cfg.epochs) and (not done_looping):

        epoch = epoch + 1
        train_correlations = []
        train_means = []
        epoch_start_time = time.time()

        for minibatch_index in range(cfg.n_train_batches):

            minibatch_avg_cost, mini_batch_mean = train_model(minibatch_index)
            train_correlations.append(float(minibatch_avg_cost))
            train_means.append(float(mini_batch_mean))

            iteration = (epoch - 1) * cfg.n_train_batches + minibatch_index

            if (iteration + 1) % validation_frequency == 0:

                epoch_train_corr = numpy.mean(train_correlations)
                epoch_train_mean = numpy.mean(train_means)

                valid_correlations = []
                valid_means = []
                for i in range(cfg.n_valid_batches):
                    valid_corr, valid_mean = validate_model(i)
                    valid_correlations.append(float(valid_corr))
                    valid_means.append(float(valid_mean))

                epoch_valid_corr = numpy.mean(valid_correlations)
                epoch_valid_mean = numpy.mean(valid_means)

                # if we got the best validation score until now
                if epoch_valid_corr < best_validation_score:
                    #improve patience if loss improvement is good enough
                    if (
                        epoch_valid_corr < best_validation_score *
                        improvement_threshold
                    ):
                        patience = max(patience, iteration * patience_increase)

                    best_validation_score = epoch_valid_corr
                    best_epoch = epoch

                epoch_time = time.time() - epoch_start_time
                logging.info(
                    'epoch {0}, train correlation {1:.2f} (mean: {2:.2f}), validation correlation {3:.2f} (mean: {4:.2f}), time {5:.2f}s' \
                        .format(epoch, -epoch_train_corr, epoch_train_mean, -epoch_valid_corr, epoch_valid_mean, epoch_time))

            if patience <= iteration:
                done_looping = True
                break

    running_time = time.time() - start_time
    filename = os.path.split(__file__)[1]

    logging.info('Optimization complete. Best validation score of {0:.2f} obtained at epoch {1:.2f}' \
        .format(-best_validation_score, best_epoch))
    logging.info('The code for file {0} ran for {1:.2f}m'.format(filename, running_time / 60.))


def test(cfg, model):

    test_model = model.test
    test_means = []
    test_correlations = []

    start_time = time.time()

    for i in range(cfg.n_test_batches):
        test_corr, test_mean = test_model(i)
        test_correlations.append(float(test_corr))
        test_means.append(float(test_mean))

    test_corr = numpy.mean(test_correlations)
    test_mean = numpy.mean(test_mean)
    running_time = time.time() - start_time

    logging.info('test correlation {0:.2f} (mean {1:.2f}), time {2:.2f}s' \
        .format(-test_corr, test_mean, running_time))



if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)-15s %(levelname)s: %(message)s',  datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    # load dataset
    dataset = './mnist.pkl.gz'
    datasets = load_data(dataset, shared=True)

    # set configuration
    cfg = Config()
    cfg.set_data_config(datasets)

    # build model
    model = ParallelDNN(cfg, datasets)

    # train
    train(cfg, model)
    test(cfg, model)


# THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,force_device=True python ./deep_cca.py 
