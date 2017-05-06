from __future__ import print_function
import sys
import os
import numpy as np
import pickle
import time
import pdb
import random

import theano
import theano.tensor as T

from model import RNN
import utils


SEQ_LEN = 200
N_INPUT = 10000
N_EMBED = 512
N_HIDDEN = 10
N_OUTPUT = 2
CELL = 'lstm'

LEARNING_RATE = 1e-3
BATCH_SIZE = 23
MAX_EPOCH = 10

USE_VAL = True

################### Load data ####################

dataset = pickle.load(open('../data/imdb_train.pkl', 'r'))

x_train, mask_train, y_train = utils.prepare_data(dataset[0], dataset[1], SEQ_LEN)
print ("Training size = %d" % x_train.shape[1], file=sys.stderr)

if USE_VAL:
    val_dataset = pickle.load(open('../data/imdb_valid.pkl', 'r'))
    x_val, mask_val, y_val = utils.prepare_data(val_dataset[0], val_dataset[1], SEQ_LEN)

    print ("Validation size = %d" % x_val.shape[1], file=sys.stderr)


################### Training  ####################

# Initial and the model
model = RNN(n_input=N_INPUT, n_embed=N_EMBED, n_hidden=N_HIDDEN, n_output=N_OUTPUT, seq_len=SEQ_LEN, cell_type=CELL)

model.set_train(lr = LEARNING_RATE, batch_size = BATCH_SIZE)

# Start training
print ("Start training ...")

if USE_VAL:
    model.train_all(x_train, mask_train, y_train, x_val, mask_val, y_val, max_epoch = MAX_EPOCH)
else:
    model.train_all(x_train, mask_train, y_train, max_epoch = MAX_EPOCH)


model.save_model('../models/example.pkl')

######################  Testing  ######################

test_dataset = pickle.load(open('../data/imdb_test.pkl', 'r'))
x_test, mask_test, y_test = utils.prepare_data(test_dataset[0], test_dataset[1], SEQ_LEN)

prob = model.predict(x_test, mask_test)
pred = np.argmax(prob, axis=1)
acc = 1. * np.sum(pred == y_test) / pred.shape[0]

print ("Acc: %f" % acc)
