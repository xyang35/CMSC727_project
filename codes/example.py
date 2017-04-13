from __future__ import print_function
import sys
import os
import numpy as np
import pickle
import time
import pdb

import theano
import theano.tensor as T

from model import RNN
import utils


SEQ_LEN = 8
N_INPUT = 10
N_HIDDEN = 10
N_OUTPUT = 2


# Generate synthesis data
feat1 = np.random.normal(0, 0.1, (SEQ_LEN, 100, N_INPUT))
feat2 = np.random.normal(5, 0.01, (SEQ_LEN, 100, N_INPUT))
mask = np.ones((SEQ_LEN, 200), dtype='float32')
feat = np.concatenate((feat1, feat2), axis=1)
feat = feat.astype('float32')
label = np.zeros(200, dtype='int32')
label[100:] = 1


# Initial and traing the model
model = RNN(n_input=N_INPUT, n_hidden=N_HIDDEN, n_output=N_OUTPUT, seq_len=SEQ_LEN, cell_type='rnn')

model.train(feat, mask, label)

model.save_model('../models/example.pkl')

# Testing
prob = model.predict(feat, mask)
pred = np.argmax(prob, axis=1)
acc = 1. * np.sum(pred == label) / pred.shape[0]

print ("Acc: %f" % acc)