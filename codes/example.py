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


SEQ_LEN = 100
N_INPUT = 10000
N_EMBED = 512
N_HIDDEN = 10
N_OUTPUT = 2

LEARNING_RATE = 1e-3
BATCH_SIZE = 23
MAX_EPOCH = 10

USE_VAL = False

################### Load data ####################

dataset = pickle.load(open('../data/imdb_train.pkl', 'r'))
data = dataset[0]
label = dataset[1]
N = len(data)

if USE_VAL:
    val_dataset = pickle.load(open('../data/imdb_val.pkl', 'r'))
    x_val, mask_val, y_val = utils.prepare_data(val_dataset[0], val_dataset[1], SEQ_LEN)

    print ("Validation size = %d" % x_val.shape[1], file=sys.stderr)


################### Training  ####################

# Initial and the model
model = RNN(n_input=N_INPUT, n_embed=N_EMBED, n_hidden=N_HIDDEN, n_output=N_OUTPUT, seq_len=SEQ_LEN, cell_type='rnn')

model.set_train(lr = LEARNING_RATE, batch_size = BATCH_SIZE)

# Start training
print ("Start training ...")

for epoch in range(MAX_EPOCH):

    # random shuffling
    temp = zip(data,label)
    random.shuffle(temp)
    (data, label) = zip(*temp)

    idx = 0
    cost = 0
    batch_count = 0
    while idx + BATCH_SIZE < N:

        # generate one batch
        x = np.zeros((SEQ_LEN, BATCH_SIZE))
        mask = np.zeros((SEQ_LEN, BATCH_SIZE), dtype='float32')
        y = np.zeros(BATCH_SIZE)
    
        count = 0
        while count < BATCH_SIZE and idx < N:
            
            # ignore sentence length larger than SEQ_LEN
            if len(data[idx]) < SEQ_LEN:
                x[-len(data[idx]):, count] = data[idx]
                mask[-len(data[idx]), count] = 1
                y[count] = label[idx]

                count += 1

            idx += 1

        # Training
        x = x.astype('int32')
        y = y.astype('int32')
        c = model.train_batch(x, mask, y)

        batch_count += 1
        if batch_count % 50 == 0:
            if USE_VAL:
                vc = model.eval_batch(x_val, mask_val, y_val)
            else:
                vc = 0

            print ("Epoch {}, Batch {}, train cost = {}, val cost = {}".format(
                epoch, batch_count, c, vc), file=sys.stderr)

        cost += c

    cost /= batch_count
    print ("Epoch {}, Average train cost = {}".format(epoch, cost), file=sys.stderr)
    

model.save_model('../models/example.pkl')

######################  Testing  ######################

test_dataset = pickle.load(open('../data/imdb_test.pkl', 'r'))
x_test, mask_test, y_test = utils.prepare_data(test_dataset[0], test_dataset[1], SEQ_LEN)

prob = model.predict(x_test, mask_test)
pred = np.argmax(prob, axis=1)
acc = 1. * np.sum(pred == y_test) / pred.shape[0]

print ("Acc: %f" % acc)
