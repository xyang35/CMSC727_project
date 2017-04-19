from __future__ import print_function
import pdb
import numpy as np
import theano
import theano.tensor as T
import pickle
import copy
import sys

from utils import *
import layers

mode = theano.Mode(linker='cvm') #the runtime algo to execute the code is in c


"""
    Single version RNN, for unimodal baseline
"""


class RNN(object):
    def __init__(self, n_input, n_embed, n_hidden, n_output, seq_len, cell_type='gru'):

        # model input (no output, unsupervised)
        self.input = T.imatrix('input')
        self.y = T.ivector('y')

        self.encode_mask = T.fmatrix('encode_mask')

        self.cell_type = cell_type
        self.n_input = n_input
        self.n_embed = n_embed
        self.seq_len = seq_len
        self.n_hidden = n_hidden
        self.n_output = n_output

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print('Building model ...')

        ######### Embedding ############
        self.embed = layers.EmbeddingLayer(
                input = self.input, n_input=self.n_input,
                n_output = self.n_embed)

        ########## Encoder ##############

        self.encoder = layers.RNNLayer(
                input=self.embed.out, mask=self.encode_mask,
                n_input=self.n_embed,
                n_hidden=self.n_hidden, seq_len=self.seq_len,
                cell_type=self.cell_type)


        ########## Predictor ##############

        # Loss layer
        self.ls = layers.SoftmaxLayer(
                input = self.encoder.feat,
                y = self.y,
                n_input = self.n_hidden,
                n_output = self.n_output)


        # combine parameters
        self.params = self.encoder.params + self.ls.params

        self.loss = self.ls.loss
        self.pred = self.ls.pred


    # Set training policy
    def set_train(self, lr=10e-5, batch_size=32, optimizer=rmsprop):

        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = optimizer

        # define symbolic batch input
        input = T.imatrix()
        encode_mask = T.fmatrix()
        y = T.ivector()

        h0 = np.zeros((batch_size,self.n_hidden), dtype=theano.config.floatX)

        updates = optimizer(self.loss, self.params, lr=lr)

        self.train = theano.function(inputs=[input, encode_mask, y],
                outputs = self.loss,
                updates = updates,
                givens = {self.input: input,
                    self.encode_mask: encode_mask,
                    self.y: y,
                    self.encoder.h0: h0},
                mode = mode)



    # train model (load all data)
    def train_all(self, train_feat=None, train_mask=None, train_target=None,
            init_epoch=0, max_epoch=10, path=None):

        print('Training model ...')

#        check_model = theano.function(inputs=[input1, input2, encode_mask, y1, y2, decode_mask],
#                outputs = [self.ls1.loss, self.ls1.y_pred, self.ls2.y_pred],
#                givens = {self.input1: input1,
#                    self.input2: input2,
#                    self.encode_mask: encode_mask,
#                    self.y1: y1,
#                    self.y2: y2,
#                    self.decode_mask: decode_mask,
#                    self.encoder1_1.h0: h0, self.encoder1_2.h0: h0,
#                    self.encoder2.h0: h0,
#                    self.decoder2_1.h0: h0, self.decoder2_2.h0: h0},
#                mode = mode)


        ##################
        # Start Training #
        ##################
        def iterate_minibatches(inputs, mask, targets, batchsize=16, shuffle=False):
            # inputs.shape = (seq_len, N, dim)
        
            if shuffle:
                indices = np.arange(inputs.shape[1])
                np.random.shuffle(indices)
            for start_idx in range(0, inputs.shape[1]-batchsize+1, batchsize):
                if shuffle:
                    excerpt = indices[start_idx:start_idx+batchsize]
                else:
                    excerpt = slice(start_idx, start_idx + batchsize)

                yield inputs[:, excerpt, :], mask[:, excerpt], targets[excerpt]

        record = np.Inf
        patience = 10
        neg_count = 0

        for epoch in range(init_epoch, max_epoch):


            if (epoch+1) % 50 == 0:    # snapshot of the model
                if path is not None:
                    self.save_model(path+'_'+str(epoch+1))

            cost = 0
            count = 0
            for inputs, encode_m, targets in iterate_minibatches(inputs=train_feat, mask=train_mask, targets=train_target, 
                batchsize=batch_size, shuffle=True):

                cost += self.train(inputs, encode_m, targets)
                count += 1
            cost /= count
        
            diff = record - cost
            record = cost
            print("Epoch {} train cost = {}, diff = {}".format(
                epoch, cost, diff), file=sys.stderr)

            #r = check_model(inputs1, inputs2, encode_m, targets1, targets2, decode_m)
            #pdb.set_trace()

            # # stopping for convergence
            # if diff < abs(cost) / 10000.:
            #     if patience <= 0:
            #         print("Stop for convergence!", file=sys.stderr)
            #         break
            #     else:
            #         patience -= 1
            # else:
            #     patience = min(10, patience+5)
    

            # # decrease learning rate if diff < 0
            # if diff < 0:
            #     neg_count += 1
            #     patience -= 3

            #     if neg_count == 2:
            #         lr = lr / 2.
            #         neg_count = 0

            #     else:
            #         lr = max(lr, self.lr / (2**(epoch / 100)))
            #     print ("learning rate = {}".format(lr), file=sys.stderr)

            #     updates = optimizer(self.loss, self.params, lr=lr)

            #     train = theano.function(inputs=[input, encode_mask, y, decode_mask],
            #             outputs = self.loss,
            #             updates = updates,
            #             givens = {self.input: input,
            #                 self.encode_mask: encode_mask,
            #                 self.y: y,
            #                 self.decode_mask: decode_mask,
            #                 self.encoder.h0: h0},
            #             mode = mode)

    # Train model (for one batch)
    def train_batch(self, train_feat=None, train_mask=None, train_target=None):

        ##################
        # Start Training #
        ##################

        return self.train(train_feat, train_mask, train_target)

    # Evaluate model (for one batch)
    def eval_batch(self, feat=None, mask=None, target=None):

        # define symbolic batch input
        input = T.imatrix()
        encode_mask = T.fmatrix()
        y = T.ivector()

        h0 = np.zeros((feat.shape[1],self.n_hidden), dtype=theano.config.floatX)

        evaluate = theano.function(inputs=[input, encode_mask, y],
                outputs = self.loss,
                givens = {self.input: input,
                    self.encode_mask: encode_mask,
                    self.y: y,
                    self.encoder.h0: h0},
                mode = mode)

        return evaluate(feat, mask, target)


    def predict(self, feat=None, mask=None, path=None, batch_size=256):

        print('Testing the model ...')
        # define symbolic batch input
        input = T.imatrix()
        encode_mask = T.fmatrix()
        y = T.ivector()
        h0 = T.fmatrix()

        self.predict = theano.function(inputs=[input, encode_mask, h0],
                outputs = self.pred,
                givens = {self.input: input,
                    self.encode_mask: encode_mask,
                    self.encoder.h0: h0},
                mode = mode)

        pred = np.zeros((feat.shape[1], self.n_output))

        # extract features
        for i in range(0, feat.shape[1], batch_size):
            s = i
            e = min(feat.shape[1], i+batch_size)

            h0_val = np.zeros((e-s,self.n_hidden), dtype=theano.config.floatX)
            temp_pred = self.predict(
                    feat[:,s:e,:], mask[:,s:e], h0_val)

            pred[s:e] = temp_pred

 
        if path is not None:
            with h5py.File(path,'w') as f:
                f.create_dataset("pred", data=pred, dtype='float32')

        return pred

    def save_model(self, path):

        print("Saving model parameters ...")

        with open(path, 'w') as f:
            pickle.dump(self.params, f)

    def set_model(self, path=None, params=None):

        print("Setting model parameters ...")

        if path is not None:
            f = open(path, 'r')
            params = pickle.load(f)

        assert len(self.params) == len(params)

        for i in range(len(params)):
            self.params[i].set_value( params[i].get_value() )
