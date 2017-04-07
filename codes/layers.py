import pdb
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import *
mode = theano.Mode(linker='cvm') #the runtime algo to execute the code is in c


class RNNLayer(object):
    def __init__(self, input=None, mask=None,
            n_input=0, n_hidden=128, seq_len=16, cell_type='rnn'):

        self.n_hidden = int(n_hidden)
        self.seq_len = int(seq_len)
        self.n_input = int(n_input)

        self.input = input
        self.mask = mask


        self.cell_type = cell_type.lower()
        if self.cell_type == 'rnn':
            self.cell = RNN_cell(n_input, n_hidden)
        elif self.cell_type == 'lstm':
            self.cell = LSTM_cell(n_input, n_hidden)
        elif self.cell_type == 'gru':
            self.cell = GRU_cell(n_input, n_hidden)
        else:
            raise NotImplementedError

        self.params = self.cell.params
        self.reg = self.cell.L2

        # recurrent
        self.h0 = T.fmatrix('h0')

        def recurrent(i_t, m_t, h_tm1):
            # broadcast in the second dimension
            m_t = m_t.dimshuffle(0, 'x')
            
            # recurrent layer
            h_t = self.cell.step(i_t, h_tm1)
            h_t *= m_t

            return h_t

        self.h, _ = theano.scan(recurrent,
                sequences = [self.input, self.mask],
                outputs_info = [self.h0])

        self.feat = self.h1[-1]


class LeastSquareLayer(object):
    def __init__(self, input = None, y = None,
            n_input=128, n_output=0):

        self.n_input = int(n_input)
        self.n_output = int(n_output)

        self.input = input
        self.y = y

        # model params
        self.W_hy = glorot_uniform(n_input, n_output, name = 'W_hy')
        self.b_y = theano.shared(value=np.zeros((n_output,), dtype=theano.config.floatX), name='b_y')
        self.params = [self.W_hy, self.b_y]

        self.pred = T.dot(input, self.W_hy) + self.b_y
        self.loss = T.mean(0.5 * (self.pred - self.y) ** 2)
 

 class SoftmaxLayer(object):
    def __init__(self, input = None, y = None, lambda_l2=0,
            n_input=128, n_output=0):

        self.n_input = int(n_input)
        self.n_output = int(n_output)
        self.lambda_l2 = float(lambda_l2)

        self.input = input
        self.y = y

        # model params
        self.W_hy = glorot_uniform(n_input, n_output, name = 'W_hy')
        self.b_y = theano.shared(value=np.zeros((n_output,), dtype=theano.config.floatX), name='b_y')
        self.params = [self.W_hy, self.b_y]
        self.reg = 0.5 * (self.W_hy ** 2).sum()


        self.pred = T,nnet.softmax(T.dot(input, self.W_hy) + self.b_y)
        self.loss = T.mean(T.nnet.categorical_crossentropy(self.pred, self.y))

        self.cost = self.loss + self.lambda_l2 * self.reg      


class RNN_cell(object):
    def __init__(self, n_input, n_hidden):

        self.W = glorot_uniform(n_input, n_hidden, name='W')
        self.U = glorot_uniform(n_hidden, n_hidden, name='U')
        self.b = theano.shared(value = np.zeros((n_hidden,), dtype=theano.config.floatX), name = 'b')

        self.params = [self.W, self.U, self.b]

        # L2 regularization
        self.L2 = 0.5 * ((self.W ** 2).sum() + \
                (self.U ** 2).sum())

    # feed forward
    def step(self, i_t, h_tm1):
        h = T.tanh(T.dot(i_t, self.W) + T.dot(h_tm1, self.U) + self.b)

        return h


class GRU_cell(object):
    def __init__(self, n_input, n_hidden):

        # update gate
        self.W_z = glorot_uniform(n_input, n_hidden, name='W_z')
        self.U_z = orthogonal(n_hidden, n_hidden, name='U_z')
        self.b_z = theano.shared(value = np.zeros((n_hidden,), dtype=theano.config.floatX), name = 'b_z')

        # reset gate
        self.W_r = glorot_uniform(n_input, n_hidden, name='W_r')
        self.U_r = orthogonal(n_hidden, n_hidden, name='U_r')
        self.b_r = theano.shared(value = np.ones((n_hidden,), dtype=theano.config.floatX), name = 'b_r')

        # hidden weights
        self.W_h = glorot_uniform(n_input, n_hidden, name='W_h')
        self.U_h = orthogonal(n_hidden, n_hidden, name='U_h')
        self.b_h = theano.shared(value = np.zeros((n_hidden,), dtype=theano.config.floatX), name = 'b_h')


        self.params = [self.W_z, self.U_z, self.b_z,
                self.W_r, self.U_r, self.b_r,
                self.W_h, self.U_h, self.b_h]

        # L2 regularization
        self.L2 = 0.5 * ((self.U_z ** 2).sum() + \
                (self.U_r ** 2).sum() + \
                (self.U_h ** 2).sum())

    # feed forward
    def step(self, i_t, h_tm1):
        x_z = T.dot(i_t, self.W_z)
        u_z = T.dot(h_tm1, self.U_z)
        z = T.nnet.hard_sigmoid(x_z + u_z + self.b_z)

        x_r = T.dot(i_t, self.W_r)
        u_r = T.dot(h_tm1, self.U_r)
        r = T.nnet.hard_sigmoid(x_r + u_r + self.b_r)

        # candidate activation
        x_h = T.dot(i_t, self.W_h)
        u_h = T.dot(r * h_tm1, self.U_h)
        can_h = T.tanh(x_h + u_h + self.b_h)

        h = (1 - z) * h_tm1 + z * can_h

        return h
