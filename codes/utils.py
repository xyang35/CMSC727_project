import pdb
from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T
import copy

'''
Content in this script:
    1. Initialization methods: glorot, orthogonal
    2. Loss for correlation network, cca network
    3. Optimizers: RMSprop, reset_updates
    4. Data generation: gen_rand_batch, gen_feat, gen_other, pack_longrange
    5. Reverse sequence: reverse
    6. Early stopping: early_stop_checker
'''

def glorot_uniform(fan_in, fan_out, name=None):
    ''' Glorot Initialization method '''

    s = np.sqrt(2. / (fan_in + fan_out))
    return uniform(fan_in, fan_out, s, name)

def orthogonal(fan_in, fan_out, scale=1.1, name=None):
    ''' Orthogonal Initialization '''
    shape = (fan_in, fan_out)
    a = np.random.normal(0.0, 1.0, (fan_in, fan_out))
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == shape else v
    q = q.astype(theano.config.floatX)
    return theano.shared(scale * q[:fan_in, :fan_out], name=name)

def uniform(fan_in, fan_out, scale=0.05, name=None):
    return theano.shared(np.asarray(np.random.uniform(low=-scale, high=scale, size=(fan_in, fan_out)), dtype=theano.config.floatX), name=name)

def corr_loss(y1, y2):
    ''' Calculate correlation of two data views
        Reference: Correlational Neural Network'''

    y1_mean = T.mean(y1, axis=0)
    y1_centered = y1 - y1_mean
    y2_mean = T.mean(y2, axis=0)
    y2_centered = y2 - y2_mean

    corr_nr = T.sum(y1_centered * y2_centered, axis=0)
    corr_dr1 = T.sqrt(T.sum(y1_centered * y1_centered, axis=0)+1e-8)
    corr_dr2 = T.sqrt(T.sum(y2_centered * y2_centered, axis=0)+1e-8)
    corr_dr = corr_dr1 * corr_dr2
    corr = corr_nr / corr_dr

    return -T.sum(corr)    # maximize correlation equal to minimize loss

def weighted_corr_loss(y1, y2, e):
    ''' Calculate correlation of two data views
        Reference: Correlational Neural Network'''

    w = 2 * (1 - e.max(axis=1))
    w = w - w + 1
    w = w.dimshuffle(0, 'x') 
    
    y1_mean = T.mean(y1, axis=0)
    y1_centered = y1 - y1_mean
    y2_mean = T.mean(y2, axis=0)
    y2_centered = y2 - y2_mean

    corr_nr = T.sum(w * y1_centered * y2_centered, axis=0)
    corr_dr1 = T.sqrt(T.sum(w * y1_centered * y1_centered, axis=0)+1e-8)
    corr_dr2 = T.sqrt(T.sum(w * y2_centered * y2_centered, axis=0)+1e-8)
    corr_dr = corr_dr1 * corr_dr2
    corr = corr_nr / corr_dr

    return -T.sum(corr)    # maximize correlation equal to minimize loss

def cca_loss(y1, y2, lamda=0.1):
    ''' Approximated cca loss of two views '''

    y1_mean = T.mean(y1, axis=0)
    y1_centered = y1 - y1_mean
    y2_mean = T.mean(y2, axis=0)
    y2_centered = y2 - y2_mean

    corr_nr = T.sum(y1_centered * y2_centered, axis=0)
    corr_dr1 = T.sqrt(T.sum(y1_centered * y1_centered, axis=0)+1e-8)
    corr_dr2 = T.sqrt(T.sum(y2_centered * y2_centered, axis=0)+1e-8)
    corr_dr = corr_dr1 * corr_dr2
    corr = corr_nr / corr_dr

    #C12 = T.dot(y1_centered.T, y2_centered)# / y1_centered.shape[0]
    #l12 = 0.5*((C12 ** 2).sum() - (T.diagonal(C12) ** 2).sum())
    C11 = T.dot(y1_centered.T, y1_centered) / y1_centered.shape[0]
    l11 = 0.5*((C11 ** 2).sum() - (T.diagonal(C11) ** 2).sum())
    C22 = T.dot(y2_centered.T, y2_centered) / y1_centered.shape[0]
    l22 = 0.5*((C22 ** 2).sum() - (T.diagonal(C22) ** 2).sum())

#    return -T.sum(corr) + lamda*(l11+l22+l12)
    return -T.sum(corr) + lamda*(l11+l22)


def rmsprop(cost, params, lr=0.01, rho=0.9, epsilon=1e-6, clip_flag=True, finetune=False):
    ''' RMSProp updates

        finetune: for supervised training initialized by unsupervised learning
    '''

    grads = T.grad(cost=cost, wrt=params)
    updates = OrderedDict()

    count = 0
    for param, grad in zip(params, grads):
        count += 1
        if clip_flag:
            grad = T.clip(grad, -5, 5)    # clip to mitigate exploding gradients

        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                broadcastable=param.broadcastable)
        accu_new = rho * accu + (1-rho) * grad **2
        
        updates[accu] = accu_new

        if finetune and count <= len(params)-2:
            # for finetuning
            updates[param] = param - (0.01*lr * grad / T.sqrt(accu_new + epsilon))
        else:
            updates[param] = param - (lr * grad / T.sqrt(accu_new + epsilon))

        # set the learning rate of attention parameters to be small
        if param.name[0] == 'A':
            #print param.name

            updates[param] = param - (0.1*lr * grad / T.sqrt(accu_new + epsilon))

    return updates


def gen_rand_batch(batch_size, seq_len, feat1, segid, feat2=None, output=None):
    # for mini-batch implementation
    # default dimension of a batch is:
    # (seq_len, batch_size, dim)

    N, dim1 = feat1.shape
    x1 = np.zeros((batch_size, seq_len, dim1), dtype=T.config.floatX)
    if feat2 is not None:
        _, dim2 = feat2.shape
        x2 = np.zeros((batch_size, seq_ln, dim2), dtype=T.config.floatX)
    mask = np.zeros((batch_size, seq_len), dtype=T.config.floatX)

    if output is not None:
        label = np.zeros((batch_size,), dtype='int32')

    start_idx = np.random.randint(N, size=batch_size)
    end_idx = start_idx + seq_len - 1
    for i in range(batch_size):
        if end_idx[i] > N - 1:
            end_idx[i] = N - 1
        elif segid[end_idx[i]] != segid[start_idx[i]]:
            b = start_idx[i] + 1
            while segid[b] == segid[start_idx[i]]:
                b += 1
            end_idx[i] = b - 1

        l = end_idx[i] - start_idx[i] + 1
        # stack to the rightmost
        x1[i, -l:, :] = feat1[start_idx[i]:end_idx[i] + 1, :]
        if feat2 is not None:
            x2[i, -l:, :] = feat2[start_idx[i]:end_idx[i] + 1, :]

        if output is not None:
            # label is assigned to the longest one
            if l > seq_len/2:
                label[i] = output[start_idx[i]]
            else:
                label[i] = output[end_idx[i]]

        mask[i, -l:] = 1

    x1 = np.transpose(x1, (1,0,2))
    mask = np.transpose(mask)
    if feat2 is not None:
        x2 = np.transpose(x2, (1,0,2))
        if output is not None:
            return x1, x2, mask, label
        else:
            return x1, x2, mask
    else:
        if output is not None:
            return x1, mask, label
        else:
            return x1, mask

def gen_feat(seq_len, stride, feat1, segid, feat2=None):
    # in testing, ordered sample with stride=seq_len/2
    N, dim1 = feat1.shape
    x1 = np.zeros((2 * N / stride, seq_len, dim1), dtype=T.config.floatX)
    if feat2 is not None:
        _, dim2 = feat2.shape
        x2 = np.zeros((2 * N / stride, seq_len, dim2), dtype=T.config.floatX)
    mask = np.zeros((2 * N / stride, seq_len), dtype=T.config.floatX)

    i = 0
    count = 0
    while i < N - seq_len:
        start_idx = i
        end_idx = i + seq_len - 1
        if segid[start_idx] == segid[end_idx]:
            i += stride
        else:
            b = start_idx + 1
            while segid[b] == segid[start_idx]:
                b += 1
            end_idx = b - 1

            i = b    # next sequence start at the beginning of next seg

        l = end_idx-start_idx+1
        # stack to the rightmost
        x1[count, -l:, :] = feat1[start_idx:end_idx + 1, :]
        if feat2 is not None:
            x2[count, -l:, :] = feat2[start_idx:end_idx + 1, :]
        mask[count, -l:] = 1
        count += 1

    x1 = x1[:count]
    mask = mask[:count]
    x1 = np.transpose(x1, (1,0,2))
    mask = np.transpose(mask)
    if feat2 is not None:
        x2 = x2[:count]
        x2 = np.transpose(x2, (1,0,2))
        return x1, x2, mask
    else:
        return x1, mask

def gen_other(seq_len, stride, something, segid):
    # in testing, ordered sample with stride=seq_len/2
    N = something.shape[0]
    s = np.zeros((2 * N / stride,), dtype='int32')

    i = 0
    count = 0
    while i < N - seq_len:
        start_idx = i
        end_idx = i + seq_len - 1
        if segid[start_idx] == segid[end_idx]:
            i += stride
        else:
            b = start_idx + 1
            while segid[b] == segid[start_idx]:
                b += 1
            end_idx = b - 1

            i = b    # next sequence start at the beginning of next seg

        l = end_idx-start_idx+1
        # label is assigned to the longest one
        if l > seq_len/2:
            s[count] = something[start_idx]
        else:
            s[count] = something[end_idx]
        count += 1

    s = s[:count]
    return s

def reverse(A):
    ''' Reverse along the sequence dimension for decoding.
        Default sequence dimenasion is 0.
    '''

    return A[::-1, ...]

class early_stop_checker(object):
    ''' Early stopping
        Reference: http://deeplearning.net/tutorial/gettingstarted.html#early-stopping
    '''

    def __init__(self, patience=10, patience_increase=2, improvement_threshold=0.995):
        self.patience = patience    # Run at least this many epochs
        self.patience_increase = patience_increase
        self.improvement_threshold = improvement_threshold
        self.stop_flag = False
        self.best_cost = np.inf
        self.best_epoch = 0

    def check(self, cost, epoch):
        record_flag = False

        if cost < self.best_cost:

            # imporve patience if loss improvement is good enough
            if cost < self.best_cost * self.improvement_threshold:
                self.patience = max(self.patience, epoch*self.patience_increase)

            self.best_cost = cost
            self.best_epoch = epoch
            record_flag = True

        if self.patience <= epoch:
            self.stop_flag = True

        return self.stop_flag, record_flag


def pack_longrange(feat, segid, seq_len, num, *others):
    """
        num: number of random sampling times for each segmentation
    """

    N, dim = feat.shape
    seg_num = len(set(segid.tolist()))
    print "seg_num: ", seg_num
    x = np.zeros((num*seg_num, seq_len, dim), dtype='float32')

    others_num = len(others)
    x_others = np.zeros((num*seg_num, others_num), dtype='int32')

    count = 0
    i = 0
    while i < N-seq_len:
        # extract a segment
        seg_start = i
        while segid[i] == segid[seg_start] and i<N-1:
            i += 1

        segment = feat[seg_start:i, :]
        N_seg = len(segment)

        # partition segment to seq_len pieces
        par_idx = np.round(np.linspace(0, N_seg, seq_len+1))
        temp = np.zeros((num, seq_len, dim), dtype='float32')
        temp_others = np.zeros((num, others_num), dtype='float32')
        for j in range(seq_len):
            # random sample
            idx = np.random.randint(low=par_idx[j],high=par_idx[j+1],size=num)
            temp[:, j, :] = segment[np.asarray(idx), :]

        # eliminate repeated rows
        temp = temp.tolist()
        temp.append(1)
        temp = np.asarray(temp)[:-1]
        temp = np.unique(temp)
        temp = np.asarray(temp.tolist(), 'float32')

        x[count:count+len(temp)] = temp


        # fill others
        for k in range(len(others)):
            x_others[count:count+len(temp), k] = others[k][seg_start]

        count += len(temp)

    print "count: ", count
    x = x[:count]
    x_others = x_others[:count]
    x = np.transpose(x, (1,0,2))

    return x, x_others
