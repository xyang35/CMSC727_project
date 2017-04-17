"""
This script is what created the dataset pickled.

1) You need to download this file and put it in the same directory as this file.
https://github.com/moses-smt/mosesdecoder/raw/master/scripts/tokenizer/tokenizer.perl . Give it execution permission.

2) Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/ and extract it in the current directory.

3) Then run this script.
"""

dataset_path='/Users/Dantong_Ji/Desktop/cmsc727/aclImdb/'

import numpy
import cPickle as pkl

from collections import OrderedDict

import glob
import os

from subprocess import Popen, PIPE

# tokenizer.perl is from Moses: https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer
tokenizer_cmd = ['./tokenizer.perl', '-l', 'en', '-q', '-']


def tokenize(sentences):

    print 'Tokenizing..',
    text = "\n".join(sentences)
    tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE)
    tok_text, _ = tokenizer.communicate(text)
    toks = tok_text.split('\n')[:-1]
    print 'Done'

    return toks


def build_dict(path):
    sentences = []
    currdir = os.getcwd()
    os.chdir('%s/pos/' % path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    os.chdir('%s/neg/' % path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    os.chdir(currdir)

    sentences = tokenize(sentences)

    print 'Building dictionary..',
    wordcount = dict()
    for ss in sentences:
        words = ss.strip().lower().split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1

    counts = wordcount.values()
    keys = wordcount.keys()

    sorted_idx = numpy.argsort(counts)[::-1]

    worddict = dict()

    for idx, ss in enumerate(sorted_idx):
        worddict[keys[ss]] = idx+2  # leave 0 and 1 (UNK)

    print numpy.sum(counts), ' total words ', len(keys), ' unique words'

    return worddict


def grab_data(path, dictionary):
    sentences = []
    scores = []
    currdir = os.getcwd()
    os.chdir(path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
            start = ff.index('_')
            end = ff.index('.txt')
            scores.append(int(ff[start+1:end]))
    os.chdir(currdir)
    sentences = tokenize(sentences)

    seqs = [None] * len(sentences)
    for idx, ss in enumerate(sentences):
        words = ss.strip().lower().split()
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]

    return seqs, scores


def main():
    # Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/
    path = dataset_path
    dictionary = build_dict(os.path.join(path, 'train'))

    train_x_pos, train_pos_scores = grab_data(path+'train/pos', dictionary)
    train_x_neg, train_neg_scores = grab_data(path+'train/neg', dictionary)
    train_x = train_x_pos + train_x_neg
    train_scores = train_pos_scores + train_neg_scores
    train_y = [1] * len(train_x_pos) + [0] * len(train_x_neg)

    test_x_pos, test_pos_scores = grab_data(path+'test/pos', dictionary)
    test_x_neg, test_neg_scores = grab_data(path+'test/neg', dictionary)
    test_x = test_x_pos + test_x_neg
    test_scores = test_pos_scores + test_neg_scores
    test_y = [1] * len(test_x_pos) + [0] * len(test_x_neg)

    f = open('imdb_train.pkl', 'wb')
    pkl.dump((train_x, train_y, train_scores), f, -1)
    f.close()
    f = open('imdb_test.pkl', 'wb')
    pkl.dump((test_x, test_y, test_scores), f, -1)
    f.close()

    f = open('imdb.dict.pkl', 'wb')
    pkl.dump(dictionary, f, -1)
    f.close()

if __name__ == '__main__':
    main()
