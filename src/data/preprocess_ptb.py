from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""
File: preprocess_ptb.py
Author: Anna-Lena Popkes
Github: https://github.com/zotroneneis
Description: Penn Treebank Preprocessing
Adapted from official TensorFlow tutoral available at:
https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb
"""

import collections
import os
import pickle
import sys
import ipdb

import numpy as np
import tensorflow as tf

Py3 = sys.version_info[0] == 3

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      return f.read().replace("\n", "eos").split()
    else:
      return f.read().decode("utf-8").replace("\n", "eos").split()


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def _tokenize_file(filename):
    tokenized_file = _read_words(filename)
    joined = ','.join(tokenized_file)
    temp = [line.split(',') + ['eof'] for line in joined.split('eos')]
    tokenized_sentences = [[word for word in sublist if word != ''] for sublist in temp]
    tokenized_sentences

    return tokenized_sentences


def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  # build vocabulary that maps each word to an integer id
  word_to_id = _build_vocab(train_path)
  save_vocab = os.path.join(data_path, "vocab_dict")

  with open(save_vocab, 'wb') as f:
      pickle.dump(word_to_id, f)

  # convert each file into a list of word id's
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary


def ptb_producer(raw_data, batch_size, num_steps, save_path):
    data_len = len(raw_data)
    batch_len = data_len // batch_size # total number of batches

    data = np.reshape(raw_data[0: batch_size * batch_len], [batch_size, batch_len])
    # Save numpy array to disk
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

def transform_word2vec_files():
    """
    Transforms training, test and validation files to matrices
    of word ID's using the vocabulary provided by the trained
    word2vec model
    """
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    # open vocabulary
    vocab_path = os.path.expanduser('~/lstmLanguageModel/data/ptb/embeddings/word2vec/vocab_dict')
    with open(vocab_path, 'rb') as f:
        word_to_id = pickle.load(f)

    # convert each file into a list of word id's
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)

    ptb_producer(train_data, batch_size=20, num_steps=35, save_path= root+'/word_ids/word2vec_ptb.train')
    # STEP 2: EVAL DATA
    ptb_producer(valid_data, batch_size=20, num_steps=35, save_path=root+'/word_ids/word2vec_ptb.eval')
    # STEP 3: TEST DATA
    ptb_producer(test_data, batch_size=20, num_steps=35, save_path=root+'/word_ids/word2vec_ptb.test')

def transform_fastText_files():
    """
    Transforms training, test and validation files to matrices
    of word ID's using the vocabulary provided by the trained
    fastText model
    """
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    # open vocabulary
    vocab_path = os.path.expanduser('~/lstmLanguageModel/data/ptb/embeddings/fastText/fastText_vocab_dict')
    with open(vocab_path, 'rb') as f:
        word_to_id = pickle.load(f)

    # convert each file into a list of word id's
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)

    ptb_producer(train_data, batch_size=20, num_steps=35, save_path= root+'/word_ids/fastText_ptb.train')
    # STEP 2: EVAL DATA
    ptb_producer(valid_data, batch_size=20, num_steps=35, save_path=root+'/word_ids/fastText_ptb.eval')
    # STEP 3: TEST DATA
    ptb_producer(test_data, batch_size=20, num_steps=35, save_path=root+'/word_ids/fastText_ptb.test')


if __name__ == "__main__":
    root = os.path.expanduser('~/lstmLanguageModel/data/ptb/')
    data_path = os.path.join(root, 'raw/')

    # Tokenize each file into a list of sentences
    # Needed for training word embeddings
    #train_path = os.path.join(data_path, "ptb.train.txt")
    #valid_path = os.path.join(data_path, "ptb.valid.txt")
    #test_path = os.path.join(data_path, "ptb.test.txt")
    #save_tokenized_sentences = root + '/embeddings/tokenized_sentences'
    #tokenized_train = _tokenize_file(train_path)
    #tokenized_test = _tokenize_file(test_path)
    #tokenized_valid = _tokenize_file(valid_path)
    #tokenized_train.extend(tokenized_test)
    #tokenized_train.extend(tokenized_valid)

    #with open(save_tokenized_sentences, "wb") as f:
    #    pickle.dump(tokenized_train, f)

    # Transform each file to array of word ID's
    #raw_data = ptb_raw_data(data_path)
    #train_data, valid_data, test_data, _ = raw_data
    # STEP 1: TRAIN DATA
    #ptb_producer(train_data, batch_size=20, num_steps=35, save_path= root+'/word_ids/ptb.train')
    # STEP 2: EVAL DATA
    #ptb_producer(valid_data, batch_size=20, num_steps=35, save_path=root+'/word_ids/ptb.eval')
    # STEP 3: TEST DATA
    #ptb_producer(test_data, batch_size=20, num_steps=35, save_path=root+'/word_ids/ptb.test')

    # Transform word2vec files
    transform_word2vec_files()

    # Transform fastText files
    # transform_fastText_files()





