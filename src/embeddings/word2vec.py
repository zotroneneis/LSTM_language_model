"""
File: word2vec.py
Author: Anna-Lena Popkes
Email: popkes@gmx.net
Github: https://github.com/zotroneneis
Description:
"""

import numpy as np
import logging
import os
import pickle
import ipdb

import gensim
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.wrappers import FastText

logger = logging.getLogger()
logging.basicConfig(filename='log_word2vec.txt', level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')

def word2vec(pathname, sentences, embedding_size=200, model_type=0, min_count=0, workers=4):
    """
    Trains and saves a word2vec model on a given list of sentences

    Params:
        pathname: path where the embedding matrices should be saved
        sentences: list of tokenized sentences
        embedding_size: size of the embeddings
        model_type: int, 0: CBOW, 1: skip-gram
        min_count: int
        worker: number of workers for parallelization

    The saved file stores the model in a format compatible with the original word2vec
    implementation.
    """

    logger.info("Train word2vec model")
    model = gensim.models.Word2Vec(sentences, size=embedding_size, sg=model_type, min_count=min_count, workers=workers)

    logger.info("Save the trained model")
    model.save(pathname+'word2vec_model')
    # Load a saved model
    # model = gensim.models.Word2Vec.load(pathname+"word2vec_model")

    logger.info("Save the output embedding matrix")
    output_embedding_matrix = model.syn1neg
    with open(pathname+"model_syn1neg", 'wb') as f:
        pickle.dump(output_embedding_matrix, f)

    logger.info("Save the model in word2vec format")
    fname = 'model_word2vecFormat'
    model.wv.save_word2vec_format(pathname+fname)

    # Get all word vectors
    # wv is a KeyedVectors instance
    logger.info("Save the keyedvectors")
    wv = model.wv
    wv.save(pathname+"model_wv")
    del model

    # The word vectors are stored in a numpy array:
    # The number of tows in syn0 is the number of words in the vocabulary
    # The number of columns corresponds to the size of the word vectors
    # model.index2word contains the corresponding list of words in the right order
    input_embedding_matrix = wv.syn0 # input to hidden matrix
    logger.info("Shape of word vector matrix: {}".format(input_embedding_matrix.shape))
    with open(pathname + "model_input_embeddings", "wb") as f:
        pickle.dump(input_embedding_matrix, f)

    logger.info("Average input and output embedding matrices")
    average_embeddings = (input_embedding_matrix + output_embedding_matrix) / 2
    with open(pathname + "model_averaged_embeddings", 'wb') as f:
        pickle.dump(average_embeddings, f)

    # The output matrix (hidden to output layer) is stored in model.syn1
    # Or model.syn1neg for negative sampling)
    # For figure see: https://stackoverflow.com/questions/40458742/gensim-word2vec-accessing-in-out-vectors


def prepareEmbeddings(filename, embedding_size=200):
    """
    Given a file (word2vec format) a vocabulary and embedding matrix are extracted

    For required file format see gensim.models.Word2Vec.wv.save_word2vec_format()

    Params:
        filename: string, path to the file containing the vocabulary and word vectors

    Returns:
        vocab_dict: dictionary mapping each word to its index
        embeddings: numpy array of shape (n_words, embedding_size) containing the word embeddings
    """
    vocab = []
    embedding_index = {}

    # Transform file into separate vocabulary and embedding matrix
    with open(filename, 'r') as f_:
        for line in f_.readlines()[1:]:
            row = line.strip().split(' ')
            embedding = np.asarray(row[1:], dtype='float32')
            # Create dictionary with (word:embedding) pairs
            embedding_index[row[0]] = embedding
            vocab.append(row[0])

    # Create dictionary with (word:index) pairs
    # vocab_dict = {k:v for v, k in enumerate(vocab)}
    # If the index 0 should represent padding we cannot include it in our dictionary
    vocab_dict = {k:v for v, k in enumerate(vocab, 1)}
    # Create embedding matrix, index 0 will remain empty because its reserved for padding
    embedding_matrix = np.zeros((len(vocab_dict)+1, embedding_size))

    # Create embedding matrix
    # embedding_matrix = np.zeros((len(vocab_dict), embedding_size))
    for word, i in vocab_dict.items():
        embed_vector = embedding_index.get(word)
        if embed_vector is not None:
            embedding_matrix[i] = embed_vector

    return vocab_dict, embedding_index, embedding_matrix


if __name__ == '__main__':
    logger.info("Start of program")

    root = os.path.expanduser('~/lstmLanguageModel/data/ptb/embeddings/word2vec/')
    tokens = os.path.expanduser('~/lstmLanguageModel/data/ptb/embeddings/tokenized_sentences')

    with open(tokens, "rb") as f:
       tokenized_sentences = pickle.load(f)

    logger.info("Train and save word2vec model")
    word2vec(root, tokenized_sentences, embedding_size=200, min_count=1, model_type=1)

    logger.info("Extract dictionary and embedding matrix")
    word2vec_model = os.path.join(root, 'model_word2vecFormat')

    vocab_dict, embedding_index, embedding_matrix = prepareEmbeddings(word2vec_model, embedding_size=200)

    logger.info("Save word dictionary and embedding matrix to file")
    save_vocab = os.path.join(root, 'vocab_dict')
    with open(save_vocab, 'wb') as f:
        pickle.dump(vocab_dict, f)

    save_embeddings = os.path.join(root, 'embeddingMatrix')
    with open(save_embeddings, 'wb') as f:
        pickle.dump(embedding_matrix, f)

    logger.info("End of program")
