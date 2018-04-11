import os
import pickle
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import tensorflow as tf
import ipdb

tb_dir = os.path.expanduser('~/lstmLanguageModel/models/tensorboard/embedding_visualization/')

embedding_file = os.path.expanduser('~/lstmLanguageModel/data/ptb/embeddings/word2vec/embeddingMatrix')
vocab_dict = os.path.expanduser('~/lstmLanguageModel/data/ptb/embeddings/word2vec/vocab_dict')

with open(embedding_file, 'rb') as f:
    embds = pickle.load(f)

vocab_size, embd_size = embds.shape

with open(vocab_dict, 'rb') as f:
    vocab_dict = pickle.load(f)

tf.reset_default_graph()
sess = tf.InteractiveSession()

X = tf.Variable([0.0], name='embedding')
place = tf.placeholder(tf.float32, shape=[vocab_size, embd_size])
set_x = tf.assign(X, place, validate_shape=False)

sess.run(tf.global_variables_initializer())
sess.run(set_x, feed_dict={place: embds})

# Sort the dictionary by value
sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])

# Get list of the items
vocab_words = [sorted_vocab[i][0] for i in range(len(sorted_vocab))]

path = tb_dir + 'vocab.tsv'
with open(path, 'w') as f:
    f.write('\n'.join(map(str,vocab_words)))

# create a TensorFlow summary writer
summary_writer = tf.summary.FileWriter(tb_dir + '/' + 'emb_viz.log', sess.graph)
config = projector.ProjectorConfig()
embedding_conf = config.embeddings.add()
embedding_conf.tensor_name = 'embedding:0'
embedding_conf.metadata_path = os.path.join(tb_dir, 'vocab.tsv')
projector.visualize_embeddings(summary_writer, config)

saver = tf.train.Saver()
saver.save(sess, os.path.join(tb_dir, "model.ckpt"))
