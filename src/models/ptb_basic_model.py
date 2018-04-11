"""
File: languageModel.py
Author: Anna-Lena Popkes
Email: popkes@gmx.net
Github: https://github.com/zotroneneis
Description: LanguageModel class
"""
import copy
import os
import pickle
import random
import re
import argparse

# Select GPU for training
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import ipdb
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.contrib.tensorboard.plugins import projector

# If needed, change path to home directory
HOME = '/home/apopkes/'

class BasicLanguageModel(object):

    def __init__(self, config):
        self.config = copy.deepcopy(config)
        x_train = os.path.join(HOME, self.config['general']['x_train'])
        x_test = os.path.join(HOME, self.config['general']['x_test'])
        x_eval = os.path.join(HOME, self.config['general']['x_eval'])

        with open(x_train, 'rb') as f:
            self.x_train = pickle.load(f)
        with open(x_test, 'rb') as f:
            self.x_test = pickle.load(f)
        with open(x_eval, 'rb') as f:
            self.x_eval = pickle.load(f)

        tf.logging.info("Size of training set: {}".format(self.x_train.shape))
        tf.logging.info("Size of evaluation set: {}".format(self.x_eval.shape))
        tf.logging.info("Size of test set: {}".format(self.x_test.shape))

        if self.config['general']['embedding_file']:
            embedding_file = os.path.join(HOME, self.config['general']['embedding_file'])

            with open(embedding_file, 'rb') as f:
                self.embds = pickle.load(f)

        self.random_seed = self.config['general']['random_seed']
        self.result_dir = os.path.join(HOME, self.config['general']['result_dir'])
        self.tb_dir = os.path.join(HOME, self.config['general']['tb_dir'])

        inverse_dict = os.path.join(HOME, self.config['general']['inverse_vocab_dict'])
        with open(inverse_dict, 'rb') as f:
            self.inverse_dict = pickle.load(f)

        dict_path = os.path.join(HOME, self.config['general']['vocab_dict'])
        with open(dict_path, 'rb') as f:
            self.vocab_dict = pickle.load(f)

        tf.logging.info('=======')
        tf.logging.info('GENERAL')
        tf.logging.info('=======')
        tf.logging.info('random_seed: {}'.format(self.random_seed))
        tf.logging.info('result_dir: {}'.format(self.result_dir))
        tf.logging.info('tensorboard_dir: {}'.format(self.tb_dir))

        # Regularization features
        self.variational_dropout = self.config['features']['variational_dropout']
        self.tied_weights = self.config['features']['weight_tying']
        self.embed_drop = self.config['features']['embed_drop']
        self.pretrained_embds = self.config['features']['pretrained_embds']

        tf.logging.info('===============')
        tf.logging.info('FEATURES')
        tf.logging.info('===============')
        tf.logging.info('variational dropout?: {}'.format(self.variational_dropout))
        tf.logging.info('weight tying?: {}'.format(self.tied_weights))
        tf.logging.info('embedding dropout?: {}'.format(self.embed_drop))
        tf.logging.info('pretrained embddings?: {}'.format(self.pretrained_embds))

        # Hyperparameters
        self.n_epochs = self.config['hparams']['n_epochs']
        self.clip_norm = self.config['hparams']['clip_norm']
        self.batch_size = self.config['hparams']['batch_size']
        self.init_scale = self.config['hparams']['init_scale']
        self.init_lr = self.config['hparams']['init_lr']
        self.n_hidden = self.config['hparams']['n_hidden']
        self.embd_size = self.config['hparams']['embd_size']
        self.lr_decay = self.config['hparams']['lr_decay']
        self.max_epoch = self.config['hparams']['max_epoch']
        self.vocab_size = self.config['hparams']['vocab_size']

        embd_drop = self.config['hparams']['embedding_dropout']
        inp_drop = self.config['hparams']['input_dropout']
        rec_drop = self.config['hparams']['recurrent_dropout']
        outp_drop = self.config['hparams']['output_dropout']

        tf.logging.info('===============')
        tf.logging.info('HYPERPARAMETERS')
        tf.logging.info('===============')
        tf.logging.info('vocab_size: {}'.format(self.vocab_size))
        tf.logging.info('batch_size: {}'.format(self.batch_size))
        tf.logging.info('clip_norm: {}'.format(self.clip_norm))
        tf.logging.info('n_hidden: {}'.format(self.n_hidden))
        tf.logging.info('embd_size: {}'.format(self.embd_size))
        tf.logging.info('max epochs: {}'.format(self.max_epoch))
        tf.logging.info('decay rate: {}'.format(self.lr_decay))
        tf.logging.info('weight init scale: {}'.format(self.init_scale))
        tf.logging.info('init learning rate: {}'.format(self.init_lr))

        tf.logging.info('embedding_drop: {}'.format(embd_drop))
        tf.logging.info('input_drop: {}'.format(inp_drop))
        tf.logging.info('recurrent_drop: {}'.format(rec_drop))
        tf.logging.info('outp_drop: {}'.format(outp_drop))

        tf.logging.info('======================')
        tf.logging.info('BUILD MODEL')
        tf.logging.info('======================')
        self.graph = self.build_graph(tf.Graph())

        # Uncomment the lines below if random search is used

        # tf.logging.info("RANDOM HYPERPARAMETERS")
        # tf.logging.info('n_hidden: {}'.format(self.n_hidden))
        # tf.logging.info('embd_size: {}'.format(self.embd_size))
        # tf.logging.info('decay rate: {}'.format(self.lr_decay))
        # tf.logging.info('max epochs: {}'.format(self.max_epoch))

        # tf.logging.info("RANDOM DROPOUT AND GRADIENT CLIPPING")
        # tf.logging.info('embedding_keep_prob: {}'.format(self.rand_embd_drop))
        # tf.logging.info('input_keep_prob: {}'.format(self.rand_inp_drop))
        # tf.logging.info('recurrent_keep_prob: {}'.format(self.rand_rec_drop))
        # tf.logging.info('outp_keep_prob: {}'.format(self.rand_outp_drop))
        # tf.logging.info('clipping norm: {}'.format(self.clip_norm))

        with self.graph.as_default():
            self.saver = tf.train.Saver(
                max_to_keep=6, )
            self.init_op = tf.global_variables_initializer()

            sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            self.sess = tf.Session(config=sess_config, graph=self.graph)

            self.init()

            # Use two summary writer to monitor performance on train and validation set
            self.sw_train = tf.summary.FileWriter(self.tb_dir+'/train', self.graph)
            self.sw_eval = tf.summary.FileWriter(self.tb_dir+'/evaluate')

    @property
    def initial_state(self):
        return self.init_state

    def _create_summaries(self):
        """
        Adds summaries for visualization in TensorBoard
        """
        with tf.name_scope("summaries"):
            self.acc_placeholder = tf.placeholder(tf.float32, shape=())
            self.perplex_placeholder = tf.placeholder(tf.float32, shape=())
            self.loss_placeholder = tf.placeholder(tf.float32, shape=())
            tf.summary.scalar('accuracy', self.acc_placeholder)
            tf.summary.scalar('perplexity', self.perplex_placeholder)
            tf.summary.scalar('loss', self.loss_placeholder)
            self.merged = tf.summary.merge_all()


    def build_graph(self, graph):
        """
        Builds the LSTM language model graph
        """
        with graph.as_default():
            tf.set_random_seed(self.random_seed)

            # Random search round one:
            #self.embd_size, self.n_hidden, self.lr_decay, self.max_epoch = self.random_hyperparameters_part1()
            # Random search round two:
            # self.rand_embd_drop, self.rand_inp_drop, self.rand_rec_drop, self.rand_outp_drop, self.clip_norm = self.random_hyperparameters_part2()

            with tf.name_scope("learning_rate"):
                self._lr = tf.Variable(0.0, trainable=False)

            with tf.variable_scope("initializer"):
                self.initializer = tf.random_uniform_initializer(-self.init_scale, self.init_scale)

            if self.pretrained_embds == False:
                with tf.variable_scope('embedding'):
                    # random embeddings
                    self.embedding_matrix = tf.get_variable( "embedding", shape=[self.vocab_size, self.embd_size], dtype=tf.float32, initializer=self.initializer)

            if self.pretrained_embds == True:
                assert self.embds is not None, "You must provide an embedding file!"
                with tf.variable_scope('embedding'):
                    # pre-trained embeddings
                    self.embedding_matrix = tf.get_variable(name='embedding', shape=self.embds.shape, initializer=tf.constant_initializer(self.embds), trainable=True)

            with tf.name_scope("dropout_rates"):
                self.embedding_dropout = tf.placeholder(tf.float32)
                self.input_dropout = tf.placeholder(tf.float32)
                self.recurrent_dropout = tf.placeholder(tf.float32)
                self.output_dropout = tf.placeholder(tf.float32)

            with tf.name_scope('seq_length'):
                self.seq_length = tf.placeholder(tf.int32, shape=())

            if self.embed_drop == True:
                with tf.name_scope("embedding_dropout"):
                    self.embedding_matrix = tf.nn.dropout(self.embedding_matrix, keep_prob=self.embedding_dropout, noise_shape=[self.vocab_size,1])

            with tf.name_scope('input'):
                self.input_batch = tf.placeholder(tf.int64, shape=(None, None))
                self.inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.input_batch)

            with tf.name_scope("labels"):
                self.label_batch = tf.placeholder(tf.int64, shape=(None, None))


            if self.variational_dropout == False:
                with tf.name_scope('rnn'):
                    # To allow for independent embedding and hidden size, the first
                    # and last LSTM layers have input/output dimensionality equal to
                    # embedding size. See thesis for more details.
                    cells = []

                    with tf.name_scope("cell_1"):
                        cell1 = tf.contrib.rnn.LSTMCell(self.embd_size, state_is_tuple=True, initializer=self.initializer)
                        cell1 = tf.contrib.rnn.DropoutWrapper(cell1,
                                    input_keep_prob=self.input_dropout,
                                    output_keep_prob=self.output_dropout,
                                    state_keep_prob=self.recurrent_dropout)
                        cells.append(cell1)

                    with tf.name_scope("cell_2"):
                        cell2 = tf.contrib.rnn.LSTMCell(self.n_hidden, state_is_tuple=True, initializer=self.initializer)
                        # cell has no input dropout since previous cell already has output dropout
                        cell2 = tf.contrib.rnn.DropoutWrapper(cell2,
                                    output_keep_prob=self.output_dropout,
                                    state_keep_prob=self.recurrent_dropout)
                        cells.append(cell2)

                    with tf.name_scope("cell_3"):
                        cell3 = tf.contrib.rnn.LSTMCell(self.embd_size, state_is_tuple=True, initializer=self.initializer)
                        # cell has no input dropout since previous cell already has output dropout
                        cell3 = tf.contrib.rnn.DropoutWrapper(cell3,
                                    output_keep_prob=self.output_dropout,
                                    state_keep_prob=self.recurrent_dropout)
                        cells.append(cell3)

            if self.variational_dropout == True:
                with tf.name_scope('rnn'):
                    cells = []
                    with tf.name_scope("cell_1"):
                        cell1 = tf.contrib.rnn.LSTMCell(self.embd_size, state_is_tuple=True, initializer=self.initializer)
                        cell1 = tf.contrib.rnn.DropoutWrapper(cell1,
                                    input_keep_prob=self.input_dropout,
                                    output_keep_prob=self.output_dropout,
                                    state_keep_prob=self.recurrent_dropout,
                                    variational_recurrent=True, dtype=tf.float32,
                                    input_size=self.embd_size)
                        cells.append(cell1)

                    with tf.name_scope("cell_2"):
                        cell2 = tf.contrib.rnn.LSTMCell(self.n_hidden, state_is_tuple=True, initializer=self.initializer)
                        cell2 = tf.contrib.rnn.DropoutWrapper(cell2,
                                    output_keep_prob=self.output_dropout,
                                    state_keep_prob=self.recurrent_dropout,
                                    input_size=self.n_hidden,
                                    variational_recurrent=True, dtype=tf.float32)
                        cells.append(cell2)

                    with tf.name_scope("cell_3"):
                        cell3 = tf.contrib.rnn.LSTMCell(self.embd_size, state_is_tuple=True, initializer=self.initializer)
                        cell3 = tf.contrib.rnn.DropoutWrapper(cell3,
                                    output_keep_prob=self.output_dropout,
                                    state_keep_prob=self.recurrent_dropout,
                                    input_size=self.n_hidden,
                                    variational_recurrent=True, dtype=tf.float32)
                        cells.append(cell3)


            cell = tf.contrib.rnn.MultiRNNCell(
                cells, state_is_tuple=True)

            # Create a zero-filled state tensor as an initial state
            with tf.name_scope("init_state"):
                self.init_state = cell.zero_state(self.batch_size, tf.float32)

            output, self.final_state = tf.nn.dynamic_rnn(
                cell,
                inputs=self.inputs,
                initial_state=self.init_state)

            with tf.name_scope("lstm_weights"):
                # Add tensorboard visualization for LSTM weights
                for i in range(len(cell.variables)//2):
                   weights, bias = cell.variables[i], cell.variables[i+1]
                   tf.summary.histogram("lstm_weights", weights)
                   tf.summary.histogram("lstm_bias", bias)

            with tf.name_scope("flat_outputs"):
                self.output_flat = tf.reshape(output, [-1, cell.output_size])

            if self.tied_weights == True:
                # tie input embedding weights to output embedding weights
                with tf.variable_scope("embedding", reuse=True):
                    self.softmax_w = tf.transpose(tf.get_variable('embedding'))

                # Set output bias vector to zero
                softmax_b = tf.zeros(shape=[self.vocab_size], dtype=tf.float32, name="softmax_b")

            if self.tied_weights == False:
                # creating new random weights:
                self.softmax_w = tf.get_variable("softmax_w", [self.embd_size, self.vocab_size], dtype=tf.float32, initializer=self.initializer)
                softmax_b = tf.get_variable("softmax_b", [self.vocab_size], dtype=tf.float32, initializer=self.initializer)

            with tf.name_scope("logits"):
                logits = tf.nn.xw_plus_b(self.output_flat, self.softmax_w, softmax_b)
                # Reshape logits to be a 3-D tensor for sequence loss
                self.logits = tf.reshape(logits, [self.batch_size, self.seq_length, self.vocab_size])

            with tf.name_scope("loss"):
                loss = tf.contrib.seq2seq.sequence_loss(
                    self.logits,
                    self.label_batch,
                    tf.ones([self.batch_size, self.seq_length], dtype=tf.float32),
                    average_across_timesteps=False, average_across_batch=True)

                self.loss = tf.reduce_sum(loss)

            with tf.name_scope('predictions'):
                self.predictions = tf.argmax(self.logits, axis=2)

            with tf.name_scope("accuracy"):
                correct_predictions = tf.to_float(tf.equal(self.label_batch, self.predictions))
                self.accuracy = tf.reduce_mean(correct_predictions)

            with tf.name_scope('train'):
                # Create a global step variable
                self.global_step = tf.Variable(
                    0,
                    trainable=False,
                    name="global_step",
                    collections=[ tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES ])

                # Get all variables created with trainable=True
                parameters = tf.trainable_variables()
                # Compute the gradient of the loss w.r.t to the params
                gradients = tf.gradients(self.loss, parameters)
                # Clip gradients. How this works: Given a tensor t, and a maximum
                # clip value clip_norm the op normalizes t so that its L2-norm is less
                # than or equal to clip_norm
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
                self.optimizer =  tf.train.GradientDescentOptimizer(learning_rate=self._lr)
                # Apply the optimizer
                self.train_step = self.optimizer.apply_gradients(zip(clipped_gradients, parameters), global_step=self.global_step)
                # If not clipping the gradients, minimize the loss directly
                # self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step)

            with tf.name_scope("lr_update"):
                self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_lr")
                self._lr_update = tf.assign(self._lr, self._new_lr)

            self._create_summaries()

        return graph

    def id_to_word(self, word_ids):
        """
        Converts a list of word id's to a string of words
        Params:
            word_ids: list of integer word IDs

        Returns:
            word_sentence: string of words
        """
        words = [self.inverse_dict[word_id] for word_id in word_ids]

        word_sentence = ' '.join(map(str,words))
        return word_sentence

    def train(self, save_every=20):
        """
        Train LSTM language model for the number of epochs specified in the config file.
        The learning rate is decayed after a certain number of epochs, using a linear decay
        rate.

        Training is stopped if the validation perplexity fails to improve for 5 epochs, the
        best checkpoint is kept.

        During training, the average per-word perplexity on the training and validation set
        are monitored and plotted on Tensorboard.
        After training, the model is tested on the test set.
        """

        with self.graph.as_default():
            # Initialize the state of the network
            _current_state = self.sess.run(self.initial_state)
            training_step = 0
            # The next variables are for early stopping
            previous_perplexity = 1000000000
            best_perplexity = 1000000000
            best_step_number = 0
            stop_at_epoch = self.n_epochs
            stop_early = False

            for epoch_id in range(self.n_epochs):
                # early stopping
                if epoch_id > stop_at_epoch:
                   break

                costs = 0.0
                iters = 0

                # Update learning rate
                lr_decay = self.lr_decay ** max(epoch_id+1-self.max_epoch, 0.0)
                new_lr = self.init_lr * lr_decay
                self.sess.run(self._lr_update, feed_dict={self._new_lr: new_lr})

                m, batch_len = self.x_train.shape
                self.n_batches = (batch_len - 1) // 35 # number of batches per epoch

                for batch_number in range(0, self.n_batches):
                    from_index = batch_number*35
                    to_index = (batch_number+1)*35

                    _inputs = self.x_train[:, batch_number*35 : (batch_number + 1) * 35]
                    _labels = self.x_train[:, batch_number*35+1 : (batch_number  + 1) * 35 + 1]

                    training_step += 1

                    _logits, _predictions, _train_step, _current_state, _loss, _acc, _lr = self.sess.run(
                            [self.logits,
                            self.predictions,
                            self.train_step,
                            self.final_state,
                            self.loss,
                            self.accuracy,
                            self._lr],
                            feed_dict={
                                self.embedding_dropout : self.config['hparams']['embedding_dropout'],
                                self.input_dropout : self.config['hparams']['input_dropout'],
                                self.recurrent_dropout : self.config['hparams']['recurrent_dropout'],
                                self.output_dropout : self.config['hparams']['output_dropout'],
                                self.seq_length: 35,
                                # self.embedding_dropout : self.rand_embd_drop,
                                # self.input_dropout : self.rand_inp_drop,
                                # self.recurrent_dropout : self.rand_rec_drop,
                                # self.output_dropout : self.rand_outp_drop,
                                self.input_batch: _inputs,
                                self.label_batch: _labels,
                                self.init_state[0][0]: _current_state[0][0],
                                self.init_state[0][1]: _current_state[0][1],
                                self.init_state[1][0]: _current_state[1][0],
                                self.init_state[1][1]: _current_state[1][1],
                                self.init_state[2][0]: _current_state[2][0],
                                self.init_state[2][1]: _current_state[2][1],
                               })

                    pred = _predictions[0]
                    iters += 35
                    costs += _loss

                    # compute average per-word perplexity
                    perplexity = np.exp(costs/iters)

                    if batch_number % 500 == 0:
                        # Add training summaries
                        summary = self.sess.run(self.merged,
                                feed_dict={
                                    self.acc_placeholder: _acc,
                                    self.loss_placeholder: _loss,
                                    self.perplex_placeholder: perplexity})
                        self.sw_train.add_summary(summary, global_step=training_step)

                        # Add evaluation summaries
                        eval_loss, eval_acc, eval_perplexity = self.evaluate(_current_state)
                        feed_dict ={self.acc_placeholder: eval_acc,
                                    self.loss_placeholder: eval_loss,
                                    self.perplex_placeholder: eval_perplexity}
                        summary = self.sess.run(self.merged,feed_dict=feed_dict)
                        self.sw_eval.add_summary(summary, global_step=training_step)


                eval_loss, eval_acc, eval_perplexity = self.evaluate(_current_state)

                tf.logging.info("Current epoch: {}".format(epoch_id))
                tf.logging.info("Current training step: {}".format(training_step))
                tf.logging.info("Current loss: {}".format(_loss))
                tf.logging.info("Current accuracy: {}".format(_acc))
                tf.logging.info("Current perplexity: {}".format(perplexity))
                tf.logging.info("EVALUATION LOSS: {}".format(eval_loss))
                tf.logging.info("EVALUATION PERPLEXITY: {}".format(eval_perplexity))
                tf.logging.info("Previous eval perplexity: {}".format(previous_perplexity))
                tf.logging.info("LEARNING RATE: {}".format(_lr))
                tf.logging.info("=========================================================================================")

                if (not stop_early and epoch_id > 15 and eval_perplexity >= best_perplexity):
                    tf.logging.info("Perplexity hasn't improved!")
                    tf.logging.info("Activate early Stopping")
                    stop_early = True
                    stop_at_epoch = epoch_id + 5
                    tf.logging.info("Change number of epochs to {}".format(stop_at_epoch))

                if eval_perplexity < best_perplexity:
                    if stop_early:
                        tf.logging.info("Early stopping active but model has improved!")
                        tf.logging.info("Disable early stopping!")
                        stop_at_epoch = self.n_epochs
                        stop_early = False

                    tf.logging.info("New best model!")
                    tf.logging.info("Previous best: {}".format(best_perplexity))
                    tf.logging.info("New best: {}".format(eval_perplexity))
                    tf.logging.info("Save new best")
                    best_perplexity = eval_perplexity
                    best_step_number = training_step
                    tf.logging.info("Save final state and checkpoint")
                    self.save(epoch_id)
                    p = os.path.join(HOME, self.result_dir)
                    np.savetxt(os.path.join(p, 'best_final_state_00.csv'), _current_state[0][0])
                    np.savetxt(os.path.join(p, 'best_final_state_01.csv'), _current_state[0][1])
                    np.savetxt(os.path.join(p, 'best_final_state_10.csv'), _current_state[1][0])
                    np.savetxt(os.path.join(p, 'best_final_state_11.csv'), _current_state[1][1])
                    np.savetxt(os.path.join(p, 'best_final_state_20.csv'), _current_state[2][0])
                    np.savetxt(os.path.join(p, 'best_final_state_21.csv'), _current_state[2][1])
                    self.best_state = _current_state


                previous_perplexity = eval_perplexity
                tf.logging.info("=========================================================================================")
                tf.logging.info("=========================================================================================")
                tf.logging.info("NEW EPOCH")
                tf.logging.info("=========================================================================================")
                tf.logging.info("=========================================================================================")


            tf.logging.info("=========================================================================================")
            test_loss, test_perplexity = self.test(self.best_state)
            tf.logging.info("TEST LOSS: {}".format(test_loss))
            tf.logging.info("TEST PERPLEXITY: {}".format(test_perplexity))
            tf.logging.info("=========================================================================================")

            tf.logging.info("Save random search params and performance")
            tf.logging.info("Order: embd_size, n_hidden, lr_decay, max_epoch, valid_perplex, test_perplex")
            model_params = [self.embd_size, self.n_hidden, self.lr_decay, self.max_epoch, eval_perplexity, test_perplexity]
            save_model_params = os.path.join(HOME, self.result_dir)
            with open(save_model_params+"/model_params_performance", 'wb') as f:
                pickle.dump(model_params, f)


    def evaluate(self, current_state):
        """
        Evaluate LSTM language model on the validation set.

        Args:
            current_state: LSTM state tuple, holding the current network state

        Returns:
            total_loss: average loss on validation set
            total_acc: average accuracy on validation set
            perplexity: average perplexity on validation set
        """

        _current_state = current_state
        m, batch_len = self.x_eval.shape
        eval_n_batches = (batch_len - 1) // 35 # number of batches per epoch

        eval_step = 0
        iters = 0
        costs = 0.0
        all_losses = []
        all_accuracies = []

        for batch_number in range(0, eval_n_batches):
            from_index = batch_number * 35
            to_index = (batch_number + 1) * 35

            # Get next batch of inputs and labels
            _inputs = self.x_eval[:, batch_number*35 : (batch_number + 1) * 35]
            _labels = self.x_eval[:, batch_number*35+1 : (batch_number  + 1) * 35 + 1]

            eval_step += 1

            # Run training step
            # The final state of the net is fed back into the net
            _predictions, _current_state, _loss, _acc = self.sess.run(
                    [self.predictions,
                    self.final_state,
                    self.loss,
                    self.accuracy],
                    feed_dict={
                        self.seq_length: 35,
                        self.embedding_dropout: 1.0,
                        self.input_dropout: 1.0,
                        self.output_dropout: 1.0,
                        self.recurrent_dropout: 1.0,
                        self.input_batch: _inputs,
                        self.label_batch: _labels,
                        self.init_state[0][0]: _current_state[0][0],
                        self.init_state[0][1]: _current_state[0][1],
                        self.init_state[1][0]: _current_state[1][0],
                        self.init_state[1][1]: _current_state[1][1],
                        self.init_state[2][0]: _current_state[2][0],
                        self.init_state[2][1]: _current_state[2][1],
                       })

            all_losses.append(_loss)
            all_accuracies.append(_acc)
            iters += 35
            costs += _loss

        total_loss = np.mean(all_losses)
        total_acc = np.mean(all_accuracies)
	# compute average per-word perplexity
        perplexity = np.exp(costs/iters)

        return total_loss, total_acc, perplexity


    def test(self, best_state):
        """
        Evaluate LSTM language model on the test set.

        Args:
            current_state: LSTM state tuple, holding the current network state

        Returns:
            total_loss: average loss on test set
            perplexity: average perplexity on test set
        """
        #_current_state = self.sess.run(self.initial_state)
        _current_state = best_state

        m, batch_len = self.x_test.shape
        test_n_batches = (batch_len - 1) // 35 # number of batches per epoch

        iters = 0
        costs = 0.0
        all_losses =[]

        for batch_number in range(0, test_n_batches):
            from_index = batch_number * 35
            to_index = (batch_number + 1) * 35

            # Get next batch of inputs and labels
            _inputs = self.x_test[:, batch_number*35 : (batch_number + 1) * 35]
            _labels = self.x_test[:, batch_number*35+1 : (batch_number  + 1) * 35+1]

            _predictions, _current_state, _loss, _acc = self.sess.run(
                    [self.predictions,
                    self.final_state,
                    self.loss,
                    self.accuracy],
                    feed_dict={
                        self.seq_length: 35,
                        self.embedding_dropout: 1.0,
                        self.input_dropout: 1.0,
                        self.output_dropout: 1.0,
                        self.recurrent_dropout: 1.0,
                        self.input_batch: _inputs,
                        self.label_batch: _labels,
                        self.init_state[0][0]: _current_state[0][0],
                        self.init_state[0][1]: _current_state[0][1],
                        self.init_state[1][0]: _current_state[1][0],
                        self.init_state[1][1]: _current_state[1][1],
                        self.init_state[2][0]: _current_state[2][0],
                        self.init_state[2][1]: _current_state[2][1],
                       })

            all_losses.append(_loss)
            iters += 35
            costs += _loss

        total_loss = np.mean(all_losses)
        perplexity = np.exp(costs/iters)

        tf.logging.info("=========================================================================================")
        tf.logging.info("TEST LOSS: {}".format(total_loss))
        tf.logging.info("TEST PERPLEXITY: {}".format(perplexity))
        tf.logging.info("=========================================================================================")

        return total_loss, perplexity

    def word_to_id(self, words):
        words = ['eofs' if w == '.' else w if w in self.dict else 'rare' for w in words ]
        words = [self.dict[w] for w in words]

        return words


    def predict(self, primer_words, n_suggestions, len_suggestions):
        """
        Given a primer sequence of words, generate new words

        Args:
            primer_words: primer words, given as a list of strings
            n_suggestions: number of suggestions
            len_suggestions: number of words that should be predicted
        """

        _current_state = self.sess.run(self.initial_state)

        # Map primer words to id
        primer_words = self.word_to_id(primer_words)

        steps = 0
        primer_words = np.tile(primer_words, (20,1))

        for _ in range(len_suggestions):

            # If the input sequence is very long we consider only
            # the last 20 words. Otherwise prediction will be slow
            if len(primer_words) > 20:
                primer_words = primer_words[-20:]

            seq_length = int(len(primer_words[0, :]))

            _logits, _current_state = self.sess.run(
                    [self.logits, self.final_state],
                    feed_dict={
                        self.seq_length: seq_length,
                        self.input_batch: primer_words,
                        self.embedding_dropout: 1.0,
                        self.input_dropout: 1.0,
                        self.output_dropout: 1.0,
                        self.recurrent_dropout: 1.0,
                        self.init_state[0][0]: _current_state[0][0],
                        self.init_state[0][1]: _current_state[0][1],
                        self.init_state[1][0]: _current_state[1][0],
                        self.init_state[1][1]: _current_state[1][1],
                        self.init_state[2][0]: _current_state[2][0],
                        self.init_state[2][1]: _current_state[2][1]})

            steps += 1

            # Get the n most likely next words
            n = 5
            # We need only the predictions for the last time step
            # So we sort only that part of the array
            with tf.Session() as sess:
                softmax = sess.run(tf.nn.softmax(_logits[:, -1, :]))

            # ipdb.set_trace()
            arr_sorted = _logits[:, -1, :].argsort(axis=1)
            # Get the indices of the n largest elements
            max_inds = arr_sorted[:, -n:]
            # Get softmax probabilities of the n best words
            probs = softmax[0, max_inds[0]]
            total = sum(probs)
            p5 = probs[0] / total
            p4 = probs[1] / total
            p3 = probs[2] / total
            p2 = probs[3] / total
            p1 = probs[4] / total

            selected_words = [np.random.choice([-1, -2, -3, -4, -5], p=[p1, p2, p3, p4, p5]) for _ in range(20)]

            best_word = max_inds[0, selected_words]

            # Append the new word
            primer_words = np.append(primer_words, best_word[:, np.newaxis], axis=1)

        # Transform the word id's into real words
        words = primer_words[:n_suggestions]
        word_list = words.tolist()

        # Map back to words
        result = [[self.inverse_dict[w] for w in s] for s in word_list]
        result = [' '.join(map(str, s)) for s in result]
        result = [re.sub(' eofs', '.', s) for s in result]

        return result

    def random_hyperparameters_part1(self):
        """
        Generates random hyperparameters for fixing the general
        model architecture

        ranges:
            embedding_size: 100-500
            n_hidden: 200-1500
            decay_rate: 0.5-0.9
            max_epoch: 5-15
        """
        rand_embedding_size = round(np.random.randint(low=100, high=501), -1)
        rand_n_hidden = round(np.random.randint(low=200, high=1501), -2)
        rand_decay_rate = round(random.uniform(0.5, 0.9), 2)
        rand_max_epoch = np.random.randint(low=5, high=16)

        return rand_embedding_size, rand_n_hidden, rand_decay_rate, rand_max_epoch


    def random_hyperparameters_part2(self):
        """
        Generates random dropout and gradient
        clipping hyperparameters

        ranges:
            dropout keep probabilitites: 0.5-1.0
            clip_norm: 5-10

        Returns:
            rand_embd_drop: keep_prob for embeddings
            rand_inp_drop: keep_prob for inputs of LSTM units
            rand_rec_drop: keep_prob for hidden-to-hidden connections of LSTM units
            rand_outp_drop: keep_prob for outputs of LSTM units
            rand_clip_norm: global norm for clipping the gradients
        """
        embedding_drop = round(random.uniform(0.5, 1.0), 2)
        input_drop = round(random.uniform(0.5, 1.0), 2)
        recurrent_drop = round(random.uniform(0.5, 1.0), 2)
        output_drop = round(random.uniform(0.5, 1.0), 2)
        clip_norm = np.random.randint(low=5, high=11)

        return embedding_drop, input_drop, recurrent_drop, output_drop, clip_norm


    def save(self, epoch_id):
        global_step_t = tf.train.get_global_step(self.graph)
        global_step = self.sess.run(global_step_t)

        tf.logging.info('Saving to {} with global step {}'.format(
            self.result_dir, global_step))
        save_name = 'model-ep_{}-{}.ckpt'.format(epoch_id, global_step)
        self.saver.save(self.sess, os.path.join(self.result_dir, save_name))


    def init(self):
        checkpoint = tf.train.get_checkpoint_state(self.result_dir)

        if checkpoint is None:
            self.sess.run(self.init_op)
        else:
            tf.logging.info(
                'Loading the model from: {}'.format(self.result_dir))
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
